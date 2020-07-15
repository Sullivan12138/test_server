import tensorflow as tf
import time
import yaml
import numpy as np

from . import globalvar
from .predict import lstm
from math import ceil
from .prome import *

minReplicas = 3
maxReplicas = 4
scaleIntervalMins = 10
averageUtilization = 4  # 4/8

history_data_tidb_cpu = []
history_data_tidb_disk = []


# 将yaml文件中的tikv副本数和tikv cpu使用率阈值读取出来
def yaml_to_dict(yaml_path):
    with open(yaml_path, "r") as test_file:
        generate_dict = yaml.load(test_file, Loader=yaml.FullLoader)  # 先将yaml转换为dict格式
        # generate_json = json.dumps(generate_dict,sort_keys=False,indent=4,separators=(',',': '))
        init_tidb_replicas = int(generate_dict['spec']['tidb']['replicas'])
        cpu_str = generate_dict['spec']['tidb']['limits']['cpu']
        if cpu_str[-1] == 'm':
            cpu_request = float(cpu_str[0:-1]) / 1000
        else:
            cpu_request = float(cpu_str)
        print('cpu_request:', cpu_request)
        return cpu_request, init_tidb_replicas


# 按照time_step生成预测数据,实际中得不到预测的10分钟后的数据
def get_predict_data(time_step):
    data_test = history_data_tidb_cpu[-time_step:]
    # print(data_test)
    max_num = np.max(data_test, axis=0)
    normalized_test_data = data_test

    # print('get_test_data x', np.array(normalized_test_data).shape)
    return max_num, normalized_test_data


# 按照batch_size和time_step生成训练数据,标签是10分钟之后的数据
def get_train_data(batch_size, time_step, predict_step):
    max_num = np.max(history_data_tidb_cpu, axis=0)
    normalized_train_data = history_data_tidb_cpu

    train_x, train_y = [], []  # 训练集
    for i in range(len(normalized_train_data) - time_step - (predict_step - 1)):
        x = normalized_train_data[i:i + time_step]
        y = normalized_train_data[i + time_step + (predict_step - 1)]
        train_x.append(x.tolist())
        train_y.append(y.tolist())
    return train_x, train_y


# 从Prometheus获取cpu数据,把data整合成一行输入,使用全局变量
def get_cpu_input(interval, prome_addr):
    global history_data_tidb_cpu

    step = 60
    end = str(int(time.time()))

    if interval == 0:
        # 只获取前1分钟的数据
        sum = 0
        start = str(int(time.time()))
        cpu_data = fetch_tikv_cpu_usage(prome_addr, start, end, step)
        # 求出所有实例的总metrics
        for ins in cpu_data:
            sum += float(ins['values'][0][1])
        history_data_tidb_cpu[0:-1] = history_data_tidb_cpu[1:]  # 丢掉最早的时刻
        history_data_tidb_cpu[-1][0] = sum
    else:
        # 获取前interval分钟的数据
        start = str(int(time.time()) - 60 * (interval - 1))
        cpu_data = fetch_tikv_cpu_usage(prome_addr, start, end, step)
        for i in range(-interval, 0):
            for ins in cpu_data:
                history_data_tidb_cpu[i][0] += float(ins['values'][i][1])


# 从Prometheus获取磁盘io数据
def get_disk_input(interval, prome_addr):
    global history_data_tidb_disk

    step = 60
    end = str(int(time.time()))

    if interval == 0:
        sum = 0
        start = str(int(time.time()))
        cpu_data = fetch_io_utilization(prome_addr, start, end, step)
        for ins in cpu_data:
            sum += float(ins['values'][0][1])
        history_data_tidb_disk[0:-1] = history_data_tidb_disk[1:]  # 丢掉最早的时刻
        history_data_tidb_disk[-1][0] = sum
    else:
        start = str(int(time.time()) - 60 * (interval - 1))
        cpu_data = fetch_io_utilization(prome_addr, start, end, step)
        for i in range(-interval, 0):
            for ins in cpu_data:
                history_data_tidb_disk[i][0] += float(ins['values'][i][1])


def train_lstm(input_size, output_size, lr, rnn_unit, weights, biases, batch_size, time_step, kp,
               predict_step, save_model_path, save_model_name, init_tikv_replicas, yaml_path, prome_addr, refer_data):

    X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
    Y = tf.placeholder(tf.float32, shape=[None, output_size])
    keep_prob = tf.placeholder('float')
    pred, _, m, mm = lstm(X, weights, biases, input_size, rnn_unit, keep_prob)
    # 损失函数
    loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1, output_size]) - tf.reshape(Y, [-1, output_size])))
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)

    saver = tf.train.Saver(max_to_keep=1)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(save_model_path)  # checkpoint存在的目录
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)  # 自动恢复model_checkpoint_path保存模型一般是最新
            print("Model restored...")
        else:
            print('No Model')

        last_scale_time = -scaleIntervalMins
        current_replicas = init_tikv_replicas
        label = 0  # 当前的序号
        train_num = 0  # 累积多久训练一次
        while True:  # 每个预测就是一次循环，每隔一段时间就训练一次，存储模型
            # 预测
            if label > 1000:  # 值没有意义，只是运行一段时间后不需要用到label,
                pass
            else:
                label += 1
            if label == 1:
                get_cpu_input(time_step, prome_addr)
            else:
                get_cpu_input(0, prome_addr)

            # 预测
            max_num, test_x = get_predict_data(time_step)
            prob = sess.run(pred, feed_dict={X: [test_x], keep_prob: 1})
            predict = prob.reshape((-1))
            for i in range(output_size):
                # predict[i] = predict[i] * max_num[i]

                # 计算期望副本数
                pre_replicas = ceil(predict[i] / averageUtilization)
                if pre_replicas > maxReplicas:
                    pre_replicas = maxReplicas
                if pre_replicas < minReplicas:
                    pre_replicas = minReplicas

                # 负载下降时不用提前减少节点
                if pre_replicas < current_replicas :
                    pre_replicas = ceil(history_data_tidb_cpu[-1] / averageUtilization)

                current_replicas = pre_replicas
                refer_data['recommendedReplicas'] = pre_replicas
                print("label_cpu:%d, time:%s, predict_step:%d, predict_tikv_cpu_usage:%f, predict_tikv_replicas:%d"
                      % (label, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), predict_step, predict[i], pre_replicas))

                '''
                # 判断是否调度，考虑调度间隔时间等
                if (label - last_scale_time >= scaleIntervalMins) and (pre_replicas != current_replicas) :
                    #修改yaml配置文件
                    generate_dict = {}
                    with open(yaml_path, "r") as yaml_file:
                        generate_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)  # 先将yaml转换为dict格式
                    with open(yaml_path, "w") as yaml_file:
                        generate_dict['spec']['tikv']['replicas'] = pre_replicas
                        yaml.dump(generate_dict, yaml_file)
                    #执行shell命令
                    exitcode, output = subprocess.getstatusoutput("kubectl apply -f %s -n pd-team-s2" % yaml_path)
                    print("exitcode: ", exitcode)
                    print("output: ", output)
                    if exitcode == 0:
                        print("Execute scaling command, tikv_replicas:%d -> %d" % (current_replicas, pre_replicas))
                        last_scale_time = label
                        current_replicas = pre_replicas
                '''

            if label >= predict_step:
                train_num += 1
            # 训练
            if label >= batch_size+predict_step-1 and train_num == batch_size:
                # batch_size=30, time_step=30, predict_step=10,第一次训练[0:30]->[39]-[29:59]->[68],这时label=69
                # x:(batch_size, time_step, input_size)y:(batch_size, output_size)
                train_x, train_y = get_train_data(batch_size, time_step,predict_step)
                # train_y = np.array(train_y)[:, 1:input_size].tolist()  # 如果输入加上时间维度，这里就需要加上
                _, loss_, M, MM = sess.run([train_op, loss, m, mm], feed_dict={X: train_x,Y: train_y,keep_prob: kp})
                print('label_cpu,loss: ', label, loss_)
                train_num = 0
                saver.save(sess, save_model_path + save_model_name)

            time.sleep(60)  # 每隔60s循环一次，实际上因为程序运行所以超过一分钟


def start_predict_cpu(name, namespace, prome_addr, yaml_path):
    input_size = 1  # 输入维度
    output_size = 1  # 输出维度
    rnn_unit = 12  # 隐藏层节点
    lr = 0.004  # 学习率
    batch_size = 20  # 每次训练的一个批次的大小   30
    time_step = 15  # 前time_step步来预测下一步  20
    predict_step = 5  # 预测predict_step分钟后的负载
    kp = 1  # dropout保留节点的比例
    save_model_path = './save/predict_cpu_5-26/'  # checkpoint存在的目录
    save_model_name = 'MyModel'  # saver.save(sess, './save/MyModel') 保存模型
    refer_data = globalvar.get_tikv_replicas()
    refer_data['name'] = name
    refer_data['namespace'] = namespace

    # prome_addr = '10.233.18.170:9090'  # monitor-prometheus的ip
    # yaml_path = '/data2/hust_tmp/cluster/tidb-cluster.yaml' #读配置文件

    cpu_request, init_tikv_replicas = yaml_to_dict(yaml_path)  # 读配置的limits，和初始的tikv的replicas

    global history_data_tidb_cpu
    history_data_tidb_cpu = np.zeros((batch_size + time_step + predict_step - 1, input_size))

    # ——————————————————定义神经网络变量——————————————————
    # 如果是加载已训练好的模型，w和b应该是相同的形状
    weights = {
        'in': tf.Variable(tf.random_uniform([input_size, rnn_unit])),  # max_val=0.125
        'out': tf.Variable(tf.random_uniform([rnn_unit, output_size]))
    }
    biases = {
        'in': tf.Variable(tf.constant(0.1, shape=[rnn_unit, ])),
        'out': tf.Variable(tf.constant(0.1, shape=[output_size, ]))
    }
    train_lstm(input_size, output_size, lr, rnn_unit, weights, biases, batch_size,
               time_step, kp, predict_step, save_model_path, save_model_name,
               init_tikv_replicas, yaml_path, prome_addr, refer_data)