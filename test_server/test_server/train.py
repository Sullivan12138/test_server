"""
用于训练模型的代码
训练ops
训练完了以后保存模型
以及用于预测的代码
"""
import tensorflow as tf
import numpy as np
import time
import sys
from .workload_data import load_history_workload
from .variables import *
from .globalvar import get_tikv_replicas
from .fetch_prome_metrics import *
import json
import yaml


def yaml_to_dict(yaml_path):
    with open(yaml_path, "r") as f:
        generate_dict = yaml.load(f, Loader=yaml.FullLoader)
        init_tikv_replicas = int(generate_dict['spec']['tikv']['replicas'])
        return init_tikv_replicas


"""
本函数以及下面的三个函数是parse_requests函数的组成部分，代表数据解析的四个部分
本函数代表获取tikv msg的qps中的coprocessor部分
"""
def get_kv_cop(kv_grpc_msg_qps):
    res = []
    for data in kv_grpc_msg_qps:
        sum = 0
        for metric in data:
            if metric['metric']['type'] == 'coprocessor':
                sum += float(metric['values'][-1][1])
        res.append(sum)
    return res


"""本函数代表将得到的数据对每个tikv实例进行求和，因为我们使用的是总和进行预测"""
def sum_up(statement_ops):
    res = []
    for data in statement_ops:
        sum = 0
        for metric in data:
            if metric['metric']['type'] != 'Use':
                sum += float(metric['values'][-1][1])
        sum_data = [sum]
        res.append(sum_data)
    return res


"""本函数的作用是确定负载中每种类型的命令所占比例"""
def determine_workload_proportion(statement_ops, sum_data, kv_cop):
    proportions = []
    for i in range(len(statement_ops)):
        proportion = [0. for _ in range(4)]
        for metric in statement_ops[i]:
            if sum_data[i][0] == 0.:
                continue
            if metric['metric']['type'] == 'Update':
                proportion[2] = float(metric['values'][-1][1]) / sum_data[i][0]
            elif metric['metric']['type'] == 'Insert':
                proportion[3] = float(metric['values'][-1][1]) / sum_data[i][0]
            elif metric['metric']['type'] == 'Select':
                # 由于read命令和scan命令在ops上的体现都是select，所以我结合tikv msg的qps来确定比例
                # scan命令会独一无二地产生一种名为coprocessor的tikv msg，并且这个tikv msg的qps和scan的ops成正比
                # 所以知道coprocessor的qps即可知道scan的ops，从而知道read的ops
                scan_qps = kv_cop[i] * scan_per_cop[scan_per_cop_level]
                proportion[1] = float(scan_qps / sum_data[i][0])
                proportion[0] = (float(metric['values'][-1][1]) - scan_qps) / sum_data[i][0]
        proportions.append(proportion)
    with open('p.json', 'w') as f:
        json.dump(proportions, f)
    return proportions


"""
本函数用于根据上述求得的比例以及每种命令的权重求出综合负载
权重由其他组员通过另外的程序得到
"""
def cal_weights(sum_data, proportions, weight):
    for i in range(len(sum_data)):
        sum_data[i][0] = sum_data[i][0] * (proportions[i][0] * weight[0] + proportions[i][1] * weight[1] + proportions[i][2] *
                                           weight[2] + proportions[i][3] * weight[3])
    return sum_data


"""本函数是对输入的requests数据进行解析，使其符合神经网络训练的数据结构"""
def parse_requests(statement_ops, kv_grpc_msg_qps):
    kv_cop = get_kv_cop(kv_grpc_msg_qps)
    sum_data = sum_up(statement_ops)
    proportions = determine_workload_proportion(statement_ops, sum_data, kv_cop)
    new_input_data = cal_weights(sum_data, proportions, weight)
    return new_input_data


"""
本函数以及下面三个函数是函数parse_predict的组成成分，代表parse_predict的四个部分
本函数以及下面三个函数的作用与上述的parse_requests的四个组成成分函数类似
"""
def get_kv_cop_predict(kv_grpc_msg_qps):
    length = len(kv_grpc_msg_qps[0]['values'])
    sum = [0. for _ in range(length)]
    for metric in kv_grpc_msg_qps:
        if metric['metric']['type'] == 'coprocessor':
            for i in range(len(metric['values'])):
                sum[i] += float(metric['values'][i][1])
    return sum


def sum_up_predict(statement_ops):
    length = len(statement_ops[0]['values'])
    sum = [0. for _ in range(length)]
    for metric in statement_ops:
        if metric['metric']['type'] != 'use':
            for i in range(len(metric['values'])):
                sum[i] += float(metric['values'][i][1])
    return sum


def determine_workload_proportion_predict(statement_ops, sum_data, kv_cop):
    length = len(statement_ops[0]['values'])
    proportions = [[0., 0., 0., 0.] for _ in range(length)]
    for metric in statement_ops:
        for i in range(len(metric['values'])):
            if sum_data[i] == 0.:
                continue
            if metric['metric']['type'] == 'Update':
                proportions[i][2] = float(metric['values'][i][1]) / sum_data[i]
            elif metric['metric']['type'] == 'Insert':
                proportions[i][3] = float(metric['values'][i][1]) / sum_data[i]
            elif metric['metric']['type'] == 'Select':
                scan_ops = kv_cop[i] * scan_per_cop[scan_per_cop_level]
                proportions[i][1] = scan_ops / sum_data[i]
                proportions[i][0] = (float(metric['values'][i][1]) - scan_ops) / sum_data[i]
    return proportions


def cal_weights_predict(sum_data, proportions, weight):
    for i in range(len(sum_data)):
        sum_data[i] = sum_data[i] * (proportions[i][0] * weight[0] + proportions[i][1] * weight[1] + proportions[i][2] *
                                     weight[2] + proportions[i][3] * weight[3])
    return sum_data


"""
本函数和上述parse_requests作用类似，但是是作用于预测部分的输入的
预测部分的输入和训练部分的输入数据结构稍有不同，所以要用两种函数来写
"""
def parse_predict(statement_ops, kv_grpc_msg_qps):
    kv_cop = get_kv_cop_predict(kv_grpc_msg_qps)
    sum_data = sum_up_predict(statement_ops)
    proportions = determine_workload_proportion_predict(statement_ops, sum_data, kv_cop)
    new_input_data = cal_weights_predict(sum_data, proportions, weight)
    return new_input_data


# 获取训练集
def get_train_data(data, batch_size, time_step, train_end, predict_step, train_begin=0):
    batch_index = []
    data_train = data[train_begin:train_end]
    data_train = np.array(data_train)
    normalized_train_data = (data_train - np.mean(data_train, axis=0)) / np.std(data_train, axis=0)  # 标准化
    print("len(data_train): ", len(data_train))
    normalized_train_data = normalized_train_data.tolist()

    train_x, train_y = [], []  # 训练集
    for i in range(len(normalized_train_data) - time_step - (predict_step - 1)):
        if i % batch_size == 0:
            batch_index.append(i)
        x = normalized_train_data[i:i + time_step]
        y = normalized_train_data[i + time_step + (predict_step - 1)]
        train_x.append(x)
        train_y.append(y)
    batch_index.append((len(normalized_train_data) - time_step - (predict_step - 1)))
    return batch_index, train_x, train_y


def get_test_data():
    test_x = []
    # 对当前时刻过去的ops的采集，这样才能预测未来。
    # 每组采集过去time_step分钟内的ops，作为预测的x值。
    # 采集了predict_step组，因为每组只能预测一分钟的数据，所以一共
    # 可以用于预测从现在到未来predict_step分钟后的数据。
    # test_x.shape:[predict_step, time_step]
    for i in range(predict_step):
        end = str(int(time.time()) - (predict_step - 1 - i) * 60)
        start = str(int(time.time()) - (predict_step + time_step - 2 - i) * 60)
        statement_ops = fetch_statement_ops(start, end, 60)
        kv_grpc_msg_qps = fetch_kv_grpc_msg_qps(start, end, 60)
        sum_data = parse_predict(statement_ops, kv_grpc_msg_qps)
        res = []
        for input_data in sum_data:
            data = [input_data]
            res.append(data)
        res = np.array(res)
        mean_x = np.mean(res, axis=0)
        std_x = np.std(res, axis=0)
        normalized_test_data = (res - mean_x) / std_x # 将原数据标准化，否则预测误差会很大
        normalized_test_data = normalized_test_data.tolist()
        test_x.append(normalized_test_data)
    return test_x, mean_x, std_x


# 获取当前的statement_ops，为了计算斜率
def get_test_y():
    end = str(int(time.time()))
    start = end
    statement_ops = fetch_statement_ops(start, end, 60)
    kv_grpc_msg_qps = fetch_kv_grpc_msg_qps(start, end, 60)
    test_y = parse_predict(statement_ops, kv_grpc_msg_qps)
    test_y = np.array(test_y)
    normalized_test_y = (test_y - np.mean(test_y, axis=0)) / np.std(test_y, axis=0)
    normalized_test_y = normalized_test_y.tolist()
    return normalized_test_y


def predict(test_x, save_model_path, weights, biases):
    X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
    keep_prob = tf.placeholder('float')
    pred, _, _, _ = lstm(X, input_size, rnn_unit, kp, weights, biases)

    saver = tf.train.Saver(max_to_keep=1)
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(save_model_path)
        if ckpt and ckpt.model_checkpoint_path: # 此处pylint报错：ckpt没有model_checkpoint_path，但是代码依然能正常运行
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Model Restored.')
        else:
            print('No Model!')

        # 预测
        test_predict = []
        for step in range(len(test_x)):
            prob = sess.run(pred, feed_dict={X: [test_x[step]], keep_prob: kp})
            predict = prob.reshape((-1))
            test_predict.extend(predict)
        print('test_predict:', np.array(test_predict).shape)
        return test_predict
        # print('true_data:', np.array(test_y).shape)
        # test_predict = np.array(test_predict).reshape(-1, output_size)
        # print('test_predict:', test_predict.shape)
        # print('test_y:', np.array(test_y).shape)
        #
        # test_y = np.array(test_y)
        #
        # y = [[] for _ in range(output_size)]
        # Y = [[] for _ in range(output_size)]
        # test_predict = np.array(test_predict).reshape(-1, len(test_predict)).tolist()
        # test_y = np.array(test_y).reshape(-1, len(test_y)).tolist()
        # for i in range(output_size):
        #     y[i] = test_y[i]
        #     Y[i] = test_predict[i]
        # file1 = 'y.json'
        # file2 = 'y2.json'
        # with open(file1, 'w') as f:
        #     json.dump(y[0], f, indent=4)
        # with open(file2, 'w') as f:
        #     json.dump(Y[0], f, indent=4)


# ——————————————————定义网络——————————————————
def lstm(X, input_size, rnn_unit, keep_prob, weights, biases):
    # X是一个三维tensor
    batch_size = tf.shape(X)[0]
    time_step = tf.shape(X)[1]
    w_in = weights['in']
    b_in = biases['in']

    input = tf.reshape(X, [-1, input_size])  # 需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    # TODO 一个bug
    input_rnn = tf.matmul(input, w_in) + b_in
    input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_unit])  # 将tensor转成3维，作为lstm cell的输入
    cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)  # reuse = sign
    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    # output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
    output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state,
                                                 dtype=tf.float32)
    # output_rnn是一个tensor，维度和input_rnn一样
    m = output_rnn
    output = tf.reshape(output_rnn, [-1, rnn_unit])  # 作为输出层的输入
    index = tf.range(0, batch_size) * time_step + (time_step - 1)  # 只取最后的输出
    output = tf.gather(output, index)  # 按照index取数据
    mm = output
    w_out = weights['out']
    b_out = biases['out']
    pred0 = tf.matmul(output, w_out) + b_out
    pred = tf.nn.dropout(pred0, keep_prob)
    return pred, final_states, m, mm


# ——————————————————训练模型—————————————————
def train_lstm(data, train_begin, train_end, refer_data, predict_duration, init_tikv_replicas):
    X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
    Y = tf.placeholder(tf.float32, shape=[None, output_size])
    keep_prob = tf.placeholder('float')
    batch_index, train_x, train_y = get_train_data(data, batch_size, time_step, train_end, predict_step, train_begin)
    print("train_x.shape", np.array(train_x).shape)
    print("train_y.shape", np.array(train_y).shape)
    pred, _, m, mm = lstm(X, input_size, rnn_unit, keep_prob, weights, biases)
    # 损失函数
    loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1, output_size]) - tf.reshape(Y, [-1, output_size])))
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)

    saver = tf.train.Saver(max_to_keep=1)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(save_model_path_requests)  # checkpoint存在的目录
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)  # 自动恢复model_checkpoint_path保存模型一般是最新
            print("Model restored...")
        else:
            print('No Model')

        # 重复训练
        for _ in range(train_time):
            for step in range(len(batch_index) - 1):
                sess.run([train_op, loss, m, mm],
                            feed_dict={X: train_x[batch_index[step]:batch_index[step + 1]],
                            Y: train_y[batch_index[step]:batch_index[step + 1]],
                            keep_prob: kp})

        saver.save(sess, save_model_path_requests + save_model_name)  # 保存模型

        # test_begin = 0
        # test_end = len(data)
        # _, test_x, test_y = get_train_data(data, batch_size, time_step, test_end, predict_step, test_begin)
        # test_predict = []
        # for step in range(len(test_x)):
        #     prob = sess.run(pred, feed_dict={X: [test_x[step]], keep_prob: kp})
        #     predict = prob.reshape((-1))
        #     test_predict.extend(predict)
        # test_predict = np.array(test_predict).reshape(-1,)
        # print(test_predict.shape)
        # test_predict = test_predict.tolist()
        # test_y = np.array(test_y).reshape(-1, )
        # print(test_y.shape)
        # test_y = test_y.tolist()
        # with open('y.json', 'w') as f:
        #     json.dump(test_y, f, indent=4)
        # with open('y2.json', 'w') as f:
        #     json.dump(test_predict, f, indent=4)

        # 以下是预测部分
        
        predict_duration = int(predict_duration)

        j = 0
        while j < predict_duration: # 每一分钟都进行一次预测
            test_x, mean_x, std_x = get_test_data()
            test_predict = []
            for step in range(len(test_x)):
                prob = sess.run(pred, feed_dict={X: [test_x[step]], keep_prob: kp})
                predict = prob.reshape((-1))
                test_predict.extend(predict)
            print('test_predict.shape:', np.array(test_predict).shape)
            # test_x_cpu = get_test_data_cpu()
            test_y = get_test_y()
            print('test_y:', np.array(test_y).shape)

            # td = MyThread(predict_cpu, args=(test_x_cpu, save_model_path_cpu))
            # td.start()
            # td.join()
            # test_predict_cpu = td.get_result()
            # if test_predict_cpu is None:
            #     print("Error run cpu predict thread")

            max_data = max(test_predict) # 未来这predict_step分钟内最高峰的数据
            max_index = test_predict.index(max_data)
            before_normalized_max_data = max_data * std_x + mean_x
            slope = (max_data - test_y[0]) / (max_index + 1)
            if slope * min_scale_interval > 1:
                open_multi_nodes = True
            else:
                open_multi_nodes = False

            # 以下是计算应该开启多少节点的代码，根据曲线的斜率以及在曲线最高点集群能够正常运转所需要的最少节点数目
            # 目前先只考虑节点数目在3-5之间变化的情况
            expected_tikv_replicas = 3
            predict_replicas = 3
            if before_normalized_max_data >= three_nodes_endurable_ops and before_normalized_max_data < four_nodes_endurable_ops:
                expected_tikv_replicas = 4
            elif before_normalized_max_data > four_nodes_endurable_ops:
                expected_tikv_replicas = 5
            if open_multi_nodes == False:
                if expected_tikv_replicas > init_tikv_replicas:
                    predict_replicas = init_tikv_replicas + 1
            else:
                predict_replicas = expected_tikv_replicas
            refer_data['replicas'] = predict_replicas
            j += 1
            time.sleep(60)
        

def start(period_duration, train_periods, predict_duration):
    init_tikv_replicas = yaml_to_dict(yaml_path) # 从yaml文件中取出当前的tikv副本数目
    refer_data = get_tikv_replicas() # 这是将要由django接口传出去的数据
    refer_data['name'] = name
    refer_data['namespace'] = namespace

    period_duration = int(period_duration)
    train_periods = int(train_periods)
    train_end = period_duration * train_periods
    train_begin = 0

    predict_duration = int(predict_duration)

    statement_ops, kv_grpc_msg_qps, _ = load_history_workload()
    print("statement_ops len: ", len(statement_ops))
    statement_ops = parse_requests(statement_ops, kv_grpc_msg_qps)
    print("statement_ops len: ", len(statement_ops))

    # ——————————————————定义神经网络变量——————————————————
    # 输入层、输出层权重、偏置
    t0 = time.time()
    train_lstm(statement_ops, train_begin, train_end, refer_data, predict_duration, init_tikv_replicas)
    t1 = time.time()
    print("时间:%.4fs" % (t1 - t0))
