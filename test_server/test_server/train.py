"""
用于训练模型的代码
训练ops
训练完了以后保存模型
"""
import tensorflow as tf
import numpy as np
import time
import sys
from workload_data import load_history_workload
from variables import *
import json


def parse_requests(statement_ops, kv_grpc_msg_qps):
    kv_cop = get_kv_cop(kv_grpc_msg_qps)
    sum_data = sum_up(statement_ops)
    proportions = determine_workload_proportion(statement_ops, sum_data, kv_cop)
    new_input_data = cal_weights(sum_data, proportions, weight)
    return new_input_data


def get_kv_cop(kv_grpc_msg_qps):
    res = []
    for data in kv_grpc_msg_qps:
        sum = 0
        for metric in data:
            if metric['metric']['type'] == 'coprocessor':
                sum += float(metric['values'][-1][1])
        res.append(sum)
    return res


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
                scan_qps = kv_cop[i] * scan_per_cop[scan_per_cop_level]
                proportion[1] = float(scan_qps / sum_data[i][0])
                proportion[0] = (float(metric['values'][-1][1]) - scan_qps) / sum_data[i][0]
        proportions.append(proportion)
    with open('p.json', 'w') as f:
        json.dump(proportions, f)
    return proportions


def cal_weights(sum_data, proportions, weight):
    for i in range(len(sum_data)):
        sum_data[i][0] = sum_data[i][0] * (proportions[i][0] * weight[0] + proportions[i][1] * weight[1] + proportions[i][2] *
                                           weight[2] + proportions[i][3] * weight[3])
    return sum_data


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


# ——————————————————定义网络——————————————————
def lstm(X, weights, biases, input_size, rnn_unit, keep_prob):
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
def train_lstm(data, input_size, output_size, lr, train_time, rnn_unit, weights, biases, train_end,
               batch_size, time_step, kp, save_model_path, train_begin=0):
    X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
    Y = tf.placeholder(tf.float32, shape=[None, output_size])
    keep_prob = tf.placeholder('float')
    batch_index, train_x, train_y = get_train_data(data, batch_size, time_step, train_end, predict_step, train_begin)
    print("train_x,shape", np.array(train_x).shape)
    print("train_y.shape", np.array(train_y).shape)
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

        # 重复训练
        for _ in range(train_time):
            for step in range(len(batch_index) - 1):
                sess.run([train_op, loss, m, mm],
                            feed_dict={X: train_x[batch_index[step]:batch_index[step + 1]],
                            Y: train_y[batch_index[step]:batch_index[step + 1]],
                            keep_prob: kp})

        test_begin = 0
        test_end = len(data)
        _, test_x, test_y = get_train_data(data, batch_size, time_step, test_end, predict_step, test_begin)
        test_predict = []
        for step in range(len(test_x)):
            prob = sess.run(pred, feed_dict={X: [test_x[step]], keep_prob: kp})
            predict = prob.reshape((-1))
            test_predict.extend(predict)
        test_predict = np.array(test_predict).reshape(-1,)
        print(test_predict.shape)
        test_predict = test_predict.tolist()
        test_y = np.array(test_y).reshape(-1, )
        print(test_y.shape)
        test_y = test_y.tolist()
        with open('y.json', 'w') as f:
            json.dump(test_y, f, indent=4)
        with open('y2.json', 'w') as f:
            json.dump(test_predict, f, indent=4)
        saver.save(sess, save_model_path + save_model_name)  # 保存模型


if __name__ == "__main__":

    period_duration = int(sys.argv[1])
    train_periods = int(sys.argv[2])
    train_end = period_duration * train_periods

    statement_ops, kv_grpc_msg_qps, _ = load_history_workload()
    print("statement_ops len: ", len(statement_ops))
    statement_ops = parse_requests(statement_ops, kv_grpc_msg_qps)
    print("statement_ops len: ", len(statement_ops))

    # ——————————————————定义神经网络变量——————————————————
    # 输入层、输出层权重、偏置
    t0 = time.time()
    train_lstm(statement_ops, input_size, output_size, lr, train_time, rnn_unit, weights, biases,
               train_end, batch_size, time_step, kp, save_model_path_requests)
    t1 = time.time()
    print("时间:%.4fs" % (t1 - t0))
