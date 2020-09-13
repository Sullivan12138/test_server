"""
用于预测的代码
目前还没有实现同时预测cpu usage和ops的功能
同时预测似乎也没有什么用
我发现不能将train和predict放在两个文件里，虽然这是两个独立的过程，
但似乎一定要放在同一个文件里
"""
import numpy as np
import yaml
import time
from fetch_prome_metrics import *
from variables import *
from train import lstm
import sys
import threading
from globalvar import get_tikv_replicas
import tensorflow as tf


def yaml_to_dict(yaml_path):
    with open(yaml_path, "r") as f:
        generate_dict = yaml.load(f, Loader=yaml.FullLoader)
        init_tikv_replicas = int(generate_dict['spec']['tikv']['replicas'])
        return init_tikv_replicas


def get_kv_cop(kv_grpc_msg_qps):
    length = len(kv_grpc_msg_qps[0]['values'])
    sum = [0. for _ in range(length)]
    for metric in kv_grpc_msg_qps:
        if metric['metric']['type'] == 'coprocessor':
            for i in range(len(metric['values'])):
                sum[i] += float(metric['values'][i][1])
    return sum


def sum_up(statement_ops):
    length = len(statement_ops[0]['values'])
    sum = [0. for _ in range(length)]
    for metric in statement_ops:
        if metric['metric']['type'] != 'use':
            for i in range(len(metric['values'])):
                sum[i] += float(metric['values'][i][1])
    return sum


def determine_workload_proportion(statement_ops, sum_data, kv_cop):
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


def cal_weights(sum_data, proportions, weight):
    for i in range(len(sum_data)):
        sum_data[i] = sum_data[i] * (proportions[i][0] * weight[0] + proportions[i][1] * weight[1] + proportions[i][2] *
                                     weight[2] + proportions[i][3] * weight[3])
    return sum_data


def parse(statement_ops, kv_grpc_msg_qps):
    kv_cop = get_kv_cop(kv_grpc_msg_qps)
    sum_data = sum_up(statement_ops)
    proportions = determine_workload_proportion(statement_ops, sum_data, kv_cop)
    new_input_data = cal_weights(sum_data, proportions, weight)
    return new_input_data


# def parse_cpu(tikv_cpu_usage):
#     length = len(tikv_cpu_usage[0]['values'])
#     sum = [0. for _ in range(length)]
#     for metric in tikv_cpu_usage:
#         for i in range(len(metric['values'])):
#             sum[i] += float(metric['values'][i][1])
#     return sum


# # 获取测试集
# def get_test_data(data, time_step, test_begin, predict_step):
#     data_test = data[test_begin:]
#     data_test = np.array(data_test)
#     mean = np.mean(data_test, axis=0)
#     std = np.std(data_test, axis=0)
#     normalized_test_data = (data_test - mean) / std  # 标准化
#     maxvalue = np.max(data_test, axis=0)
#     normalized_test_data = normalized_test_data.tolist()
#     # size=(len(normalized_test_data)+time_step-1)//time_step
#     test_x, test_y = [], []
#     for i in range(len(normalized_test_data) - time_step - (predict_step - 1)):
#         x = normalized_test_data[i:i + time_step]
#         y = normalized_test_data[i + time_step + (predict_step - 1)]
#         test_x.append(x)
#         test_y.append(y)
#     print('get_test_data x', np.array(test_x).shape)
#     print('get_test_data y', np.array(test_y).shape)
#
#     return maxvalue, mean, std, test_x, test_y


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
        sum_data = parse(statement_ops, kv_grpc_msg_qps)
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
    test_y = parse(statement_ops, kv_grpc_msg_qps)
    test_y = np.array(test_y)
    normalized_test_y = (test_y - np.mean(test_y, axis=0)) / np.std(test_y, axis=0)
    normalized_test_y = normalized_test_y.tolist()
    return normalized_test_y


# def get_test_data_cpu():
#     test_x_cpu = []
#     for i in range(predict_step):
#         end = str(int(time.time()) - (predict_step - 1 - i) * 60)
#         start = str(int(time.time()) - (predict_step + time_step - 2 - i) * 60)
#         tikv_cpu_usage = fetch_tikv_cpu_usage(start, end, 60)
#         sum_data = parse_cpu(tikv_cpu_usage)
#         res = []
#         for input_data in sum_data:
#             data = [input_data]
#             res.append(data)
#         res = np.array(res)
#         normalized_test_data = (res - np.mean(res, axis=0)) / np.std(res, axis=0)
#         normalized_test_data = normalized_test_data.tolist()
#         test_x_cpu.append(normalized_test_data)
#     return test_x_cpu


def predict(test_x, save_model_path):
    X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
    keep_prob = tf.placeholder('float')
    pred, _, _, _ = lstm(X, input_size, rnn_unit, kp)

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


# def predict_cpu(test_x, save_model_path):
#     X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
#     keep_prob = tf.placeholder('float')
#     pred, _, _, _ = lstm(X, weights_cpu, biases_cpu, input_size, rnn_unit, kp)

#     saver = tf.train.Saver(max_to_keep=1)
#     with tf.Session() as sess:
#         ckpt = tf.train.get_checkpoint_state(save_model_path)
#         if ckpt and ckpt.model_checkpoint_path:
#             saver.restore(sess, ckpt.model_checkpoint_path)
#             print('Model Restored.')
#         else:
#             print('No Model!')

#         # 预测
#         test_predict = []
#         for step in range(len(test_x)):
#             prob = sess.run(pred, feed_dict={X: [test_x[step]], keep_prob: kp})
#             predict = prob.reshape((-1))
#             test_predict.extend(predict)
#         print('test_predict:', np.array(test_predict).shape)
#         return test_predict


# class MyThread(threading.Thread):

#     def __init__(self, func, args=()):
#         super(MyThread, self).__init__()
#         self.func = func
#         self.args = args

#     def run(self):
#         self.result = self.func(*self.args)

#     def get_result(self):
#         try:
#             return self.result
#         except Exception:
#             return None


def start_predict(name, namespace, predict_duration):
    init_tikv_replicas = yaml_to_dict(yaml_path) # 从yaml文件中取出当前的tikv副本数目
    refer_data = get_tikv_replicas()
    refer_data['name'] = name
    refer_data['namespace'] = namespace
    predict_duration = int(predict_duration)

    j = 0
    while j < predict_duration: # 每一分钟都进行一次预测
        test_x, mean_x, std_x = get_test_data()
        test_predict = predict(test_x, save_model_path_requests)
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


if __name__ == "__main__":
    start_predict("st-2", "pd-team-s2", "180")
