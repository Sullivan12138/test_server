import time
import sys
from variables import *
from train import train_lstm
from workload_data import load_history_workload
import tensorflow as tf

weights_cpu = {
    'in': tf.Variable(tf.random_uniform([input_size, rnn_unit])),  # max_val=0.125
    'out': tf.Variable(tf.random_uniform([rnn_unit, output_size]))
}
biases_cpu = {
    'in': tf.Variable(tf.constant(0.1, shape=[rnn_unit, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[output_size, ]))
}


def parse_cpu(tikv_cpu_usage):
    res = []
    for data in tikv_cpu_usage:
        sum = 0
        for metric in data:
            sum += float(metric['values'][-1][1])
        sum_data = [sum]
        res.append(sum_data)
    return res


if __name__ == '__main__':
    period_duration = int(sys.argv[1])
    train_periods = int(sys.argv[2])
    train_end = period_duration * train_periods

    _, _, tikv_cpu_usage = load_history_workload()

    tikv_cpu_usage = parse_cpu(tikv_cpu_usage)

    t0 = time.time()
    train_lstm(tikv_cpu_usage, input_size, output_size, lr, train_time, rnn_unit, weights_cpu, biases_cpu,
               train_end, batch_size, time_step, kp, save_model_path_cpu)
    t1 = time.time()
    print("时间:%.4fs" % (t1 - t0))
