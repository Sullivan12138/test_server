"""
特征选择那个版本的输入获取代码，目前没有用
"""
import datetime
import time

from tikv_prome import *


feature_inputs = [fetch_tikv_cpu_usage, fetch_io_utilization, fetch_disk_latency,
                  fetch_tikv_grpc_poll_cpu, fetch_io_in_bandwidth, fetch_io_out_bandwidth, fetch_grpc_duration_put,
                  fetch_grpc_duration_prewrite, fetch_grpc_duration_commit, fetch_grpc_duration_cop,
                  fetch_grpc_duration_get, fetch_grpc_duration_scan]
feature_inputs2 = [fetch_tikv_cpu_usage, fetch_tikv_grpc_poll_cpu, fetch_grpc_duration_put,
                   fetch_grpc_duration_prewrite, fetch_grpc_duration_commit, fetch_grpc_duration_cop,
                   fetch_grpc_duration_get, fetch_grpc_duration_scan, fetch_tikv_rocksdb_cpu]
feature_kinds = ['cpu', 'io util', 'io latency read', 'io latency write', 'grpc cpu', 'io bandwidth in',
                 'io bandwidth out', 'put', 'prewrite', 'commit', 'cop', 'get', 'scan']
feature_kinds2 = ['cpu', 'grpc cpu', 'put', 'prewrite', 'commit', 'cop', 'get', 'scan', 'rocksdb cpu']

start = "2020-08-30 15:00:00"
timeArray = time.strptime(start, "%Y-%m-%d %H:%M:%S")
start_time = int(time.mktime(timeArray))


# 从Prometheus获取cpu数据,把data整合成一行输入,使用全局变量
def get_input(prome_addr, iter):

    step = 60
    end = str(start_time + 60 * iter)
    f = open('metrics.txt', 'a+')

    # 只获取前1分钟的数据
    start = end
    for item in feature_inputs2:
        input_data = item(prome_addr, start, end, step)
        # 求出所有实例的总metrics
        sum = 0
        for ins in input_data:
            if ins['values'][-1][1] != 'NaN':
                sum += float(ins['values'][-1][1])
        print(sum, file=f)
    f.close()
    # print(history_data, file=data)
    # else:
    #     # 获取前interval分钟的数据
    #     start = str(end_time - 60 * (interval - 1))
    #
    #     input_data = []
    #     for item in feature_inputs2:
    #         input_data.append(item(prome_addr, start, end, step))
    #     for i in range(-interval, 0):
    #         history_data = []
    #         for j in range(len(feature_inputs2)):
    #             sum = 0
    #             for ins in input_data[j]:
    #                 if ins['values'][i][1] != 'NaN':
    #                     sum += float((ins['values'][i][1]))
    #             history_data.append(sum)


if __name__ == "__main__":
    i = 0
    time_step = 15
    prome_addr = '10.233.18.170:9090'
    f = open('metrics.txt', 'w')
    f.close()
    while i < 1200:
        get_input(prome_addr, i)
        i += 1
