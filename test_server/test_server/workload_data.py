"""
这个文件的作用是从prometheus获取数据并保存到文件
同时提供读取文件的接口
"""
import time
import json
import sys
from fetch_prome_metrics import *
from os import listdir

dir_name_statement_ops = './data/statement_ops/'
dir_name_kv_grpc_msg_qps = './data/kv_grpc_msg_qps/'
dir_name_cpu_usage = './data/cpu_usage/'
start = "2020-08-29 15:00:00"
timeArray = time.strptime(start, "%Y-%m-%d %H:%M:%S")
start_time = int(time.mktime(timeArray))


def write_file(file_name, data):
    f = open(file_name, 'w')
    f.close()
    with open(file_name, 'a+') as f:
        json.dump(data, f, indent=4)


def dump_history_workload(get_data_duration):
    for i in range(get_data_duration):
        end = str(start_time + 60 * i)
        start = end
        filename = end + '.log'
        write_file(dir_name_statement_ops + filename, fetch_statement_ops(start, end, 60))
        write_file(dir_name_kv_grpc_msg_qps + filename, fetch_kv_grpc_msg_qps(start, end, 60))
        write_file(dir_name_cpu_usage + filename, fetch_tikv_cpu_usage(start, end, 60))


def read_file(dir_name):
    data = []
    files = listdir(dir_name)
    files.sort()
    for file in files:
        f = open(dir_name + file, 'r')
        load_dict = json.load(f)
        f.close()
        data.append(load_dict)
    return data


def load_history_workload():
    statement_ops = read_file(dir_name_statement_ops)
    kv_grpc_msg_qps = read_file(dir_name_kv_grpc_msg_qps)
    tikv_cpu_usage = read_file(dir_name_cpu_usage)
    return statement_ops, kv_grpc_msg_qps, tikv_cpu_usage


def start_dump(duration):
    duration = int(duration)
    dump_history_workload(duration)