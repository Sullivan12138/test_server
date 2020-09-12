"""
这里包含从prometheus获取各种metrics的方法
和特征选择那一版本的预测代码相配合，目前没有用
"""

import requests


def fetch_tikv_cpu_usage(prome_addr, start, end, step=30):
    r = requests.get(
        'http://%s/api/v1/query_range?query=sum(rate(tikv_thread_cpu_seconds_total[1m])) by (instance)&start=%s&end=%s&step=%s' % (
            prome_addr, start, end, step))
    res = r.json()
    if res['status'] == 'error':
        raise Exception(
            'an error occurred when fetching tikv cpu usage: errorType={}: {}'.format(res['errorType'], res['error']))
    return res['data']['result']


# 可获取
def fetch_tikv_grpc_poll_cpu(prome_addr, start, end, step=30):
    r = requests.get(
        'http://%s/api/v1/query_range?query=sum(rate(tikv_thread_cpu_seconds_total{'
        'name=~"grpc.*"}[1m])) by (instance)&start=%s&end=%s&step=%s' % (prome_addr, start, end, step))
    res = r.json()
    if res['status'] == 'error':
        raise Exception(
            'an error occurred when fetching tikv grpc poll cpu: errorType={}: {}'.format(res['errorType'], res['error'])
        )
    return res['data']['result']


def fetch_tikv_rocksdb_cpu(prome_addr, start, end, step=30):
    r = requests.get(
        'http://%s/api/v1/query_range?query=sum(rate(tikv_thread_cpu_seconds_total{name=~"rocksdb.*"}[1m])) by (instance)&start=%s&end=%s&step=%s' % (prome_addr, start, end, step))
    res = r.json()
    if res['status'] == 'error':
        raise Exception(
            'an error occurred when fetching tikv grpc poll cpu: errorType={}: {}'.format(res['errorType'], res['error'])
        )
    return res['data']['result']


def fetch_grpc_duration(prome_addr, start, end, step=30):
    # r = requests.get(
    #     'http://%s/api/v1/query_range?query=sum(rate(tikv_grpc_msg_duration_seconds_sum['
    #     '1m])) by (type) / sum(rate(tikv_grpc_msg_duration_seconds_count[1m])) by (type)&start=%s&end=%s&step=%s' %
    #     (prome_addr, start, end, step))
    r = requests.get(
        'http://%s/api/v1/query_range?query=histogram_quantile(0.99, sum(rate(tikv_grpc_msg_duration_seconds_bucket{'
        'type != "kv_gc"}[1m])) by(le, type))&start=%s&end=%s&step=%s' %
        (prome_addr, start, end, step))
    res = r.json()
    if res['status'] == 'error':
        raise Exception(
            'an error occurred when fetching tikv grpc duration put: errorType={}: {}'.format(res['errorType'],
                                                                                              res['error'])
        )
    return res['data']['result']


def fetch_grpc_duration_put(prome_addr, start, end, step=30):
    metrics = fetch_grpc_duration(prome_addr, start, end, step)
    put_metric = []
    for metric in metrics:
        if (metric['metric']['type'] == 'raw_batch_put') or (metric['metric']['type'] == 'raw_put'):
            put_metric.append(metric)
    return put_metric


def fetch_grpc_duration_prewrite(prome_addr, start, end, step=30):
    metrics = fetch_grpc_duration(prome_addr, start, end, step)
    prewrite_metric = []
    for metric in metrics:
        if metric['metric']['type'] == 'kv_prewrite':
            prewrite_metric.append(metric)
    return prewrite_metric


def fetch_grpc_duration_commit(prome_addr, start, end, step=30):
    metrics = fetch_grpc_duration(prome_addr, start, end, step)
    commit_metric = []
    for metric in metrics:
        if metric['metric']['type'] == 'kv_commit':
            commit_metric.append(metric)
    return commit_metric


def fetch_grpc_duration_cop(prome_addr, start, end, step=30):
    metrics = fetch_grpc_duration(prome_addr, start, end, step)
    cop_metric = []
    for metric in metrics:
        if metric['metric']['type'] == 'coprocessor' or metric['metric']['type'] == 'coprocessor_stream':
            cop_metric.append(metric)
    return cop_metric


def fetch_grpc_duration_get(prome_addr, start, end, step=30):
    metrics = fetch_grpc_duration(prome_addr, start, end, step)
    get_metric = []
    for metric in metrics:
        if 'get' in metric['metric']['type']:
            get_metric.append(metric)
    return get_metric


def fetch_grpc_duration_scan(prome_addr, start, end, step=30):
    metrics = fetch_grpc_duration(prome_addr, start, end, step)
    scan_metric = []
    for metric in metrics:
        if 'scan' in metric['metric']['type']:
            scan_metric.append(metric)
    return scan_metric


def fetch_io_utilization(prome_addr, start, end, step=30):
    return prome_addr, start, end, step


def fetch_disk_latency(prome_addr, start, end, step=30):
    return prome_addr, start, end, step


def fetch_io_in_bandwidth(prome_addr, start, end, step=30):
    return prome_addr, start, end, step


def fetch_io_out_bandwidth(prome_addr, start, end, step=30):
    return prome_addr, start, end, step


