"""
从prometheus获取metrics的方法
目前版本的预测代码使用这些metrics
"""

import requests
addr = '10.233.18.170:9090'


def fetch_kv_grpc_msg_qps(start, end, step=30):
    r = requests.get(
        'http://%s/api/v1/query_range?query=sum(rate(tikv_grpc_msg_duration_seconds_count{'
        'type!="kv_gc"}[1m])) by (instance,type)&start=%s&end=%s&step=%s' % (
            addr, start, end, step))
    res = r.json()
    if res['status'] == 'error':
        raise Exception(
            'an error occurred when fetching tikv requests: errorType={}: {}'.format(res['errorType'], res['error']))
    return res['data']['result']


def fetch_statement_ops(start, end, step=30):
    r = requests.get('http://%s/api/v1/query_range?query=sum(rate(tidb_executor_statement_total[1m])) by ('
                     'type)&start=%s&end=%s&step=%s' % (addr, start, end, step))
    res = r.json()
    if res['status'] == 'error':
        raise Exception(
            'an error occurred when fetching tikv requests: errorType={}: {}'.format(res['errorType'], res['error']))
    return res['data']['result']


def fetch_tikv_cpu_usage(start, end, step=30):
    r = requests.get(
        'http://%s/api/v1/query_range?query=sum(rate(tikv_thread_cpu_seconds_total[1m])) by ('
        'instance)&start=%s&end=%s&step=%s' % (
            addr, start, end, step))
    res = r.json()
    if res['status'] == 'error':
        raise Exception(
            'an error occurred when fetching tikv cpu usage: errorType={}: {}'.format(res['errorType'], res['error']))
    return res['data']['result']


