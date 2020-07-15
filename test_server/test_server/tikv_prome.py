from typing import Any, Callable

import requests


def fetch_tidb_cpu_usage(prome_addr, start, end, step=30):
    r = requests.get(
        'http://%s/api/v1/query_range?query=rate(process_cpu_seconds_total{job="tidb"}[1m])&start=%s&end=%s&step=%s' % (
            prome_addr, start, end, step))
    res = r.json()
    if res['status'] == 'error':
        raise Exception(
            'an error occurred when fetching tidb cpu usage: errorType={}: {}'.format(res['errorType'], res['error']))
    return res['data']['result']


def fetch_tidb_heap_memory_usage(prome_addr, start, end, step=14):
    r = requests.get(
        'http://%s/api/v1/query_range?query=go_memstats_heap_inuse_bytes{job=~"tidb.*"}&start=%s&end=%s&step=%s' % (
            prome_addr, start, end, step))
    res = r.json()
    if res['status'] == 'error':
        raise Exception(
            'an error occurred when fetching tidb heap memeory usage: errorType={}: {}'.format(res['errorType'],
                                                                                               res['error']))
    return res['data']['result']


# 可获取
def fetch_tikv_cpu_usage(prome_addr, start, end, step=30):
    r = requests.get(
        'http://%s/api/v1/query_range?query=sum(rate(tikv_thread_cpu_seconds_total[1m])) by (instance)&start=%s&end=%s&step=%s' % (
            prome_addr, start, end, step))
    res = r.json()
    if res['status'] == 'error':
        raise Exception(
            'an error occurred when fetching tikv cpu usage: errorType={}: {}'.format(res['errorType'], res['error']))
    return res['data']['result']


# 不需要此metric
def fetch_tikv_memory_usage(prome_addr, start, end, step=30):
    instance = label_values(prome_addr, 'tikv_engine_size_bytes', 'instance')
    r = requests.get(
        'http://%s/api/v1/query_range?query=avg(process_resident_memory_bytes{instance=~"%s"}) by ('
        'instance)&start=%s&end=%s&step=%s' % (
            prome_addr, instance, start, end, step))
    res = r.json()
    if res['status'] == 'error':
        raise Exception('an error occurred when fetching tikv memory usage: errorType={}: {}'.format(res['errorType'],
                                                                                                     res['error']))
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


# 不能获取
def fetch_tikv_disk_read_latency(prome_addr, start, end, step=30):
    r = requests.get(
        'http://%s/api/v1/query_range?query=sum(rate(node_disk_read_time_seconds_total[1m])) by (device) / '
        'sum(rate(node_disk_reads_completed_total'
        '[1m])) by (device)&start=%s&end=%s&step=%s' % (prome_addr, start, end, step))
    res = r.json()
    if res['status'] == 'error':
        raise Exception(
            'an error occurred when fetching tikv disk latency: errorType={}: {}'.format(res['errorType'], res['error'])
        )
    return res['data']['result']


#
def fetch_tikv_disk_write_latency(prome_addr, start, end, step=30):
    r = requests.get(
        'http://%s/api/v1/query_range?query=sum(rate(node_disk_write_time_seconds_total[1m])) by (device) / '
        'sum(rate(node_disk_writes_completed_total'
        '[1m])) by (device)&start=%s&end=%s&step=%s' % (prome_addr, start, end, step))
    res = r.json()
    if res['status'] == 'error':
        raise Exception(
            'an error occurred when fetching tikv disk latency: errorType={}: {}'.format(res['errorType'], res['error'])
        )
    return res['data']['result']


#
def fetch_disk_io_util(prome_addr, start, end, step=30):
    r = requests.get(
        'http://%s/api/v1/query_range?query=sum(rate(node_disk_io_time_seconds_total{job="node-exporter"'
        '}[2m])) by ( '
        'device)&start=%s&end=%s&step=%s' % (prome_addr, start, end, step))
    res = r.json()
    if res['status'] == 'error':
        raise Exception(
            'an error occurred when fetching tikv grpc poll cpu: errorType={}: {}'.format(res['errorType'],
                                                                                          res['error'])
        )
    return res['data']['result']


#
def fetch_io_in_bandwidth(prome_addr, start, end, step=30):
    r = requests.get(
        'http://%s/api/v1/query_range?query=sum(rate(node_disk_read_bytes_total['
        '1m]) by (device)&start=%s&end=%s&step=%s' % (prome_addr, start, end, step))
    res = r.json()
    if res['status'] == 'error':
        raise Exception(
            'an error occurred when fetching tikv io bandwidth: errorType={}: {}'.format(res['errorType'],
                                                                                          res['error'])
        )
    return res['data']['result']


#
def fetch_io_out_bandwidth(prome_addr, start, end, step=30):
    r = requests.get(
        'http://%s/api/v1/query_range?query=sum(rate(node_disk_write_bytes_total['
        '1m])) by (device)&start=%s&end=%s&step=%s' % (prome_addr, start, end, step))
    res = r.json()
    if res['status'] == 'error':
        raise Exception(
            'an error occurred when fetching tikv io bandwidth: errorType={}: {}'.format(res['errorType'],
                                                                                          res['error'])
        )
    return res['data']['result']


def fetch_grpc_duration(prome_addr, start, end, step=30):
    r = requests.get(
        'http://%s/api/v1/query_range?query=sum(rate(tikv_grpc_msg_duration_seconds_sum['
        '1m])) by (type) / sum(rate(tikv_grpc_msg_duration_seconds_count[1m])) by (type)&start=%s&end=%s&step=%s' %
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


# 并不能获取到数据
def fetch_io_utilization(prome_addr, start, end, step=15):
    host = label_values(prome_addr, 'node_disk_reads_completed_total', 'instance')
    query = ('rate(node_disk_io_time_seconds_total{{instance="{host}"}}[1m])' +
             ' or irate(node_disk_io_time_seconds_total{{instance="{host}"}}[5m])').format(host=host)

    res = requests.get(
        'http://%s/api/v1/query_range?query=%s&start=%s&end=%s&step=%s' % (
            prome_addr, query, start, end, step)).json()
    if res['status'] == 'error':
        raise Exception('an error occurred when fetching tikv memory usage: errorType={}: {}'.format(res['errorType'],
                                                                                                     res['error']))

    return res['data']['result']


# 并不能获取到数据
def fetch_disk_latency(prome_addr, start, end, step=15):
    host = label_values(prome_addr=prome_addr, metric='node_disk_reads_completed_total', label='instance')
    device = label_values(prome_addr=prome_addr,
                          metric='node_disk_reads_completed_total{instance="%s", device!~"dm-.+"}' % host,
                          label='device')

    query = ('(rate(node_disk_read_time_seconds_total{{device=~"{device}", instance="{host}"}}[1m])' +
             ' / rate(node_disk_reads_completed_total{{device=~"{device}", instance="{host}"}}[1m]))' +
             ' or (irate(node_disk_read_time_seconds_total{{device=~"{device}", instance="{host}"}}[5m])' +
             ' / irate(node_disk_reads_completed_total{{device=~"{device}", instance="{host}"}}[5m]))').format(
        device=device,
        host=host)

    res = requests.get(
        'http://%s/api/v1/query_range?query=%s&start=%s&end=%s&step=%s' % (
            prome_addr, query, start, end, step)).json()
    if res['status'] == 'error':
        raise Exception('an error occurred when fetching tikv memory usage: errorType={}: {}'.format(res['errorType'],
                                                                                                     res['error']))

    return res['data']['result']


def fetch_disk_read_iops(prome_addr, start, end, step=15):
    host = label_values(prome_addr=prome_addr, metric='node_disk_reads_completed_total', label='instance')
    device = label_values(prome_addr=prome_addr,
                          metric='node_disk_reads_completed_total{instance="%s", device!~"dm-.+"}' % host,
                          label='device')

    query = (
            'rate(node_disk_reads_completed_total{{device=~"{device}", instance="{host}"}}[1m])'
            + ' or irate(node_disk_reads_completed_total{{device=~"{device}", instance="{host}"}}[5m])').format(
        device=device,
        host=host)

    res = requests.get(
        'http://%s/api/v1/query_range?query=%s&start=%s&end=%s&step=%s' % (
            prome_addr, query, start, end, step)).json()
    if res['status'] == 'error':
        raise Exception('an error occurred when fetching tikv memory usage: errorType={}: {}'.format(res['errorType'],
                                                                                                     res['error']))

    return res['data']['result']


def fetch_disk_write_iops(prome_addr, start, end, step=15):
    host = label_values(prome_addr=prome_addr, metric='node_disk_reads_completed_total', label='instance')
    device = label_values(prome_addr=prome_addr,
                          metric='node_disk_reads_completed_total{instance="%s", device!~"dm-.+"}' % host,
                          label='device')

    query = (
            'rate(node_disk_writes_completed_total{{device=~"{device}", instance="{host}"}}[1m])'
            + ' or irate(node_disk_writes_completed_total{{device=~"{device}", instance="{host}"}}[5m])').format(
        device=device,
        host=host)

    res = requests.get(
        'http://%s/api/v1/query_range?query=%s&start=%s&end=%s&step=%s' % (
            prome_addr, query, start, end, step)).json()
    if res['status'] == 'error':
        raise Exception('an error occurred when fetching tikv memory usage: errorType={}: {}'.format(res['errorType'],
                                                                                                     res['error']))

    return res['data']['result']


def fetch_disk_read_bandwidth(prome_addr, start, end, step=15):
    host = label_values(prome_addr=prome_addr, metric='node_disk_reads_completed_total', label='instance')
    device = label_values(prome_addr=prome_addr,
                          metric='node_disk_reads_completed_total{instance="%s", device!~"dm-.+"}' % host,
                          label='device')

    query = ('rate(node_disk_read_bytes_total{{device=~"{device}", instance="{host}"}}[1m]) ' +
             'or irate(node_disk_read_bytes_total{{device=~"{device}", instance="{host}"}}[5m])').format(
        device=device,
        host=host)

    res = requests.get(
        'http://%s/api/v1/query_range?query=%s&start=%s&end=%s&step=%s' % (
            prome_addr, query, start, end, step)).json()
    if res['status'] == 'error':
        raise Exception('an error occurred when fetching tikv memory usage: errorType={}: {}'.format(res['errorType'],
                                                                                                     res['error']))

    return res['data']['result']


def fetch_disk_write_bandwidth(prome_addr, start, end, step=15):
    host = label_values(prome_addr=prome_addr, metric='node_disk_reads_completed_total', label='instance')
    device = label_values(prome_addr=prome_addr,
                          metric='node_disk_reads_completed_total{instance="%s", device!~"dm-.+"}' % host,
                          label='device')

    query = ('rate(node_disk_written_bytes_total{{device=~"{device}", instance="{host}"}}[1m])' +
             ' or irate(node_disk_written_bytes_totalwritten{{device=~"{device}", instance="{host}"}}[5m])').format(
        device=device,
        host=host)

    res = requests.get(
        'http://%s/api/v1/query_range?query=%s&start=%s&end=%s&step=%s' % (
            prome_addr, query, start, end, step)).json()
    if res['status'] == 'error':
        raise Exception('an error occurred when fetching tikv memory usage: errorType={}: {}'.format(res['errorType'],
                                                                                                     res['error']))

    return res['data']['result']


def label_values(prome_addr, metric, label):
    r = requests.get('http://{}/api/v1/query?query={}'.format(prome_addr, metric))
    res = r.json()
    if res['status'] == 'error':
        raise Exception('label_values fails, errorType: {}, error: {}'.format(res['errorType'], res['error']))
    if res['data']['resultType'] != 'vector':
        raise Exception('expect resultType: vector, got: {}'.format(res['data']['resultType']))
    return "|".join(list(set(list(map(lambda item: item['metric'][label], res['data']['result'])))))


def get_max_memory_usage(prome_addr, start, end, component):
    if component == "tikv":
        data = fetch_tikv_memory_usage(prome_addr, start, end)
    elif component == "tidb":
        data = fetch_tidb_heap_memory_usage(prome_addr, start, end)
    else:
        raise Exception("illegal component: {}".format(component))
    max_usage = 0
    for instances in data:
        for item in instances['values']:
            max_usage = max(max_usage, int(item[1]))
    return float(max_usage)


def get_max_cpu_usage(prome_addr, start, end, component):
    if component == "tikv":
        data = fetch_tikv_cpu_usage(prome_addr, start, end)
    elif component == "tidb":
        data = fetch_tidb_cpu_usage(prome_addr, start, end)
    else:
        raise Exception("illegal component: {}".format(component))
    max_usage = 0
    for instances in data:
        for item in instances['values']:
            max_usage = max(max_usage, float(item[1]))
    return float(max_usage)


def get_cpu_usage(prome_addr, start, end, component):
    if component == "tikv":
        data = fetch_tikv_cpu_usage(prome_addr, start, end)
    elif component == "tidb":
        data = fetch_tidb_cpu_usage(prome_addr, start, end)
    else:
        raise Exception("illegal component: {}".format(component))
    return data


def get_memory_usage(prome_addr, start, end, component):
    if component == "tikv":
        data = fetch_tikv_memory_usage(prome_addr, start, end)
    elif component == "tidb":
        data = fetch_tidb_heap_memory_usage(prome_addr, start, end)
    else:
        raise Exception("illegal component: {}".format(component))
    return data


def get_io_util(prome_addr, host_and_device_name_list, start, end):
    data = fetch_io_utilization(prome_addr=prome_addr,
                                start=start,
                                end=end)
    match_any: Callable[[Any, Any], bool] = lambda s, t: \
        len(list(filter(lambda item: s["instance"].startswith(f"{item[0]}:") and
                                     s["device"] == item[1], t))) != 0

    return list(filter(lambda item: match_any(item["metric"], host_and_device_name_list), data))


def get_disk_latency(prome_addr, host_and_device_name_list, start, end):
    data = fetch_disk_latency(prome_addr=prome_addr,
                              start=start,
                              end=end)
    match_any: Callable[[Any, Any], bool] = lambda s, t: \
        len(list(filter(lambda item: s["instance"].startswith(f"{item[0]}:") and
                                     s["device"] == item[1], t))) != 0

    return list(filter(lambda item: match_any(item["metric"], host_and_device_name_list), data))


def get_disk_read_iops(prome_addr, host_and_device_name_list, start, end):
    data = fetch_disk_read_iops(prome_addr=prome_addr,
                                start=start,
                                end=end)
    match_any: Callable[[Any, Any], bool] = lambda s, t: \
        len(list(filter(lambda item: s["instance"].startswith(f"{item[0]}:") and
                                     s["device"] == item[1], t))) != 0

    return list(filter(lambda item: match_any(item["metric"], host_and_device_name_list), data))


def get_disk_write_iops(prome_addr, host_and_device_name_list, start, end):
    data = fetch_disk_write_iops(prome_addr=prome_addr,
                                 start=start,
                                 end=end)
    match_any: Callable[[Any, Any], bool] = lambda s, t: \
        len(list(filter(lambda item: s["instance"].startswith(f"{item[0]}:") and
                                     s["device"] == item[1], t))) != 0

    return list(filter(lambda item: match_any(item["metric"], host_and_device_name_list), data))


def get_disk_read_bandwidth(prome_addr, host_and_device_name_list, start, end):
    data = fetch_disk_read_bandwidth(prome_addr=prome_addr,
                                     start=start,
                                     end=end)
    match_any: Callable[[Any, Any], bool] = lambda s, t: \
        len(list(filter(lambda item: s["instance"].startswith(f"{item[0]}:") and
                                     s["device"] == item[1], t))) != 0

    return list(filter(lambda item: match_any(item["metric"], host_and_device_name_list), data))


def get_disk_write_bandwidth(prome_addr, host_and_device_name_list, start, end):
    data = fetch_disk_write_bandwidth(prome_addr=prome_addr,
                                      start=start,
                                      end=end)
    match_any: Callable[[Any, Any], bool] = lambda s, t: \
        len(list(filter(lambda item: s["instance"].startswith(f"{item[0]}:") and
                                     s["device"] == item[1], t))) != 0

    return list(filter(lambda item: match_any(item["metric"], host_and_device_name_list), data))


def get_value_serials(values):
    return list(map(lambda value: float(value[1]), list(filter(lambda value: value[1] != "NaN", values))))