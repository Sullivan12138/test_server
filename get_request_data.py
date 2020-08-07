import requests
import json
import time

url = 'http://10.233.28.181:2379/pd/api/v1/regions'
prome_addr = '10.233.18.170:9090'
dir = 'data/'


def fetch_request(prome_addr, start, end, step=30):
    r = requests.get(
        'http://%s/api/v1/query_range?query=sum(rate(tidb_tikvclient_request_seconds_count[1m])) by (instance, '
        'type)&start=%s&end=%s&step=%s' % (
            prome_addr, start, end, step))
    res = r.json()
    if res['status'] == 'error':
        raise Exception(
            'an error occurred when fetching tikv requests: errorType={}: {}'.format(res['errorType'], res['error']))
    return res['data']['result']


filename = 'requests.log'
print(filename)
i = 0
with open(dir + filename, 'a+') as f:
    while i < 1200:
        try:
            # res = requests.get(url)#获取训练集
            end = str(int(time.time()))
            start = end
            res = fetch_request(prome_addr, start, end, 60)
            sum = 0
            for metric in res:
                sum += metric['values'][-1][1]
            f.write(str(sum))
            time.sleep(60)
            i += 1
        except:
            print('request url error')
            time.sleep(60)
            i += 1

