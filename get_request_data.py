import requests
import json
import time

url = 'http://10.233.28.181:2379/pd/api/v1/regions'
prome_addr = '10.233.18.170:9090'
dirname = 'data/requests/'


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


i = 0
while i < 1200:
    end = str(int(time.time()))
    start = end
    filename = end + '.log'
    f = open(dirname + filename, 'w')
    f.close()
    with open(dirname + filename, 'a+') as f:
        res = fetch_request(prome_addr, start, end, 60)
        json.dump(res, f, indent=4)
    i += 1
    time.sleep(60)

