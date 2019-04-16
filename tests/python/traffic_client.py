import os
import sys
import numpy as np
import argparse
import random
from threading import Thread
import nexus

from generator import *

class NexusClient(ClientBase):
    def __init__(self, service_addr, dataset):
        user_id = random.randint(1, 10000000)
        self.client = nexus.Client(service_addr, user_id)
        self.dataset = dataset
        self.total_images = len(dataset.images)
        self.img_idx = random.randint(0, self.total_images-1)
    
    def query(self):
        img = self.dataset.images[self.img_idx]
        self.img_idx = (self.img_idx + 1) % self.total_images
        reply = self.client.request(img)
        if reply is None:
            return (0, [])
        if reply.status != 0:
            return (-reply.status, reply.query_latency)
        return (reply.latency_us, reply.query_latency)


def run_test(server_addr, dataset, rate, duration, output, nthreads):
    clients = []
    for _ in range(nthreads):
        client = NexusClient(server_addr, dataset)
        clients.append(client)
    gen = Generator(clients)
    gen.run(rate, duration)
    gen.output_latencies(output)

def run(sla, rps, duration, dataset):
    print('Test rps %s' % rps)
    nthreads = int(rps * float(sla) / 1e3 * 1.1)
    print('Number of threads: %s' % nthreads)

    output = 'traffic_sla%s_rps%s.txt' % (sla, rps)
    thread = Thread(target=run_test,
                    args=('127.0.0.1:9001', dataset, rps, duration, output, nthreads))
    thread.daemon = True
    thread.start()
    thread.join()
    total, good, drop, noresp, lat_50, lat_99, lat_max = parse_result(output, sla)
    percent = float(good) / total
    print('Traffic: %.2f%% (total/good/drop/noresp: %s/%s/%s/%s), 50th/99th/max latency: %s/%s/%s ms' % (
        percent*100, total, good, drop, noresp, lat_50, lat_99, lat_max))
    ret = True
    if percent < .99:
        ret = False
    retry = True
    if percent < .98:
        retry = False
    return ret, retry

def run_with_retry(sla, rps, duration, dataset):
    retry = 0
    while retry < 3:
        good, flag = run(sla, rps, duration, dataset)
        if good:
            return True
        if not flag:
            return False
        retry += 1
    return False

def eval_traffic(sla, base_rps):
    duration = 60
    #datapath = os.path.join(os.path.expanduser("~"), 'datasets/traffic/jackson_day')
    datapath = os.path.join(os.path.expanduser("~"), 'datasets/traffic/jackson_night')
    print('Dataset: %s' % datapath)
    dataset = Dataset(datapath)
    print('Latency sla: %s ms' % sla)

    min_rps = base_rps
    max_rps = base_rps
    while True:
        good = run_with_retry(sla, max_rps, duration, dataset)
        if not good:
            break
        min_rps = max_rps
        max_rps += 40
    while max_rps - min_rps > 1:
        rps = (max_rps + min_rps) / 2
        good = run_with_retry(sla, rps, duration, dataset)
        if good:
            min_rps = rps
        else:
            max_rps = rps
    print('Max throughput: %s' % min_rps)


def automatic():
    if len(sys.argv) < 3:
        print('%s sla base_rps' % sys.argv[0])
        exit()
    sla = int(sys.argv[1])
    base_rps = eval(sys.argv[2])
    # print('sla: %s' % sla)
    # print('num of models: %s' % n)
    # print('share prefix: %s' % share_prefix)
    eval_traffic(sla, base_rps)


def manual():
    if len(sys.argv) < 4:
        print('%s sla n rate [share_prefix]' % sys.argv[0])
        exit()
    sla = int(sys.argv[1])
    n = int(sys.argv[2])
    rate = eval(sys.argv[3])
    if len(sys.argv) > 4:
        share_prefix = (int(sys.argv[4]) == 1)
    else:
        share_prefix = False
    print('sla: %s' % sla)
    print('num of models: %s' % n)
    print('rate: %s' % rate)
    print('share prefix: %s' % share_prefix)

    duration = 60
    datapath = os.path.join(os.path.expanduser("~"), 'datasets/vgg_face')
    dataset = Dataset(datapath)

    run(sla, n, rate, share_prefix, duration, dataset)


if __name__ == "__main__":
    
    FORMAT = "[%(asctime)-15s %(levelname)s] %(message)s"
    logging.basicConfig(format=FORMAT)
    logging.getLogger().setLevel(logging.INFO)

    #manual()
automatic()
