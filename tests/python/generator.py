import os
import glob
import time
import random
import logging
from threading import Thread
from multiprocessing import Queue
from datetime import datetime
import numpy as np
from grpc.beta import implementations
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2


class Dataset(object):
    def __init__(self, data_dir, max_count=1000):
        self.images = []
        for fn in os.listdir(data_dir):
            with open(os.path.join(data_dir, fn), 'rb') as f:
                self.images.append(f.read())
            if max_count > 0 and len(self.images) >= max_count:
                break

    def rand_idx(self):
        return random.randint(0, len(self.images) - 1)


class Worker(Thread):
    def __init__(self, idx, dataset, queue, output, app_id):
        super(Worker, self).__init__()
        self.daemon = True
        self.index = idx
        self.dataset = dataset
        self.queue = queue
        self.lats = []
        self.img_idx = random.randint(0, len(self.dataset.images) - 1)
        self.output= output
	self.app_id = app_id


    def run(self):
	host = "localhost"
	port = "9000"
	channel = implementations.insecure_channel(host, int(port))
	stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
	request = predict_pb2.PredictRequest()
	request.model_spec.name = 'vgg'+str(self.app_id)
	request.model_spec.signature_name = 'predict_images'
        while True:
            idx = self.queue.get()
            if idx == -1:
                break
            img = self.dataset.images[self.img_idx]
	    request.inputs['images'].CopyFrom(
	        tf.contrib.util.make_tensor_proto(img, shape=[1]))
            start = datetime.now()
	    try:
	    	result = stub.Predict(request, 5.0)
	    except:
		self.lats.append((self.img_idx, 100000.0))
	    else:
            	end = datetime.now()
            	lat = (end - start).total_seconds() * 1000.0
                self.lats.append((self.img_idx, lat))

            self.img_idx = (self.img_idx + 1) % len(self.dataset.images)

class Generator(object):
    def __init__(self, dataset, output, app_id):
        self.dataset = dataset
        self.queue = Queue()
        self.workers = []
        self.beg = None
        for i in range(0, 500):
            worker = Worker(i, dataset, self.queue, output, app_id)
            worker.start()
            self.workers.append(worker)

    def run(self, rate, duration):
        count = 0
        gap = 1. / rate
        total = duration * rate
        beg = time.time()
        self.beg = time.time()
        logging.info('Start sending request at {} req/s'.format(rate))
        while True:
            now = time.time()
            while count * gap <= now - beg:
                self.queue.put(1)
                count += 1
                now = time.time()
                if count >= total:
                    break
            if count >= total or now - beg >= duration:
                break
            to_sleep = beg + count * gap - now
            if to_sleep > 0:
                time.sleep(to_sleep)
        elapse = time.time() - self.beg
        logging.info('Generate {} requests in {} sec, rate: {} req/s'.format(
            count, elapse, float(count) / elapse))

    def stop_all(self):
        for _ in range(len(self.workers)):
            self.queue.put(-1)
        for t in self.workers:
            t.join()
        elapse = time.time() - self.beg
        logging.info('Finished all requests in {} sec'.format(elapse))

    def output_latencies(self, output):
        self.stop_all()
        with open(output, 'a') as fout:
            for worker in self.workers:
                for img_idx, lat in worker.lats:
                    fout.write('%s\n' % (lat))
        logging.info('Output latencies to %s' % output)
