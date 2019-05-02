import multiprocessing
import os
import sys
import argparse
import subprocess
import shutil
import uuid
import yaml
import numpy as np


_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
_profiler = os.path.join(_root, 'build/profiler')
_models = {}


def parse_int_list(s):
    res = []
    for split in s.split(','):
        if '-' in split:
            st, ed = map(int, split.split('-'))
            res.extend(range(st, ed+1))
        else:
            res.append(int(split))
    return res


def load_model_db(path):
    with open(path) as f:
        model_db = yaml.load(f.read())
    for model_info in model_db['models']:
        framework = model_info['framework']
        model_name = model_info['model_name']
        if framework not in _models:
            _models[framework] = {}
        _models[framework][model_name] = model_info

    if 'tf_share' in model_db:
        d = {}
        for model_info in model_db['tf_share']:
            for suffix_info in model_info['suffix_models']:
                name = suffix_info['model_name']
                if name in d:
                    raise ValueError(f'Duplicated model {name}')
                d[name] = model_info
        _models['tf_share'] = d


def find_max_batch(framework, model_name, gpus):
    global args
    cmd_base = '%s -model_root %s -image_dir %s -gpu %s -framework %s -model %s -model_version %s' % (
        _profiler, args.model_root, args.dataset, gpus[0], framework, model_name, args.version)
    if args.height > 0 and args.width > 0:
        cmd_base += ' -height %s -width %s' % (args.height, args.width)
    if args.prefix:
        cmd_base += ' -share_prefix'
    left = 0
    right = 256
    curr_tp = None
    out_of_memory = False
    while True:
        prev_tp = curr_tp
        curr_tp = None
        cmd = cmd_base + ' -min_batch %s -max_batch %s' % (right, right)
        print(cmd)
        # exit(0)
        proc = subprocess.Popen(cmd, shell=True, universal_newlines=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        out, err = proc.communicate()
        if 'out of memory' in err or 'out of memory' in out:
            print('batch %s: out of memory' % right)
            out_of_memory = True
            break
        flag = False
        for line in out.split('\n'):
            if flag:
                items = line.split(',')
                lat = float(items[1]) + float(items[2]) # mean + std
                curr_tp = right * 1e6 / lat
                break
            if line.startswith('batch,latency'):
                flag = True
        if curr_tp is None:
            # Unknown error happens, need to fix first
            print(err)
            exit(1)
        print('batch %s: throughput %s' % (right, curr_tp))
        if prev_tp is not None and curr_tp / prev_tp < 1.01:
            break
        left = right
        right *= 2
    if not out_of_memory:
        return right
    while right - left > 1:
        print(left, right)
        mid = (left + right) // 2
        cmd = cmd_base + ' -min_batch %s -max_batch %s' % (mid, mid)
        print(cmd)
        proc = subprocess.Popen(cmd, shell=True, universal_newlines=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        out, _ = proc.communicate()
        if 'out of memory' in out or 'OOM' in out:
            right = mid
        else:
            left = mid
    return left


def run_profiler(gpu, prof_id, input_queue, output_queue):
    while True:
        cmd = input_queue.get()
        if cmd is None:
            break
        cmd.append(f'-gpu={gpu}')
        print(' '.join(cmd))
        proc = subprocess.Popen(cmd, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = proc.communicate()
        if proc.returncode != 0:
            sys.stderr.write(err)
            return

        lines = iter(out.splitlines())
        for line in lines:
            if line.startswith(prof_id):
                break
        gpu_name = next(lines).strip()
        next(lines)  # Forward latency
        next(lines)  # batch,latency(us),std(us),memory(B)
        forward_stats = []
        preprocess_lats = []
        postprocess_lats = []
        for line in lines:
            if line.startswith('Preprocess latencies(us)'):
                break
            batch_size, lat, std, mem = line.split(',')
            forward_stats.append((int(batch_size), float(lat), float(std), int(mem)))
        for line in lines:
            if line.startswith('Postprocess latencies(us)'):
                break
            preprocess_lats.extend(map(float, line.split()))
        for line in lines:
            postprocess_lats.extend(map(float, line.split()))
        assert len(forward_stats) == 1
        output_queue.put((gpu_name, forward_stats[0], preprocess_lats, postprocess_lats))


def profile_model(framework, model_name, min_batch_limit, max_batch_limit, gpus):
    global args
    prof_id = '%s:%s:%s' % (framework, model_name, args.version)
    if args.height > 0 and args.width > 0:
        prof_id += ':%sx%s' % (args.height, args.width)
    print('Profile %s' % prof_id)

    if min_batch_limit > 0 and max_batch_limit > 0:
        max_batch = max_batch_limit
    else:
        max_batch = find_max_batch(framework, model_name, gpus)
        if max_batch_limit > 0 and max_batch > max_batch_limit:
            max_batch = max_batch_limit
    min_batch = max(1, min_batch_limit)
    print('Min batch: %s' % min_batch)
    print('Max batch: %s' % max_batch)

    input_queue = multiprocessing.Queue(max_batch - min_batch + 1)
    output_queue = multiprocessing.Queue(max_batch - min_batch + 1)
    workers = []
    for gpu in gpus:
        worker = multiprocessing.Process(target=run_profiler, args=(gpu, prof_id, input_queue, output_queue))
        worker.start()
        workers.append(worker)
    for batch in range(min_batch, max_batch + 1):
        cmd = [_profiler,
               f'-model_root={args.model_root}', f'-image_dir={args.dataset}',
               f'-framework={framework}', f'-model={model_name}', f'-model_version={args.version}',
               f'-min_batch={batch}', f'-max_batch={batch}']
        if args.height > 0 and args.width > 0:
            cmd.append(f'-height={args.height}')
            cmd.append(f'-width={args.width}')
        if args.prefix:
            cmd.append('-share_prefix')
        input_queue.put(cmd)
    for _ in gpus:
        input_queue.put(None)
    input_queue.close()

    output = str(uuid.uuid4()) + '.txt'
    forward_stats = []
    preprocess_lats = []
    postprocess_lats = []
    gpu_name = None
    for _ in range(min_batch, max_batch + 1):
        try:
            gpu_name, forward_stat, pre, post = output_queue.get()
        except KeyboardInterrupt:
            print('exiting...')
            for worker in workers:
                worker.terminate()
            output_queue.close()
            for worker in workers:
                worker.join()
            print('worker joined')
            output_queue.join_thread()
            raise
        forward_stats.append(forward_stat)
        preprocess_lats.extend(pre)
        postprocess_lats.extend(post)

        forward_stats.sort()
        with open(output, 'w') as f:
            f.write(f'{prof_id}\n')
            f.write(f'{gpu_name}\n')
            f.write('Forward latency\n')
            f.write('batch,latency(us),std(us),memory(B)\n')
            for batch_size, lat, std, mem in forward_stats:
                f.write(f'{batch_size},{lat},{std},{mem}\n')
            f.write('Preprocess latency\n')
            f.write('mean(us),std(us)\n')
            f.write(f'{np.average(preprocess_lats)},{np.std(preprocess_lats)}\n')
            f.write('Postprocess latency\n')
            f.write('mean(us),std(us)\n')
            f.write(f'{np.average(postprocess_lats)},{np.std(postprocess_lats)}\n')
    print('joining worker...')
    for worker in workers:
        worker.join()
    print('worker joined')

    prof_dir = os.path.join(args.model_root, 'profiles')
    assert gpu_name is not None
    gpu_dir = os.path.join(prof_dir, gpu_name)
    if not os.path.exists(gpu_dir):
        os.mkdir(gpu_dir)
    prof_file = os.path.join(gpu_dir, '%s.txt' % prof_id.replace(':', '-'))
    if os.path.exists(prof_file) and not args.force:
        print('%s already exists' % prof_file)
        print('Save profile to %s' % output)
    else:
        shutil.move(output, prof_file)
        print('Save profile to %s' % prof_file)


def main():
    parser = argparse.ArgumentParser(description='Profile models')
    parser.add_argument('framework',
                        choices=['caffe', 'caffe2', 'tensorflow', 'darknet', 'tf_share'],
                        help='Framework name')
    parser.add_argument('model', type=str, help='Model name')
    parser.add_argument('model_root', type=str,
                        help='Nexus model root directory')
    parser.add_argument('dataset', type=str,
                        help='Dataset directory')
    parser.add_argument('--version', type=int, default=1,
                        help='Model version (default: 1)')
    parser.add_argument('--gpus', default='0-7', help='GPU indexes. e.g. 0-2,4,5,7-8')
    parser.add_argument('--height', type=int, default=0, help='Image height')
    parser.add_argument('--width', type=int, default=0, help='Image width')
    parser.add_argument('--prefix', action='store_true',
                        help='Enable prefix batching')
    parser.add_argument('--max_batch', type=int, default=0,
                        help='Limit the max batch size (default: 0)')
    parser.add_argument('--min_batch', type=int, default=0,
                        help='Limit the min batch size (default: 0)')
    parser.add_argument('-f', '--force', action='store_true',
                        help='Overwrite the existing model profile in model DB')
    global args
    args = parser.parse_args()

    if args.prefix and args.framework == 'tf_share':
        sys.stderr.write('Cannot use -prefix with TFShare models')
        return
    load_model_db(os.path.join(args.model_root, 'db', 'model_db.yml'))
    if args.model not in _models[args.framework]:
        sys.stderr.write('%s:%s not found in model DB\n' % (args.framework, args.model))
        return

    gpus = parse_int_list(args.gpus)
    profile_model(args.framework, args.model, args.min_batch, args.max_batch, gpus)
    

if __name__ == '__main__':
    main()
