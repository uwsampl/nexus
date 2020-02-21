import math
import multiprocessing
import os
import io
import sys
import argparse
import subprocess
import shutil
import tempfile
import uuid
import yaml
import numpy as np
from PIL import Image


_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
_profiler = os.path.join(_root, 'build/profiler')
_models = {}
_dataset_dir = None


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
        model_db = yaml.safe_load(f.read())
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
        _profiler, args.model_root, _dataset_dir, gpus[0], framework, model_name, args.version)
    if args.height > 0 and args.width > 0:
        cmd_base += ' -height %s -width %s' % (args.height, args.width)
    if args.prefix:
        cmd_base += ' -share_prefix'
    left = 0
    right = 256
    curr_tp = None
    out_of_memory = False
    while True:
        print('Finding max_batch in [%d, inf)' % right)
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
            print('Batch size %s: out of memory' % right)
            out_of_memory = True
            break
        flag = False
        for line in out.split('\n'):
            if flag:
                items = line.split(',')
                lat = float(items[1]) + float(items[2])  # mean + std
                curr_tp = right * 1e6 / lat
                break
            if line.startswith('batch,latency'):
                flag = True
        if curr_tp is None:
            # Unknown error happens, need to fix first
            print(err)
            exit(1)
        if prev_tp is not None and curr_tp / prev_tp < 1.01:
            break
        left = right
        right *= 2
    if not out_of_memory:
        return right
    while right - left > 1:
        print('Finding max_batch in [%d, %s)' % (left, right))
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
        proc = subprocess.Popen(
            cmd, bufsize=0, encoding='utf-8', stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

        sio = io.StringIO(newline='')
        try:
            while True:
                if proc.poll() is not None:
                    break
                ch = proc.stdout.read(1)
                sio.write(ch)
                sys.stdout.write(ch)
                sys.stdout.flush()
        except KeyboardInterrupt:
            proc.terminate()
            proc.wait()
            raise
        if proc.returncode != 0:
            sys.stderr.write('Profiler exited with', proc.returncode)
            sys.stderr.flush()
            return

        lines = iter(sio.getvalue().splitlines())
        sio.close()
        for line in lines:
            if line.startswith(prof_id):
                break
        gpu_name = next(lines).strip()
        gpu_uuid = next(lines).strip()
        next(lines)  # Forward latency
        next(lines)  # batch,latency(us),std(us),memory(B),repeat
        forward_stats = []
        for line in lines:
            if line.startswith('Preprocess latency (mean,std,repeat)'):
                break
            batch_size, lat, std, mem, repeat = line.split(',')
            forward_stats.append((int(batch_size), float(
                lat), float(std), int(mem), int(repeat)))

        mean, std, repeat = next(lines).split(',')
        preprocess_lats = (float(mean), float(std), int(repeat))

        next(lines)  # Postprocess latency (mean,std,repeat)
        mean, std, repeat = next(lines).split(',')
        postprocess_lats = (float(mean), float(std), int(repeat))

        output_queue.put((gpu_name, gpu_uuid, forward_stats,
                          preprocess_lats, postprocess_lats))


def print_profile(output, prof_id, gpu_name, gpu_uuid, forward_stats, preprocess_lats, postprocess_lats):
    forward_stats.sort()
    with open(output, 'w') as f:
        f.write(f'{prof_id}\n')
        f.write(f'{gpu_name}\n')
        f.write(f'{gpu_uuid}\n')
        f.write('Forward latency\n')
        f.write('batch,latency(us),std(us),memory(B),repeat\n')
        for batch_size, lat, std, mem, repeat in forward_stats:
            f.write(f'{batch_size},{lat},{std},{mem},{repeat}\n')
        f.write('Preprocess latency (mean,std,repeat)\n')
        mean, std, repeat = preprocess_lats
        f.write(f'{mean},{std},{repeat}\n')
        f.write('Postprocess latency (mean,std,repeat)\n')
        mean, std, repeat = postprocess_lats
        f.write(f'{mean},{std},{repeat}\n')


def merge_mean_std(tuple1, tuple2):
    mean1, std1, n1 = tuple1
    mean2, std2, n2 = tuple2
    mean = ((n1 - 1) * mean1 + (n2 - 1) * mean2) / (n1 + n2 - 1)
    var = (n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2
    var += n1 * (mean1 - mean) ** 2
    var += n2 * (mean2 - mean) ** 2
    var /= (n1 + n2 - 1)
    return mean, math.sqrt(var), n1 + n2


def get_profiler_cmd(args):
    cmd = [_profiler,
           f'-model_root={args.model_root}', f'-image_dir={_dataset_dir}',
           f'-framework={args.framework}', f'-model={args.model}', f'-model_version={args.version}']
    if args.height > 0 and args.width > 0:
        cmd.append(f'-height={args.height}')
        cmd.append(f'-width={args.width}')
    if args.prefix:
        cmd.append('-share_prefix')
    return cmd


def profile_model_concurrent(gpus, min_batch, max_batch, prof_id, output):
    input_queue = multiprocessing.Queue(max_batch - min_batch + 1)
    output_queue = multiprocessing.Queue(max_batch - min_batch + 1)
    workers = []
    for gpu in gpus:
        worker = multiprocessing.Process(target=run_profiler, args=(
            gpu, prof_id, input_queue, output_queue))
        worker.start()
        workers.append(worker)
    for batch in range(min_batch, max_batch + 1):
        cmd = get_profiler_cmd(args)
        cmd += [f'-min_batch={batch}', f'-max_batch={batch}']
        input_queue.put(cmd)
    for _ in gpus:
        input_queue.put(None)
    input_queue.close()

    forward_stats = []
    preprocess_lats = None
    postprocess_lats = None
    for _ in range(min_batch, max_batch + 1):
        try:
            gpu_name, gpu_uuid, forward_stat, pre, post = output_queue.get()
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
        forward_stats.extend(forward_stat)
        if preprocess_lats is None:
            preprocess_lats = pre
            postprocess_lats = post
        else:
            preprocess_lats = merge_mean_std(preprocess_lats, pre)
            postprocess_lats = merge_mean_std(postprocess_lats, post)
        if len(gpus) > 1:
            gpu_uuid = 'generic'
        print_profile(output, prof_id, gpu_name, gpu_uuid,
                      forward_stats, preprocess_lats, postprocess_lats)

    print('joining worker...')
    for worker in workers:
        worker.join()
    print('worker joined')


def profile_model_single_gpu(gpu, min_batch, max_batch, prof_id, output):
    input_queue = multiprocessing.Queue()
    output_queue = multiprocessing.Queue()
    worker = multiprocessing.Process(target=run_profiler, args=(
        gpu, prof_id, input_queue, output_queue))
    worker.start()
    cmd = get_profiler_cmd(args)
    cmd += [f'-min_batch={min_batch}', f'-max_batch={max_batch}']
    input_queue.put(cmd)
    input_queue.put(None)
    input_queue.close()

    try:
        gpu_name, gpu_uuid, forward_stats, preprocess_lats, postprocess_lats = output_queue.get()
    except KeyboardInterrupt:
        print('exiting...')
        worker.terminate()
        output_queue.close()
        worker.join()
        print('worker joined')
        output_queue.join_thread()
        raise
    print_profile(output, prof_id, gpu_name, gpu_uuid,
                  forward_stats, preprocess_lats, postprocess_lats)

    print('joining worker...')
    worker.join()
    print('worker joined')


def generate_dataset(width, height):
    global _dataset_dir
    _dataset_dir = tempfile.mkdtemp(prefix='random-{}x{}-'.format(width, height))
    buf = np.random.randint(256, size=(224, 224, 3), dtype=np.uint8)
    img = Image.frombytes('RGB', (224, 224), buf)
    img.save(os.path.join(_dataset_dir, '{}x{}.jpg'.format(width, height)))


def profile_model(args):
    gpus = parse_int_list(args.gpu_list)
    if len(gpus) != 1 and args.gpu_uuid:
        raise ValueError('--gpu_uuid cannot be set with more than one --gpus')
    generate_dataset(args.width, args.height)

    prof_id = '%s:%s:%s' % (args.framework, args.model, args.version)
    if args.height > 0 and args.width > 0:
        prof_id += ':%sx%s' % (args.height, args.width)
    print('Start profiling %s' % prof_id)

    if args.min_batch > 0 and args.max_batch > 0:
        max_batch = args.max_batch
    else:
        max_batch = find_max_batch(args.framework, args.model, gpus)
        if args.max_batch > 0 and max_batch > args.max_batch:
            max_batch = args.max_batch
    min_batch = max(1, args.min_batch)
    print('Min batch: %s' % min_batch)
    print('Max batch: %s' % max_batch)
    print('Start profiling. This could take a long time.')

    output = prof_id.replace(':', '-') + '-' + str(uuid.uuid4()) + '.txt'
    if len(gpus) > 1:
        profile_model_concurrent(gpus, min_batch, max_batch, prof_id, output)
    else:
        profile_model_single_gpu(
            gpus[0], min_batch, max_batch, prof_id, output)
    with open(output) as f:
        next(f)
        gpu_name = next(f).strip()
        gpu_uuid = next(f).strip()

    prof_dir = os.path.join(args.model_root, 'profiles')
    assert gpu_name is not None
    gpu_dir = os.path.join(prof_dir, gpu_name)
    if args.gpu_uuid:
        gpu_dir = os.path.join(gpu_dir, gpu_uuid)
    if not os.path.exists(gpu_dir):
        os.makedirs(gpu_dir, exist_ok=True)
    prof_file = os.path.join(gpu_dir, '%s.txt' % prof_id.replace(':', '-'))
    if os.path.exists(prof_file) and not args.force:
        print('%s already exists' % prof_file)
        print('Save profile to %s' % output)
    else:
        shutil.move(output, prof_file)
        print('Save profile to %s' % prof_file)
    shutil.rmtree(_dataset_dir)


def main():
    parser = argparse.ArgumentParser(description='Profile models')
    parser.add_argument('--framework', required=True,
                        choices=['caffe', 'caffe2',
                                 'tensorflow', 'darknet', 'tf_share'],
                        help='Framework name')
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--model_root', type=str, required=True,
                        help='Nexus model root directory')
    parser.add_argument('--version', type=int, default=1,
                        help='Model version (default: 1)')
    parser.add_argument('--gpu_list', required=True,
                        help='GPU indexes. e.g. "0" or "0-2,4,5,7-8".')
    parser.add_argument('--height', type=int, required=True, help='Image height')
    parser.add_argument('--width', type=int, required=True, help='Image width')
    parser.add_argument('--prefix', action='store_true',
                        help='Enable prefix batching')
    parser.add_argument('--max_batch', type=int, default=0,
                        help='Limit the max batch size')
    parser.add_argument('--min_batch', type=int, default=0,
                        help='Limit the min batch size')
    parser.add_argument('-f', '--force', action='store_true',
                        help='Overwrite the existing model profile in model DB')
    parser.add_argument('--gpu_uuid', action='store_true', default=False,
                        help='Save profile result to a subdirectory with the GPU UUID')
    global args
    args = parser.parse_args()

    if args.prefix and args.framework == 'tf_share':
        sys.stderr.write('Cannot use -prefix with TFShare models')
        return
    load_model_db(os.path.join(args.model_root, 'db', 'model_db.yml'))
    if args.model not in _models[args.framework]:
        sys.stderr.write('%s:%s not found in model DB\n' %
                         (args.framework, args.model))
        return

    profile_model(args)


if __name__ == '__main__':
    main()
