import os
import sys
import argparse
import subprocess
import shutil
import uuid
import yaml


_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
_profiler = os.path.join(_root, 'build/bin/profiler')
_models = {}


def load_model_db(path):
    with open(path) as f:
        model_db = yaml.load(f.read())
    for model_info in model_db['models']:
        framework = model_info['framework']
        model_name = model_info['model_name']
        if framework not in _models:
            _models[framework] = {}
        _models[framework][model_name] = model_info


def find_max_batch(framework, model_name):
    global args
    cmd_base = '%s -model_root %s -image_dir %s -gpu %s -framework %s -model %s -model_version %s' % (
        _profiler, args.model_root, args.dataset, args.gpu, framework, model_name, args.version)
    if args.height > 0 and args.width > 0:
        cmd_base += ' -height %s -width %s' % (args.height, args.width)
    if args.prefix:
        cmd_base += ' -share_prefix'
    left = 1
    right = 64
    curr_tp = None
    out_of_memory = False
    while True:
        prev_tp = curr_tp
        curr_tp = None
        cmd = cmd_base + ' -min_batch %s -max_batch %s' % (right, right)
        print(cmd)
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
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
        if right == 1024:
            break
        left = right
        right *= 2
    if not out_of_memory:
        return right
    while right - left > 1:
        print(left, right)
        mid = (left + right) / 2
        cmd = cmd_base + ' -min_batch %s -max_batch %s' % (mid, mid)
        print(cmd)
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        out, err = proc.communicate()
        if 'out of memory' in err or 'out of memory' in out:
            right = mid
        else:
            left = mid
    # minus 2 just to be conservative in case framework has memory leak
    return left - 2


def profile_model(framework, model_name):
    global args
    prof_id = '%s:%s:%s' % (framework, model_name, args.version)
    if args.height > 0 and args.width > 0:
        prof_id += ':%sx%s' % (args.height, args.width)
    print('Profile %s' % prof_id)

    max_batch = find_max_batch(framework, model_name)
    print('Max batch: %s' % max_batch)

    output = str(uuid.uuid4()) + '.txt'
    cmd = '%s -model_root %s -image_dir %s -gpu %s -framework %s -model %s -model_version %s -max_batch %s -output %s' % (
        _profiler, args.model_root, args.dataset, args.gpu, framework, model_name, args.version,
        max_batch, output)
    if args.height > 0 and args.width > 0:
        cmd += ' -height %s -width %s' % (args.height, args.width)
    if args.prefix:
        cmd += ' -share_prefix'
    print(cmd)

    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    proc.communicate()
    if not os.path.exists(output):
        lines = err.split('\n')
        sys.stderr.write('\n'.join(lines[-50:]))
        sys.stderr.write('\n')
        return
    with open(output) as f:
        pid = f.readline().strip().replace(':', '-')
        gpu = f.readline().strip().replace('(', '').replace(')','')
    prof_dir = os.path.join(args.model_root, 'profiles')
    gpu_dir = os.path.join(prof_dir, gpu)
    if not os.path.exists(gpu_dir):
        os.mkdir(gpu_dir)
    prof_file = os.path.join(gpu_dir, '%s.txt' % pid)
    if os.path.exists(prof_file) and not args.force:
        print('%s already exists' % prof_file)
        return
    shutil.move(output, prof_file)
    print('Save profile to %s' % prof_file)


def main():
    parser = argparse.ArgumentParser(description='Profile models')
    parser.add_argument('framework',
                        choices=['caffe', 'caffe2', 'tensorflow', 'darknet'],
                        help='Framework name')
    parser.add_argument('model', type=str, help='Model name')
    parser.add_argument('model_root', type=str,
                        help='Nexus model root directory')
    parser.add_argument('dataset', type=str,
                        help='Dataset directory')
    parser.add_argument('--version', type=int, default=1,
                        help='Model version (default: 1)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index')
    parser.add_argument('--height', type=int, default=0, help='Image height')
    parser.add_argument('--width', type=int, default=0, help='Image width')
    parser.add_argument('--prefix', action='store_true',
                        help='Enable prefix batching')
    parser.add_argument('-f', '--force', action='store_true',
                        help='Overwrite the existing model profile in model DB')
    global args
    args = parser.parse_args()

    load_model_db(os.path.join(args.model_root, 'db', 'model_db.yml'))
    if args.model not in _models[args.framework]:
        sys.stderr.write('%s:%s not found in model DB\n' % (args.framework, args.model))
        return

    profile_model(args.framework, args.model)
    

if __name__ == '__main__':
    main()
