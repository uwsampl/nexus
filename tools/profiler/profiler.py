import os
import argparse
import subprocess
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
    cmd_base = '%s -model_root %s -image_dir %s -gpu %s -framework %s -model %s' % (
        _profiler, args.model_dir, args.dataset, args.gpu, framework, model_name)
    left = 1
    right = 64
    curr_tp = None
    out_of_memory = False
    while True:
        prev_tp = curr_tp
        cmd = cmd_base + ' -min_batch %s -max_batch %s' % (right, right)
        print(cmd)
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        out, err = proc.communicate()
        if 'out of memory' in err:
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
            if line.startswith('batch'):
                flag = True
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
        if 'out of memory' in err:
            right = mid
        else:
            left = mid
    return left


def profile_model(framework, model_name):
    global args
    print('Profile %s:%s' % (framework, model_name))
    max_batch = find_max_batch(framework, model_name)
    print('Max batch: %s' % max_batch)
    cmd = '%s -model_root %s -image_dir %s -gpu %s -framework %s -model %s -max_batch %s -output %s:%s:%s.txt' % (
        _profiler, args.model_dir, args.dataset, args.gpu, framework, model_name,
        max_batch, framework, model_name, args.version)
    print(cmd)
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    proc.communicate()


def main():
    parser = argparse.ArgumentParser(description='Profile models')
    parser.add_argument('-f', '--framework',
                        choices=['caffe', 'tensorflow', 'darknet'],
                        help='Framework name')
    parser.add_argument('-m', '--model', type=str, help='Model name')
    parser.add_argument('-v', '--version', type=int, default=1,
                        help='Model version')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index')
    parser.add_argument('--dataset', type=str,
                        default='/home/haichen/datasets/imagenet/ILSVRC2012/val',
                        help='Dataset directory')
    parser.add_argument('--model_dir', type=str,
                        default='/home/haichen/nexus-models',
                        help='Model root directory')
    global args
    args = parser.parse_args()

    load_model_db(os.path.join(args.model_dir, 'db', 'model_db.yml'))
    if args.framework:
        frameworks = [args.framework]
    else:
        frameworks = _models.keys()
    for framework in frameworks:
        if args.model:
            if args.model not in _models[framework]:
                continue
            profile_model(framework, args.model)
        else:
            for model in _models[framework]:
                profile_model(framework, model)
    

if __name__ == '__main__':
    main()
