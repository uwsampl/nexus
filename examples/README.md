# Running Nexus with the Simple Example

We have prepared a simple example application to walk you through how to run
Nexus in concrete steps. Before we start, make sure you have followed the
[building instructions](../BUILDING.md) and have Nexus and its dependencies
built. Let's assume the Nexus locates at `$NEXUS_DIR` and the model zoo locates
at `$MODEL_DIR`.

## Download Model Zoo

```bash
git clone https://gitlab.cs.washington.edu/syslab/nexus-models $MODEL_DIR
cd $MODEL_DIR
git lfs checkout
```

## Profile ResNet-50 on GPU 0

```bash
python3.7 $NEXUS_DIR/tools/gen-random-pic.py --dirname=/tmp/random-224x224 \
    --width=224 --height=224 --number=100
python3.7 $NEXUS_DIR/tools/profiler/profiler.py --gpu_list=0 --gpu_uuid \
    --framework=tensorflow --model=resnet_0 \
    --model_root=$MODEL_ROOT --dataset=/tmp/random-224x224/
rm -rf /tmp/random-224x224
```

## Run Nexus Scheduler and Backend, and Application Frontend

```bash
$NEXUS_DIR/build/scheduler -model_root=$MODEL_ROOT -alsologtostderr -colorlogtostderr -v 1
$NEXUS_DIR/build/backend -model_root=$MODEL_ROOT -gpu=0 -alsologtostderr -colorlogtostderr
$NEXUS_DIR/build/simple -framework=tensorflow -model=resnet_0 -latency=50 -alsologtostderr -colorlogtostderr
```

## Send a Client Request

```bash
curl https://upload.wikimedia.org/wikipedia/commons/4/4c/Chihuahua1_bvdb.jpg | python3.7 ./examples/simple_app/src/client.py -
```

The image should be classified as a *chihuahua*.
