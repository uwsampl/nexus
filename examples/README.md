# Running Nexus with the Simple Example

We have prepared a simple example application to walk you through how to run
Nexus in concrete steps. We provide a Docker image so that you don't have to
spend hours on building the dependencies. To download the Docker image, you
can run:

```bash
docker pull abcdabcd987/nexus
```

If you want to run Nexus on the host OS, make sure you have followed the
[building instructions](../BUILDING.md) and have Nexus and its dependencies
built. The commands in the following sections assumes Docker, but to run it
on the host, you can simply drop lines containing `docker`, omit the command
line arguments that specifies server address, and replace `/nexus` with the
path to your Nexus build.

## Download Model Zoo

```bash
git clone https://gitlab.cs.washington.edu/syslab/nexus-models
cd nexus-models
export MODEL_DIR=$(pwd)
git lfs checkout
```

## Profile ResNet-50 on GPU 0

```bash
docker run -it --rm --gpus all -v $MODEL_DIR:$MODEL_DIR abcdabcd987/nexus \
    python3.7 /nexus/tools/profiler/profiler.py --gpu_list=0 --gpu_uuid --model_root=$MODEL_DIR
        --framework=tensorflow --model=resnet_0 --width=224 --height=224
```

## Run Nexus Scheduler and Backend, and Application Frontend

```bash
docker network create nexus-net

docker run -it --rm --gpus all --network=nexus-net -v=$MODEL_DIR:$MODEL_DIR --name=nexus-scheduler -p=10001 abcdabcd987/nexus \
    /nexus/build/scheduler  -model_root=$MODEL_DIR -alsologtostderr -colorlogtostderr -v 1

docker run -it --rm --gpus all --network=nexus-net -v=$MODEL_DIR:$MODEL_DIR --name=nexus-gpu0 -p=8001 -p=8002 abcdabcd987/nexus \
    /nexus/build/backend -model_root=$MODEL_DIR -gpu=0 -alsologtostderr -colorlogtostderr \
                         -sch_addr=nexus-scheduler:10001

docker run -it --rm --gpus all --network=nexus-net --name=nexus-simple-frontend -p=9001 -p=9002 abcdabcd987/nexus \
    /nexus/build/simple -framework=tensorflow -model=resnet_0 -latency=50 -width=224 -height=224 -alsologtostderr -colorlogtostderr \
                        -sch_addr=nexus-scheduler:10001
```

## Send a Client Request

```bash
curl https://upload.wikimedia.org/wikipedia/commons/4/4c/Chihuahua1_bvdb.jpg | docker run --rm -i --network=nexus-net abcdabcd987/nexus \
    python3.7 /nexus /examples/simple_app/src/client.py - --server=nexus-simple-frontend:9001
```

The [image](https://upload.wikimedia.org/wikipedia/commons/4/4c/Chihuahua1_bvdb.jpg)
should be classified as a *chihuahua*.
