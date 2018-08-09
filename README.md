Nexus
=====
Nexus is a scalable and efficient serving system for DNN applications on GPU
cluster.

Compile and Deploy Nexus
========================

Download nexus and sample models
--------------

### Download Nexus
 
Download nexus and framworks it supports from github
```
$ git clone --recursive https://github.com/uwsaml/nexus.git
$ cd nexus
$ git submodule update --init --recursive
```

### Download Model Zoo

Nexus publishes public model zoo on the Amazon S3. To download the model zoo
from S3, you need to install [AWS CLI](https://aws.amazon.com/cli/) via
`pip install awscli`, and configure AWS CLI by `aws configure`.

The configuration will prompt you to provide your AWS access key ID and
secret access key. Instructions for creating the access key pair can be found
at [AWS user guide](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-getting-started.html).

To download the Nexus public model zoo, run
```
$ mkdir nexus-models
$ aws s3 sync s3://nexus-models nexus-models/
```

Pre-requisites
--------------

Recommended development environment on Ubuntu (>=16.04)

### General dependencies
```
$ [sudo] apt-get install libboost-all-dev libgflags-dev libgoogle-glog-dev \
libgtest-dev libyaml-cpp-dev libopenblas-dev libleveldb-dev liblmdb-dev \
libhdf5-serial-dev
```

### Required libraries
* protobuf >= 3.5.0
* [grpc](https://github.com/grpc/grpc/blob/master/INSTALL.md) >= v1.4.x
* OpenCV >= 3.0
* bazel >= 0.10.0
* CUDA >= 8.0
* CUDNN >= 6.0

### Pre-requsites for Deploy Nexus Service

Nexus is able to be compiled from source, but if it is necessary to run Nexus service across multiple servers, we use the use docker swarm. So we recommand to deploy nexus by docker.

To build nexus docker images and deploy Nexus, following packages are required
* [CUDA 8.0](https://developer.nvidia.com/cuda-80-ga2-download-archive)
* [docker](https://docs.docker.com/install/linux/docker-ce/ubuntu/)(>=1.12)
* [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

Set Config for Nexus
--------------------

There is a file names Makefile.config in the nexus folder. And it stores location of cuda and cudnn. $(CUDA_PATH) should be set as your/path/to/cuda, and $(CUDNN_PATH) should be set as your/path/to/cudnn. 

There are some variables in this file.$(USE_GPU) decides wether backends of nexus need to run with GPU.

Other varaibles such as $(USE_CAFFE) decide if the corresponding frameworks nexus need to support. Nexus is not able to support caffe and caffe2 at the same time.


Local Tests without Docker
--------------------------

### Compile from source

Nexus can be compiled from source with make command
```
$ cd nexus
$ make all
```

In the Nexus repo, we provide a few sample applications located at `nexus/apps`.
Go to the directory at each application, e.g., nexus/apps/simple_app/, you can
compile the application by
```
$ cd nexus/apps/simple_app
$ make
```

### Local Tests

There are some commands to test these local sample applications. We will use simple_app as an example of these applications.

First, run the scheduler program
```
$ cd nexus
$ ./build/bin/scheduler -model_root /path/to/model/zoo
```
Then, run the backend program
```
$ ./build/bin/backend -model_root /path/to/model/zoo -gpu $(GPU_INDEX) -sch_addr 127.0.0.1
```
At last, run the sample_app, let's test vgg_face model whose framework is caffe2
```
$ cd nexus/apps/sample_app
$ ./bin/simple_app -framework caffe2 -model vgg_face
```
These three programs construct a simple nexus, it can be tested by test_client.py in nexus/tests/python directory
```
$ cd nexus/tests/python
$ python test_client.py
```

Build Docker Image
------------------

Since we recommand to use docker, there are four docker images that needs to be built.

First you need to build the base docker image that installs all dependent libraries required for Nexus.

Following docker build commands contain -t and -f options. -t option is followed 
by name and optionally a tag in the ‘name:tag’ format. -f option is followed by 
name of the Dockerfile (Default is ‘PATH/Dockerfile’).

```
$ cd nexus/dockerfiles
$ docker build -t nexus/base -f NexusBaseDockerfile .
```

Next we can build the docker images for backend, scheduler, and application
library respectively.
```
$ docker build -t nexus/backend -f NexusBackendDockerfile .
$ docker build -t nexus/scheduler -f NexusSchedulerDockerfile .
$ docker build -t nexus/applib -f NexusAppLibDockerfile .
```

### Compile sample applications

In the Nexus repo, we provide a few sample applications located at `nexus/apps`.
In each application directory, you can build the application docker image by
```
$ docker build -t nexus/app_name -f Dockerfile .
```

Nexus Deployment
----------------

### Step 1: Create a virtual cluster (optional)
If you only need to deploy Nexus on a single server, you can skip this step.
To run Nexus service across multiple servers, we use the docker swarm (requires
docker version >= 1.20) to manage a cluster.

#### Add servers to swarm cluster

First create the swarm on a master server, run
```
$ docker swarm init
```
This server serves as a manager node of the swarm.

To add worker nodes to this swarm, we need to get the token by running command
on the master node
```
$ docker swarm join-token worker
```
The example output of this command is
```
$ docker swarm join \
--token SWMTKN-1-49nj1cmql0jkz5s954yi3oex3nedyz0fb0xx14ie39trti4wxv-8vxv8rssmk743ojnwacrr2e7c \
192.168.99.100:2377
```
Copy this piece of code and run it on the worker server.

To check nodes and their roles in the swarm, run the following command on a manager node
```
$ docker node ls
```

#### Create overlay network

After the swarm is created, we then need to create a overlay network to allow
dockers in the swarm to communicate with each other. Run the following command on
a manager node.
```
$ docker network create --driver overlay --attachable --subnet 10.0.0.0/16 nexus-network
```

### Step 3: Generate model profile

In order to evaluate the maximum batch size for each model in each GPU, we need to generate profiles by profiler in tools directory. 
```
cd nexus/tools/profiler
python profiler.py $(framework) $(model) $(model_root) $(dataset)
```
The frameworks we currently support contain tensorflow, caffe, caffe2, and darknet. The value of $(framework) can be chosen from them. 
$(model) represents name of the model. 
$(model_root) is the path to model root directory. And $(dataset) is the path to the dataset. 

There is an example to generate a profile of model in our public model zoo
```
python profile.py caffe vgg_face /nexus-models/ /path/to/dataset 
``` 

There are some optional arguments of profiler. 

Running with argument -h will show help message about arguments and exit. 

Running with --version VERSION argument will designate version of the model, default value of version is 1. 

Running with --gpu GPU argument will designate the index of gpu. --height HEIGHT and --width WIDTH specify the size of the input image. 

Argument -f means to overwrite the existing model profile in model database.
 
If we add an new model, it is necessary to generate model profile for each GPU. If we add an new GPU, it is also necessary to generate model profile of each model for this new GPU.
 
### Step 3: Start Nexus service

First, we use docker to start the scheduler in a container.
Usage of docker run command is `docker run [OPTIONS] IMAGE [COMMAND] [ARG...]`.
The -d option means running container in background and printing container ID. And the -v option means to 
bind mount a volume with IMAGE.
```
$ docker run [--network nexus-network] -d -v /path/to/nexus-models:/nexus-models:ro \
--name scheduler nexus/scheduler scheduler -model_root /nexus-models
```

After the scheduler starts, we need to retrieve the IP address of scheduler. 
Use docker inspect command with -f option whose effect is to format the output using the given Go template.
```
$ docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' scheduler
```

Start a backend server on each GPU.
```
$ docker run [--network nexus-network] --runtime=nvidia -d \
-v /path/to/nexus-models:/nexus-models:ro --name backend0 nexus/backend backend \
-model_root /nexus-models -sch_addr $(scheduler IP) -gpu $(gpu index)
```

Start an application serving at port 12345, e.g., object recognition.
-p 12345:9001 argument means to map local port 12345 to docker port 9001. 
```
$ docker run [--network nexus-network] -d -p 12345:9001 --name obj_rec \
nexus/obj_rec /app/bin/obj_rec -sch_addr $(scheduler IP)
```

### Step 4: Test sample
There is a sample for test at `nexus/tests/python`. Before running the file for test, we need to generate a library under the `nexus/python/proto` dirtory. Then set the PYTHONPATH environment variable.
```
$ make python
$ cd tests/python
$ export PYTHONPATH=/path/to/nexus/python:$PYTHONPATH
```
