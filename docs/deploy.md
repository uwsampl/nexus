Deploy Nexus Service
====================

Pre-requisites
--------------
To deploy Nexus, following packages are required
* [CUDA 8.0](https://developer.nvidia.com/cuda-80-ga2-download-archive)
* [docker](https://docs.docker.com/install/linux/docker-ce/ubuntu/)
* [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

Deployment
----------
### Step 1: Download Model Zoo
Nexus publishes public model zoo on the Amazon S3. To download the model zoo
from S3, you need to install [AWS CLI](https://aws.amazon.com/cli/) via
`pip install awscli`, and configure AWS CLI by
```
$ aws configure
AWS Access Key ID [None]: AKIAIOSFODNN7EXAMPLE
AWS Secret Access Key [None]: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
Default region name [None]:
Default output format [None]:
```
During the configuration, you need to provide your AWS access key ID and
secret access key. Instructions for creating the access key pair can be found
on [AWS website](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-getting-started.html).

To download the Nexus public model zoo, run
```
$ mkdir nexus-models
$ aws s3 sync s3://nexus-models nexus-models/
```

### Step 2: Create a virtual cluster (optional)
If you only need to deploy Nexus on a single server, you can skip this step.
To run Nexus service across multiple servers, we use the docker swarm (requires
docker version >= 1.20) to manage a cluster.

#### Add servers to swarm cluster

First create the swarm on a master server, run
```
docker swarm init
```
This server serves as a manager node of the swarm.

To add worker nodes to this swarm, we need to get the token by running command
on the master node
```
docker swarm join-token worker
```
The example output of this command is
```
docker swarm join \
--token SWMTKN-1-49nj1cmql0jkz5s954yi3oex3nedyz0fb0xx14ie39trti4wxv-8vxv8rssmk743ojnwacrr2e7c \
192.168.99.100:2377
```
Copy this piece of code and run it on the worker server.

To check nodes and their roles in the swarm, run the following command on a manager node
```
docker node ls
```

#### Create overlay network

After the swarm is created, we then need to create a overlay network to allow
dockers in the swarm to communicate with each other. Run the following command on
a manager node.
```
docker network create --driver overlay --attachable --subnet 10.0.0.0/16 nexus-network
```

### Step 3: Start Nexus service
First, start the scheduler
```
$ docker run [--network nexus-network] -d -v /path/to/nexus-models:/nexus-models:ro --name scheduler nexus/scheduler scheduler -config /nexus/config/scheduler_docker.yml
```
After the scheduler starts, we need to retrieve the IP address of scheduler.
```
$ docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' scheduler
```
Start a backend server on each GPU.
```
$ docker run [--network nexus-network] --runtime=nvidia -d -v /path/to/nexus-models:/nexus-models:ro --name backend0 nexus/backend backend -config /nexus/config/backend_docker.yml -sch_addr $(scheduler IP):10001 -gpu $(gpu index)
```
Start an application serving at port 12345, e.g., object recognition.
```
$ docker run [--network nexus-network] -d -p 12345:9001 --name objrec nexus/obj_rec /app/bin/obj_rec -sch_addr $(scheduler IP):10001
```
