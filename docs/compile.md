Compile Nexus from Source
=========================

Pre-requisites
--------------

Recommended development environment on Ubuntu (>=16.04)

General dependencies
```
$ [sudo] apt-get install libboost-all-dev libgflags-dev libgoogle-glog-dev \
libgtest-dev libyaml-cpp-dev libopenblas-dev libleveldb-dev liblmdb-dev \
libhdf5-serial-dev
```

Required libraries
* protobuf >= 3.2.0
* [grpc](https://github.com/grpc/grpc/blob/master/INSTALL.md) >= v1.4.x
* OpenCV >= 3.0
* CUDA 8.0

Compile Nexus
-------------
```
$ git clone https://github.com/uwsaml/nexus.git
$ cd nexus
$ git submodule update --init
$ make all
```

Build docker image
------------------
This step requires the installation of [docker](https://docs.docker.com/install/linux/docker-ce/ubuntu/)(>=1.12).

There are four docker images that needs to be built. First we need to build the
base docker image that installs all dependent libraries required for Nexus.
```
$ cd nexus/dockerfiles
$ docker build -t nexus/base -f NexusBaseDockerfile .
```

Next we can the docker images for backend, scheduler, and application library
respectively.
```
$ docker build -t nexus/base -f NexusBackendDockerfile .
$ docker build -t nexus/base -f NexusSchedulerDockerfile .
$ docker build -t nexus/base -f NexusAppLibDockerfile .
```

Compile sample applications
---------------------------
In the Nexus repo, we provide a few sample applications located at `nexus/apps`.
In each application, you can compile the application by running `make` under the
application directory. You can also build the application docker image by
```
$ docker build -t nexus/app_name -f Dockerfile .
```
