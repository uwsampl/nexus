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
* protobuf >= 3.5.0
* [grpc](https://github.com/grpc/grpc/blob/master/INSTALL.md) >= v1.4.x
* OpenCV >= 3.0
* bazel >= 0.10.0
* CUDA >= 8.0
* CUDNN >= 6.0

Compile Nexus
-------------
```
$ git clone --recursive https://github.com/uwsaml/nexus.git
$ cd nexus
$ git submodule update --init --recursive
$ make all
```

Compile sample applications
---------------------------
In the Nexus repo, we provide a few sample applications located at `nexus/apps`.
Go to the directory at each application, e.g., nexus/apps/obj_rec/, you can
compile the application by
```
$ make
```
