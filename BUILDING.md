# Building Nexus on Ubuntu 18.04

## Install system-wide packages

```bash
# Build system and utilities
sudo apt-get install -y unzip build-essential git autoconf automake libtool pkg-config curl make zlib1g-dev wget

# For OpenCV
sudo apt-get install -y libswscale-dev libjpeg-dev libpng-dev

# Python 2.7 for building Tensorflow
sudo apt-get install -y python-dev python-pip
pip install --upgrade --user pip six numpy wheel setuptools mock 'future>=0.17.1'

# Python 3.7 for Nexus
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y python3.7 python3.7-dev
curl https://bootstrap.pypa.io/get-pip.py | python3.7
python3.7 -m pip install --upgrade --user numpy protobuf Pillow pyyaml

# CMake > 3.12
# See https://apt.kitware.com/ for more details.
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | sudo apt-key add -
sudo apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
sudo apt-get update
sudo apt-get install -y cmake
```

## Install NVIDIA driver

```bash
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:graphics-drivers/ppa
sudo apt-get update
sudo apt-get install -y nvidia-headless-440
```

## Install CUDA 10.0

```bash
wget -n -O cuda_10.0.130_410.48_linux.run https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda_10.0.130_410.48_linux
sudo sh cuda_10.0.130_410.48_linux.run -silent -toolkit
sudo unlink /usr/local/cuda
```

## Install cuDNN 7.6.5

Download cuDNN 7.6.5 for CUDA 10.0 from [NVIDIA](https://developer.nvidia.com/rdp/cudnn-download)

```bash
tar xf cudnn-10.0-linux-x64-v7.6.5.32.tgz
sudo mv cuda/include/cudnn.h /usr/local/cuda-10.0/include
sudo mv cuda/lib64/libcudnn* /usr/local/cuda-10.0/lib64
sudo chmod a+r /usr/local/cuda-10.0/include/cudnn.h /usr/local/cuda-10.0/lib64/libcudnn*
sudo ldconfig
```

## Clone Nexus

```bash
git clone https://github.com/uwsampl/nexus.git
cd nexus
```

## Build Nexus dependencies

```bash
./build-deps.bash
./build-tensorflow.bash
```

By default, the script will build TensorFlow with the following
[CUDA compute capabilities](https://en.wikipedia.org/wiki/CUDA#GPUs_supported):
`5.2, 6.1, 7.5`. If you want to change any build options
for TensorFlow, set the environment variables specified in
[`./build-tensorflow.bash`](build-tensorflow.bash)

## Build Nexus

```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebugInfo -DCUDA_PATH=/usr/local/cuda-10.0 -DUSE_TENSORFLOW=ON -DUSE_CAFFE2=OFF
make -j$(nproc)
python3.7 -m pip install --user --editable ./python
```
