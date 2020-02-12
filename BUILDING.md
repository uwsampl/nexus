# Building Nexus on Ubuntu 18.04

## Install system-wide packages

```bash
# Build system and utilities
sudo apt-get install -y unzip build-essential git autoconf automake libtool pkg-config curl make zlib1g-dev wget

# For OpenCV
sudo apt-get install -y libswscale-dev libjpeg-dev libpng-dev

# Python 3.7 for Nexus
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get install -y python3.7 python3.7-dev
curl https://bootstrap.pypa.io/get-pip.py | python3.7
python3.7 -m pip install --upgrade --user numpy matplotlib protobuf grpcio opencv-python pyyaml six

# CMake 3.16.3 (install globally)
# ref: https://cmake.org/install/
# ref: https://stackoverflow.com/questions/44633043/cmake-libcurl-was-built-with-ssl-disabled-https-not-supported
sudo apt-get install -y libcurl4-openssl-dev
wget https://github.com/Kitware/CMake/releases/download/v3.16.3/cmake-3.16.3.tar.gz
tar xf cmake-3.16.3.tar.gz
cd cmake-3.16.3
./bootstrap --system-curl --parallel=$(nproc)
make -j$(nproc)
sudo make install
cd ..
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

## Build Nexus

```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebugInfo -DCUDA_PATH=/usr/local/cuda-10.0 -DUSE_TENSORFLOW=ON -DUSE_CAFFE2=OFF
make -j$(nproc)
```
