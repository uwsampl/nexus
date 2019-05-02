# INSTALL on Ubuntu 16.04

```bash
sudo apt-get install -y unzip build-essential git autoconf automake libtool curl make zlib1g-dev
mkdir -p download
cd download

# CUDA 10.0
wget https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda_10.0.130_410.48_linux
chmod +x cuda_10.0.130_410.48_linux
sudo ./cuda_10.0.130_410.48_linux -silent -toolkit
sudo unlink /usr/local/cuda

# cuDNN v7.5.0
# ref: https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#installlinux-tar
# Sign in and download from https://developer.nvidia.com/rdp/cudnn-download
tar xf cudnn-10.0-linux-x64-v7.5.0.56.tgz
sudo cp cuda/include/cudnn.h /usr/local/cuda-10.0/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda-10.0/lib64
sudo chmod a+r /usr/local/cuda-10.0/include/cudnn.h /usr/local/cuda-10.0/lib64/libcudnn*
sudo ldconfig

# CMake 3.14.0
# ref: https://cmake.org/install/
# ref: https://stackoverflow.com/questions/44633043/cmake-libcurl-was-built-with-ssl-disabled-https-not-supported
sudo apt-get install -y libcurl4-openssl-dev
wget https://github.com/Kitware/CMake/releases/download/v3.14.0/cmake-3.14.0.tar.gz
tar xf cmake-3.14.0.tar.gz
cd cmake-3.14.0
./bootstrap --system-curl --parallel=$(nproc)
make -j$(nproc)
sudo make install
cd ..

# yaml-cpp 0.6.2
# ref: https://github.com/jbeder/yaml-cpp/tree/yaml-cpp-0.6.2
wget https://github.com/jbeder/yaml-cpp/archive/yaml-cpp-0.6.2.tar.gz
tar xf yaml-cpp-0.6.2.tar.gz
cd yaml-cpp-yaml-cpp-0.6.2/
mkdir build
cd build
cmake .. -DBUILD_SHARED_LIBS=ON
make -j$(nproc)
sudo make install
cd ../..

# Boost 1.69.0
# ref: https://www.boost.org/doc/libs/1_69_0/more/getting_started/unix-variants.html
wget https://dl.bintray.com/boostorg/release/1.69.0/source/boost_1_69_0.tar.gz
tar xf boost_1_69_0.tar.gz
cd boost_1_69_0
./bootstrap.sh
./b2 --without-python --without-mpi --layout=system -j$(nproc)
sudo ./b2 install --without-python --without-mpi --layout=system -j$(nproc)
cd ..

# OpenCV 4.0.1
# ref: https://docs.opencv.org/4.0.1/d7/d9f/tutorial_linux_install.html
sudo apt-get install -y libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
sudo apt-get install -y python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev
wget https://github.com/opencv/opencv/archive/4.0.1.zip -O opencv-4.0.1.zip
unzip opencv-4.0.1.zip
cd opencv-4.0.1
mkdir build
cd build
cmake .. -DOPENCV_GENERATE_PKGCONFIG=ON -DBUILD_opencv_python2=OFF -DBUILD_opencv_python3=OFF -DBUILD_JAVA=OFF -DBUILD_WITH_DEBUG_INFO=ON -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF
make -j$(nproc)
sudo make install
cd ../..

# gflags 2.2.2
# ref: https://github.com/gflags/gflags/blob/master/INSTALL.md
wget https://github.com/gflags/gflags/archive/v2.2.2.tar.gz -O gflags-2.2.2.tar.gz
tar xf gflags-2.2.2.tar.gz
cd gflags-2.2.2
mkdir build
cd build
cmake .. -DBUILD_SHARED_LIBS=ON
make -j$(nproc)
sudo make install
cd ../..

# glog 0.3.5
# ref: https://github.com/google/glog/blob/master/cmake/INSTALL.md
wget https://github.com/google/glog/archive/v0.3.5.tar.gz -O glog-0.3.5.tar.gz
tar xf glog-0.3.5.tar.gz
cd glog-0.3.5
mkdir build
cd build
cmake .. -DBUILD_SHARED_LIBS=ON
make -j$(nproc)
sudo make install
cd ../..

# gtest 1.8.1
# ref: https://github.com/google/googletest/blob/master/googletest/README.md
wget https://github.com/google/googletest/archive/release-1.8.1.tar.gz -O googletest-release-1.8.1.tar.gz
tar xf googletest-release-1.8.1.tar.gz
cd googletest-release-1.8.1
mkdir build
cd build
cmake .. -DBUILD_SHARED_LIBS=OFF
make -j$(nproc)
sudo make install
cd ../..

# protobuf 3.6.1.2 (need to sync with TensorFlow)
# ref: https://github.com/protocolbuffers/protobuf/blob/master/src/README.md
wget https://github.com/protocolbuffers/protobuf/archive/v3.6.1.2.tar.gz -O protobuf-3.6.1.2.tar.gz
tar xf protobuf-3.6.1.2.tar.gz
cd protobuf-3.6.1.2/
./autogen.sh
./configure
make -j$(nproc)
sudo make install
sudo ldconfig
cd ..

# grpc 1.19.1
# ref: https://github.com/grpc/grpc/blob/master/BUILDING.md
git clone -b v1.19.1 https://github.com/grpc/grpc grpc-1.19.1
cd grpc-1.19.1
git submodule update --init
make -j$(nproc)
sudo make install
cd ..

# bazel 0.21.0
# ref: https://docs.bazel.build/versions/master/install-ubuntu.html
wget https://github.com/bazelbuild/bazel/releases/download/0.21.0/bazel-0.21.0-installer-linux-x86_64.sh
chmod +x bazel-0.21.0-installer-linux-x86_64.sh
sudo ./bazel-0.21.0-installer-linux-x86_64.sh
bazel version

# gcc 8.3
# ref: https://solarianprogrammer.com/2016/10/07/building-gcc-ubuntu-linux/
wget https://ftpmirror.gnu.org/gcc/gcc-8.3.0/gcc-8.3.0.tar.gz
tar xf gcc-8.3.0.tar.gz
cd gcc-8.3.0
contrib/download_prerequisites
cd ../
mkdir gcc-8.3.0-build
cd gcc-8.3.0-build
../gcc-8.3.0/configure -v --build=x86_64-linux-gnu --host=x86_64-linux-gnu --target=x86_64-linux-gnu --enable-checking=release --enable-languages=c,c++ --disable-multilib --program-suffix=-8.3
make -j$(nproc)
sudo make install
cd ..
echo "/usr/local/lib64" | sudo tee /etc/ld.so.conf.d/local-lib64.conf
sudo ldconfig

# python 3.7 by pyenv
sudo apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python-openssl git
curl https://pyenv.run | bash
echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
pyenv install 3.7.2
pyenv global 3.7.2
python --version

# python packages
python -m pip install pip --upgrade
python -m pip install numpy matplotlib protobuf grpcio opencv-python --upgrade


# eigen (for caffe2)
wget http://bitbucket.org/eigen/eigen/get/3.3.7.tar.gz -O eigen-3.3.7.tar.gz
tar xf eigen-3.3.7.tar.gz
cd eigen-eigen-323c052e1731
mkdir build
cd build
cmake ..
make -j$(nproc)
sudo make install
cd ../..

# YAY!!! FINALLY US!!! nexus
cd ..
git clone git@github.com:abcdabcd987/nexus.git -b lqchen
cd nexus
git submodule update --init --recursive
mkdir build
cmake .. -DCMAKE_CXX_COMPILER=g++-8.3 -DCMAKE_BUILD_TYPE=Debug -DUSE_GPU=ON -DCUDA_PATH=/usr/local/cuda-10.0 -DUSE_TENSORFLOW=ON -DUSE_CAFFE=OFF -DUSE_CAFFE2=OFF -DUSE_DARKNET=OFF
make -j$(nproc)
```
