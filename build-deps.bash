#!/bin/bash
set -e
set -x

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
SRC_DIR="$SCRIPT_DIR/build-dep-src"
INSTALL_DIR="$SCRIPT_DIR/build-dep-install"

mkdir -p "$SRC_DIR"
mkdir -p "$INSTALL_DIR"

cd "$SRC_DIR"


###### yaml-cpp 0.6.3 ######
# ref: https://github.com/jbeder/yaml-cpp/tree/yaml-cpp-0.6.3
if [ ! -d "$INSTALL_DIR/yaml-cpp" ]; then
    wget -N https://github.com/jbeder/yaml-cpp/archive/yaml-cpp-0.6.3.tar.gz -O yaml-cpp-0.6.3.tar.gz
    tar xf yaml-cpp-0.6.3.tar.gz
    cd yaml-cpp-yaml-cpp-0.6.3/
    mkdir build
    cd build
    cmake .. -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR/yaml-cpp" -DYAML_BUILD_SHARED_LIBS=ON
    make -j$(nproc)
    make install
    cd ../..
fi


###### Boost 1.72.0 ######
# ref: https://www.boost.org/doc/libs/1_72_0/more/getting_started/unix-variants.html
if [ ! -d "$INSTALL_DIR/boost" ]; then
    wget -N https://dl.bintray.com/boostorg/release/1.72.0/source/boost_1_72_0.tar.gz -O boost_1_72_0.tar.gz
    tar xf boost_1_72_0.tar.gz
    cd boost_1_72_0
    ./bootstrap.sh --prefix="$INSTALL_DIR/boost"
    ./b2 install --prefix="$INSTALL_DIR/boost" --without-python --without-mpi --layout=system -j$(nproc)
    cd ..
fi


###### OpenCV 4.2.0 ######
# ref: https://docs.opencv.org/4.2.0/d7/d9f/tutorial_linux_install.html
if [ ! -d "$INSTALL_DIR/opencv" ]; then
    wget -N https://github.com/opencv/opencv/archive/4.2.0.zip -O opencv-4.2.0.zip
    unzip opencv-4.2.0.zip
    cd opencv-4.2.0
    mkdir build
    cd build
    cmake .. -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR/opencv" -DOPENCV_GENERATE_PKGCONFIG=ON -DBUILD_opencv_python2=OFF -DBUILD_opencv_python3=OFF -DBUILD_JAVA=OFF -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DWITH_QT=OFF -DWITH_GTK=OFF -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DBUILD_LIST=core,imgcodecs,imgproc
    make -j$(nproc)
    make install
    cd ../..
fi


###### gflags 2.2.2 ######
# ref: https://github.com/gflags/gflags/blob/master/INSTALL.md
if [ ! -d "$INSTALL_DIR/gflags" ]; then
    wget -N https://github.com/gflags/gflags/archive/v2.2.2.tar.gz -O gflags-2.2.2.tar.gz
    tar xf gflags-2.2.2.tar.gz
    cd gflags-2.2.2
    mkdir build
    cd build
    cmake .. -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR/gflags" -DBUILD_SHARED_LIBS=ON
    make -j$(nproc)
    make install
    cd ../..
fi


###### glog 0.4.0 ######
# ref: https://github.com/google/glog/blob/master/cmake/INSTALL.md
if [ ! -d "$INSTALL_DIR/glog" ]; then
    wget -N https://github.com/google/glog/archive/v0.4.0.tar.gz -O glog-0.4.0.tar.gz
    tar xf glog-0.4.0.tar.gz
    cd glog-0.4.0
    mkdir build
    cd build
    cmake .. -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR/glog" -DBUILD_SHARED_LIBS=ON
    make -j$(nproc)
    make install
    cd ../..
fi


###### gtest 1.10.0 ######
# ref: https://github.com/google/googletest/blob/master/googletest/README.md
if [ ! -d "$INSTALL_DIR/gtest" ]; then
    wget -N https://github.com/google/googletest/archive/release-1.10.0.tar.gz -O googletest-release-1.10.0.tar.gz
    tar xf googletest-release-1.10.0.tar.gz
    cd googletest-release-1.10.0
    mkdir build
    cd build
    cmake .. -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR/gtest" -DBUILD_SHARED_LIBS=OFF
    make -j$(nproc)
    make install
    cd ../..
fi

###### protobuf 3.8.0 (need to sync with TensorFlow) ######
# TensorFlow dependency: https://github.com/tensorflow/tensorflow/blob/906f537c0be010929a0bda3c7d061de9d3d8d5b0/tensorflow/workspace.bzl#L472
# ref: https://github.com/protocolbuffers/protobuf/blob/master/src/README.md
if [ ! -d "$INSTALL_DIR/protobuf" ]; then
    wget -N https://github.com/protocolbuffers/protobuf/archive/v3.8.0.tar.gz -O protobuf-3.8.0.tar.gz
    tar xf protobuf-3.8.0.tar.gz
    cd protobuf-3.8.0/
    ./autogen.sh
    ./configure --prefix="$INSTALL_DIR/protobuf"
    make -j$(nproc)
    make install
    cd ..
fi

###### grpc 1.27.0 ######
# ref: https://github.com/grpc/grpc/blob/master/BUILDING.md
if [ ! -d "$INSTALL_DIR/grpc" ]; then
    if [ ! -d grpc-1.27 ]; then
        git clone -b v1.27.0 --depth 1 https://github.com/grpc/grpc grpc-1.27
        cd grpc-1.27
        git submodule update --init
    else
        cd grpc-1.27
    fi
    make -j$(nproc)
    make install --prefix="$INSTALL_DIR/grpc"
    cd ..
fi

###### bazel 1.2.1 ######
if [ ! -d "$INSTALL_DIR/bazel" ]; then
    wget -N https://github.com/bazelbuild/bazel/releases/download/1.2.1/bazel-1.2.1-linux-x86_64 -O bazel-1.2.1-linux-x86_64
    chmod +x bazel-1.2.1-linux-x86_64
    mkdir -p "$INSTALL_DIR/bazel"
    mv bazel-1.2.1-linux-x86_64 "$INSTALL_DIR/bazel/"
    ln -sf bazel-1.2.1-linux-x86_64 "$INSTALL_DIR/bazel/bazel"
fi

exit 0
