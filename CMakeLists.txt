cmake_minimum_required(VERSION 3.11)
project(nexus)
message("***start compiling nexus...")

#---DEBUG---
set(PROJECT_BINARY_DIR ${PROJECT_SOURCE_DIR}/build)
message("Source dir = ${PROJECT_SOURCE_DIR}")
message("Binary dir = ${PROJECT_BINARY_DIR}")
SET(CMAKE_VERBOSE_MAKEFILE ON)
#-----------
set(CMAKE_CXX_STANDARD 11)

#---set---verison---
set(NEXUS_VERSION_MAJOR 0)
set(NEXUS_VERSION_MINOR 0)
set(NEXUS_VERSION_PATCH 0)
set(NEXUS_VERSION "${NEXUS_VERSION_MAJOR}.${NEXUS_VERSION_MINOR}.${NEXUS_VERSION_PATCH}")

message("***Compiling is Nexus ${NEXUS_VERSION} ...\n")

#=== set === config ===
#---set----CUDA&CUDNN---
set(CUDA_PATH /usr/local/cuda)
set(CUDNN_PATH /usr/local/cuda)

#---set---options---
#USE_CAFFE and USE_CAFFE2 cannot be set to 1 at the same time
option(USE_DARKNET "Enable to support darknet" OFF)
option(USE_CAFFE2 "Enable to support caffe2" ON)
option(USE_CAFFE "Enable to support caffe" OFF)
option(USE_TENSORFLOW "Enable to support tensorflow" OFF)
option(USE_GPU "Enable to run in GPU" ON)
#================== 

#=== Translate === Defination === In === Makefile ===\
set(ROOTDIR ${PROJECT_SOURCE_DIR})

set(TEST_BIN build/bin/test)

#====================================================

#=== set === c++ === config ===
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(WARNING "-Wall -Wfatal-errors -Wno-unused -Wno-unused-result")
#shell: "pkg-config --cflags protobuf"
set(CMAKE_PROTOBUF_FLAGS "-lpthread -I/usr/local/include")

set(CMAKE_CXX_FLAGS "-std=c++11 -O3 -fPIC ${WARNING} -Isrc -Ibuild/gen -MMD -MP ${CMAKE_PROTOBUF_FLAGS}" )

#shell: "pkg-config --libs opencv"
set(OPENCV_FLAGS "-L/usr/local/lib -L/usr/local/share/OpenCV/3rdparty/lib -lopencv_shape -lopencv_stitching -lopencv_objdetect -lopencv_superres -lopencv_videostab -lopencv_calib3d -lopencv_features2d -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs -lopencv_video -lopencv_photo -lopencv_ml -lopencv_imgproc -lopencv_flann -lopencv_viz -lippicv -lopencv_core")

#shell: "pkg-config --libs protobuf"
set(PROTOBUF_FLAGS "-L/usr/local/lib -lprotobuf -lpthread")

#shell: "pkg-config --libs grpc++ grpc"
set(GRPC_FLAGS "-L/usr/local/lib -lgrpc++ -lgrpc")

set(LD_FLAGS "-lm -pthread -lglog -lgflags -lgtest -lgtest_main -lboost_system -lboost_thread -lboost_filesystem -lyaml-cpp ${OPENCV_FLAGS} ${PROTOBUF_FLAGS} ${GRPC_FLAGS}")
set(DLL_LINK_FLAGS "-shared")
if (USE_GPU)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${CUDA_PATH}/include")
	set(LD_FLAGS "${LD_FLAGS} -L${CUDA_PATH}/lib64") 
	set(LD_FLAGS "${LD_FLAGS} -L${CUDA_PATH}/lib64 -lcurand -lcuda -lcudart -L${CUDA_PATH}/bin") 
endif()
message("***Current C++ Compiling Config are '${CMAKE_CXX_FLAGS}'\n")

set(DARKNET_ROOT_DIR ${ROOTDIR}/frameworks/darknet)
set(DARKNET_BUILD_DIR ${DARKNET_ROOT_DIR})
set(CAFFE_ROOT_DIR ${ROOTDIR}/frameworks/caffe)
set(CAFFE_BUILD_DIR ${CAFFE_ROOT_DIR}/build)
set(CAFFE2_ROOT_DIR ${ROOTDIR}/frameworks/caffe2)
set(CAFFE2_BUILD_DIR ${CAFFE2_ROOT_DIR}/build)
set(CAFFE2_INSTALL_DIR ${CAFFE2_ROOT_DIR}/install)
set(TENSORFLOW_ROOT_DIR ${ROOTDIR}/frameworks/tensorflow)
set(TENSORFLOW_BUILD_DIR ${TENSORFLOW_ROOT_DIR}/build)
set(BACKEND_CXXFLAGS ${CMAKE_CXX_FLAGS})
set(BACKEND_LD_FLAGS "-L${ROOTDIR}")#

if(USE_CAFFE2)
    #list(APPEND BACKEND_DEPS caffe2)
    #set(BACKEND_CXXFLAGS "${BACKEND_CXXFLAGS} -DUSE_CAFFE2 -I${CAFFE2_INSTALL_DIR}/include")
    #set(BACKEND_LD_FLAGS "${BACKEND_LD_FLAGS} -L${CAFFE2_INSTALL_DIR}/lib -lcaffe2 -lcaffe2_gpu -Wl,-rpath,${CAFFE2_INSTALL_DIR}/lib")
    option(USE_CAFFE, OFF)
endif()

if(USE_CAFFE)
		#set(BACKEND_CXXFLAGS "${BACKEND_CXXFLAGS} -DUSE_CAFFE -I${CAFFE_ROOT_DIR}/include -I${CAFFE_BUILD_DIR}/src")
		#set(BACKEND_LD_FLAGS "${BACKEND_LD_FLAGS} -L${CAFFE_BUILD_DIR}/lib -lcaffe -Wl,-rpath,${CAFFE_BUILD_DIR}/lib")
endif()

if(USE_DARKNRT)
		#set(BACKEND_CXXFLAGS "${BACKEND_CXXFLAGS} -DUSE_DARKNET -I${DARKNET_ROOT_DIR}/src -I${DARKNET_ROOT_DIR}/include")
		#set(BACK_LD_FLAGS "${BACKEND_LD_FLAGS} -L${DARKNET_BUILD_DIR} -ldarknet -Wl,-rpath,${DARKNET_BUILD_DIR}")
endif()

if(USE_TENSORFLOW)
		#shell: "export PKG_CONFIG_PATH=/directory/contains/'tensorflow.pc'/"
		#shell: "pkg-config --cflags tensorflow"
		#set(CXXFLAGS_TENSORFLOW "-DUSE_TENSORFLOW -I${TENSORFLOW_BUILD_DIR}/include/tensorflow -I${TENSORFLOW_BUILD_DIR}/include/tensorflow/tensorflow/contrib/makefile/downloads -I${TENSORFLOW_BUILD_DIR}/include/tensorflow/tensorflow/contrib/makefile/downloads/eigen -I${TENSORFLOW_BUILD_DIR}/include/tensorflow/tensorflow/contrib/makefile/downloads/gemmlowp -I${TENSORFLOW_BUILD_DIR}/include/tensorflow/tensorflow/contrib/makefile/downloads/nsync/public")
		#shell: "pkg-config --libs tensorflow"
		#set(LD_FLAGS_TENSORFLOW "-L${TENSORFLOW_BUILD_DIR}/lib/tensorflow -rdynamic -ltensorflow_cc -ldl -lpthread -Wl,-rpath,${TENSORFLOW_BUILD_DIR}/lib/tensorflow")
		#set(BACKEND_CXXFLAGS "${BACKEND_CXXFLAGS} ${CXXFLAGS_TENSORFLOW}")
		#set(BACK_LD_FLAGS "${BACKEND_LD_FLAGS} ${LD_FLAGS_TENSORFLOW}")
endif()
set(CMAKE_CXX_FLAGS ${BACKEND_CXXFLAGS})

add_subdirectory(frameworks) 
add_subdirectory(src/nexus)
add_subdirectory(tools/profiler)
add_subdirectory(tests)

