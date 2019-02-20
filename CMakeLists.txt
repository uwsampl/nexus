cmake_minimum_required(VERSION 3.13)
project(nexus)
list(INSERT CMAKE_MODULE_PATH 0 ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

option(USE_GPU        "Use GPU"        ON )
option(USE_TENSORFLOW "Use TensorFlow" ON )
option(USE_DARKNET    "Use Darknet"    ON )
option(USE_CAFFE2     "Use Caffe2"     ON )
option(USE_CAFFE      "Use Caffe"      OFF)
if(USE_CAFFE2 AND USE_CAFFE)
    message(FATAL_ERROR "`USE_CAFFE2` and `USE_CAFFE` cannot be set at the same time.")
endif()
option(CUDA_PATH  "Path to the Cuda toolkit"  "/usr/local/cuda")
option(CUDNN_PATH "Path to the CuDNN library" "/usr/local/cuda")

set(GFLAGS_USE_TARGET_NAMESPACE ON)
find_package(Boost REQUIRED COMPONENTS system filesystem)
find_package(Protobuf REQUIRED)
find_package(gRPC REQUIRED)
find_package(glog REQUIRED)
find_package(gflags REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(OpenCV REQUIRED)

set(GENERATED_SRC_DIR ${CMAKE_CURRENT_BINARY_DIR}/gen)
file(MAKE_DIRECTORY ${GENERATED_SRC_DIR}/src/nexus/proto)
protobuf_generate_cpp(PROTO_NNQUERY_CC PROTO_NNQUERY_H
        ${GENERATED_SRC_DIR}/src/nexus/proto
        src/nexus/proto/nnquery.proto)
protobuf_generate_cpp(PROTO_CONTROL_CC PROTO_CONTROL_H
        ${GENERATED_SRC_DIR}/src/nexus/proto
        src/nexus/proto/control.proto)
grpc_generate_cpp(GRPC_CONTROL_CC GRPC_CONTROL_H
        ${GENERATED_SRC_DIR}/src/nexus/proto
        src/nexus/proto/control.proto)
# TODO: nnquery.proto for Python
# TODO: gtest



add_library(common OBJECT
        ${PROTO_NNQUERY_CC}
        ${PROTO_NNQUERY_H}
        ${PROTO_CONTROL_CC}
        ${PROTO_CONTROL_H}
        ${GRPC_CONTROL_CC}
        ${GRPC_CONTROL_H}
        src/nexus/common/backend_pool.cpp
        src/nexus/common/connection.cpp
        src/nexus/common/data_type.cpp
        src/nexus/common/image.cpp
        src/nexus/common/message.cpp
        src/nexus/common/metric.cpp
        src/nexus/common/model_db.cpp
        src/nexus/common/server_base.cpp
        src/nexus/common/time_util.cpp
        src/nexus/common/util.cpp)
target_include_directories(common PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${GENERATED_SRC_DIR}/src)
target_compile_features(common PRIVATE cxx_std_11)
target_link_libraries(common PUBLIC ${OpenCV_LIBS})
set_property(TARGET common PROPERTY POSITION_INDEPENDENT_CODE ON)



add_executable(scheduler
        src/nexus/scheduler/backend_delegate.cpp
        src/nexus/scheduler/frontend_delegate.cpp
        src/nexus/scheduler/scheduler.cpp
        src/nexus/scheduler/scheduler_main.cpp)
target_include_directories(scheduler PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${GENERATED_SRC_DIR}/src)
target_compile_features(scheduler PRIVATE cxx_std_11)
target_link_libraries(scheduler PRIVATE common
        yaml-cpp gflags::gflags glog::glog gRPC::grpc++ protobuf::libprotobuf
        Boost::filesystem Boost::system)



add_library(nexus SHARED
        src/nexus/app/app_base.cpp
        src/nexus/app/frontend.cpp
        src/nexus/app/model_handler.cpp
        src/nexus/app/request_context.cpp
        src/nexus/app/rpc_service.cpp
        src/nexus/app/worker.cpp)
target_include_directories(nexus PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${GENERATED_SRC_DIR}/src)
target_compile_features(nexus PRIVATE cxx_std_11)
target_link_libraries(nexus PRIVATE common gflags::gflags glog::glog)



add_executable(backend
        src/nexus/backend/backend_main.cpp
        src/nexus/backend/backend_server.cpp
        src/nexus/backend/backup_client.cpp
        src/nexus/backend/batch_task.cpp
        src/nexus/backend/caffe2_model.cpp
        src/nexus/backend/caffe_densecap_model.cpp
        src/nexus/backend/caffe_model.cpp
        src/nexus/backend/darknet_model.cpp
        src/nexus/backend/gpu_executor.cpp
        src/nexus/backend/model_exec.cpp
        src/nexus/backend/model_ins.cpp
        src/nexus/backend/rpc_service.cpp
        src/nexus/backend/share_prefix_model.cpp
        src/nexus/backend/slice.cpp
        src/nexus/backend/task.cpp
        src/nexus/backend/tensorflow_model.cpp
        src/nexus/backend/utils.cpp
        src/nexus/backend/worker.cpp)
target_include_directories(backend PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${GENERATED_SRC_DIR}/src)
target_compile_features(backend PRIVATE cxx_std_11)
target_link_libraries(backend PRIVATE common
        yaml-cpp gflags::gflags glog::glog gRPC::grpc++ protobuf::libprotobuf
        Boost::filesystem Boost::system)


if(USE_TENSORFLOW)
    set(TENSORFLOW_BUILD_ROOT ${CMAKE_CURRENT_BINARY_DIR}/tensorflow/execroot/org_tensorflow/bazel-out/k8-opt)
    set(TENSORFLOW_SO
            ${TENSORFLOW_BUILD_ROOT}/bin/tensorflow/libtensorflow.so
            ${TENSORFLOW_BUILD_ROOT}/bin/tensorflow/libtensorflow_cc.so
            ${TENSORFLOW_BUILD_ROOT}/bin/tensorflow/libtensorflow_framework.so)
    add_custom_command(
        OUTPUT ${TENSORFLOW_SO}
        WORKING_DIRECTORY
            ${CMAKE_CURRENT_SOURCE_DIR}/frameworks/tensorflow
        COMMAND
            cp ${CMAKE_CURRENT_SOURCE_DIR}/frameworks/tf_configure.bazelrc ${CMAKE_CURRENT_SOURCE_DIR}/frameworks/tensorflow/.tf_configure.bazelrc
        COMMAND
            bazel --output_base=${CMAKE_CURRENT_BINARY_DIR}/tensorflow build --config=opt //tensorflow:libtensorflow.so //tensorflow:libtensorflow_cc.so //tensorflow:libtensorflow_framework.so //tensorflow:install_headers)
    add_custom_target(tensorflow DEPENDS ${TENSORFLOW_SO})
    target_include_directories(backend PRIVATE ${TENSORFLOW_BUILD_ROOT}/include)
    target_link_libraries(backend PRIVATE ${TENSORFLOW_SO})
    add_dependencies(backend tensorflow)
endif()
