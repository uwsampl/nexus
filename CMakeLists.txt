cmake_minimum_required(VERSION 3.13)
project(nexus LANGUAGES CXX)
list(INSERT CMAKE_MODULE_PATH 0 ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

option(USE_GPU        "Use GPU"        ON )
option(USE_TENSORFLOW "Use TensorFlow" ON )
option(USE_DARKNET    "Use Darknet"    ON )
option(USE_CAFFE2     "Use Caffe2"     ON )
option(USE_CAFFE      "Use Caffe"      OFF)
if(USE_CAFFE2 AND USE_CAFFE)
    message(FATAL_ERROR "`USE_CAFFE2` and `USE_CAFFE` cannot be set at the same time.")
endif()
set(CUDA_PATH "/usr/local/cuda" CACHE STRING "Path to the Cuda toolkit")

find_package(Boost REQUIRED COMPONENTS system filesystem)
find_package(Protobuf REQUIRED)
find_package(gRPC REQUIRED)
find_package(glog REQUIRED)
find_package(gflags REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(OpenCV REQUIRED)
include(ProcessorCount)
ProcessorCount(NPROC)


###### protobuf & gRPC ######
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
set(GENERATED_PY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/python/nexus/proto)
add_custom_command(
    OUTPUT ${GENERATED_PY_DIR}/nnquery_pb2.py
    COMMAND mkdir -p ${GENERATED_PY_DIR}
    COMMAND touch ${GENERATED_PY_DIR}/__init__.py
    COMMAND protobuf::protoc
        --proto_path=${CMAKE_CURRENT_SOURCE_DIR}/src/nexus/proto
        --python_out=${GENERATED_PY_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/src/nexus/proto/nnquery.proto
    DEPENDS
        protobuf::protoc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/nexus/proto/nnquery.proto)
add_custom_target(python DEPENDS ${GENERATED_PY_DIR}/nnquery_pb2.py)



###### common source files, headers, and linking options ######
add_library(common OBJECT
        ${PROTO_NNQUERY_CC}
        ${PROTO_NNQUERY_H}
        ${PROTO_CONTROL_CC}
        ${PROTO_CONTROL_H}
        ${GRPC_CONTROL_CC}
        ${GRPC_CONTROL_H}
        src/nexus/common/backend_pool.cpp
        src/nexus/common/buffer.cpp
        src/nexus/common/connection.cpp
        src/nexus/common/data_type.cpp
        src/nexus/common/device.cpp
        src/nexus/common/image.cpp
        src/nexus/common/message.cpp
        src/nexus/common/metric.cpp
        src/nexus/common/model_db.cpp
        src/nexus/common/server_base.cpp
        src/nexus/common/time_util.cpp
        src/nexus/common/util.cpp)
target_include_directories(common PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${GENERATED_SRC_DIR}/src)
target_compile_features(common PUBLIC cxx_std_11)
target_link_libraries(common PUBLIC
        yaml-cpp gflags glog::glog gRPC::grpc++ protobuf::libprotobuf
        ${OpenCV_LIBS} Boost::filesystem Boost::system)
set_property(TARGET common PROPERTY POSITION_INDEPENDENT_CODE ON)



###### scheduler ######
add_executable(scheduler
        src/nexus/scheduler/backend_delegate.cpp
        src/nexus/scheduler/frontend_delegate.cpp
        src/nexus/scheduler/sch_info.cpp
        src/nexus/scheduler/scheduler.cpp
        src/nexus/scheduler/scheduler_main.cpp)
target_include_directories(scheduler PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${GENERATED_SRC_DIR}/src)
target_compile_features(scheduler PRIVATE cxx_std_11)
target_link_libraries(scheduler PRIVATE common)



###### frontend libnexus.so ######
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
target_link_libraries(nexus PUBLIC common)



###### backend ######
add_library(backend_obj
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
target_compile_features(backend_obj PUBLIC cxx_std_11)
target_link_libraries(backend_obj PUBLIC common)

add_executable(backend
        src/nexus/backend/backend_main.cpp)
target_compile_features(backend PRIVATE cxx_std_11)
target_link_libraries(backend PRIVATE backend_obj)

## backend: CUDA support ##
if(USE_GPU)
    target_compile_definitions(common PUBLIC USE_GPU)
    target_include_directories(common PUBLIC ${CUDA_PATH}/include)
    target_link_libraries(common PUBLIC
            ${CUDA_PATH}/lib64/stubs/libcuda.so
            ${CUDA_PATH}/lib64/libcudart.so
            ${CUDA_PATH}/lib64/libcurand.so)
endif()

## backend: TensorFlow support ##
if(USE_TENSORFLOW)
    target_compile_definitions(backend_obj PUBLIC USE_TENSORFLOW)
    set(TENSORFLOW_BUILD_ROOT ${CMAKE_CURRENT_BINARY_DIR}/tensorflow/execroot/org_tensorflow/bazel-out)
    if(CMAKE_BUILD_TYPE MATCHES DEBUG)
        set(TENSORFLOW_BUILD_ROOT ${TENSORFLOW_BUILD_ROOT}/k8-dbg)
        set(TENSORFLOW_BUILD_OPT --compilation_mode=dbg --strip=never --copt="-DNDEBUG" --copt="-march=x86-64")
    else()
        set(TENSORFLOW_BUILD_ROOT ${TENSORFLOW_BUILD_ROOT}/k8-opt)
        set(TENSORFLOW_BUILD_OPT --config=opt)
    endif()

    set(TENSORFLOW_SO
            ${TENSORFLOW_BUILD_ROOT}/bin/tensorflow/libtensorflow_cc.so
            ${TENSORFLOW_BUILD_ROOT}/bin/tensorflow/libtensorflow_framework.so)
    add_custom_command(
        OUTPUT ${TENSORFLOW_SO}
        WORKING_DIRECTORY
            ${CMAKE_CURRENT_SOURCE_DIR}/frameworks/tensorflow
        COMMAND
            cp  ${CMAKE_CURRENT_SOURCE_DIR}/frameworks/tf_configure.bazelrc
                ${CMAKE_CURRENT_SOURCE_DIR}/frameworks/tensorflow/.tf_configure.bazelrc
        COMMAND
            bazel --output_base=${CMAKE_CURRENT_BINARY_DIR}/tensorflow build ${TENSORFLOW_BUILD_OPT}
                --action_env CUDA_TOOLKIT_PATH=${CUDA_PATH}
                --action_env CUDNN_INSTALL_PATH=${CUDA_PATH}
                //tensorflow:libtensorflow.so
                //tensorflow:libtensorflow_cc.so
                //tensorflow:libtensorflow_framework.so
                //tensorflow:install_headers
        COMMAND
            bazel shutdown)
    add_custom_target(tensorflow DEPENDS ${TENSORFLOW_SO})
    target_include_directories(backend_obj PUBLIC ${TENSORFLOW_BUILD_ROOT}/genfiles/tensorflow/include)
    target_include_directories(backend_obj PUBLIC ${TENSORFLOW_BUILD_ROOT}/genfiles/tensorflow/include/external/com_google_absl)
    target_link_libraries(backend_obj PUBLIC ${TENSORFLOW_SO})
    add_dependencies(backend_obj tensorflow)
endif()

## backend: Caffe2 support ##
if(USE_CAFFE2)
    target_compile_definitions(backend PRIVATE USE_CAFFE2)
    set(CAFFE2_BUILD_ROOT ${CMAKE_CURRENT_BINARY_DIR}/caffe2)
    set(CAFFE2_SO
            ${CAFFE2_BUILD_ROOT}/install/lib/libcaffe2.so
            ${CAFFE2_BUILD_ROOT}/install/lib/libcaffe2_gpu.so)
    file(MAKE_DIRECTORY ${CAFFE2_BUILD_ROOT})
    add_custom_command(
        OUTPUT ${CAFFE2_SO}
        WORKING_DIRECTORY ${CAFFE2_BUILD_ROOT}
        COMMAND
            patch -p1 -R --dry-run -d ${CMAKE_CURRENT_SOURCE_DIR}/frameworks/caffe2 -i ${CMAKE_CURRENT_SOURCE_DIR}/frameworks/caffe2-patch.diff
            || patch -p1 -d ${CMAKE_CURRENT_SOURCE_DIR}/frameworks/caffe2 -i ${CMAKE_CURRENT_SOURCE_DIR}/frameworks/caffe2-patch.diff
        COMMAND ${CMAKE_COMMAND} -DUSE_NNPACK=OFF -DUSE_NCCL=OFF -DUSE_MPI=OFF -DUSE_LMDB=OFF -DUSE_LEVELDB=OFF
            -DUSE_NUMA=OFF -DUSE_MPI=OFF -DBUILD_PYTHON=OFF -DBUILD_BINARY=OFF -DBUILD_TEST=OFF
            -DCUDA_TOOLKIT_ROOT_DIR=${CUDA_PATH}
            -DCMAKE_INSTALL_PREFIX=${CAFFE2_BUILD_ROOT}/install
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DCMAKE_DEBUG_POSTFIX=""
            ${CMAKE_CURRENT_SOURCE_DIR}/frameworks/caffe2
        COMMAND make -j${NPROC} caffe2_gpu
        COMMAND make -j${NPROC} install
    )
    add_custom_target(caffe2 DEPENDS ${CAFFE2_SO})
    target_include_directories(backend_obj PUBLIC ${CAFFE2_BUILD_ROOT}/install/include)
    target_link_libraries(backend_obj PUBLIC ${CAFFE2_SO})
    add_dependencies(backend_obj caffe2)
endif()

## backend: Caffe support ##
if(USE_CAFFE)
    target_compile_definitions(backend PRIVATE USE_CAFFE)
    set(CAFFE_SRC_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/frameworks/caffe)
    set(CAFFE_BUILD_ROOT ${CAFFE_SRC_ROOT}/build)
    set(CAFFE_SO ${CAFFE_BUILD_ROOT}/lib/libcaffe.so)
    add_custom_command(
        OUTPUT ${CAFFE_SO}
        WORKING_DIRECTORY ${CAFFE_SRC_ROOT}
        COMMAND make -j${NPROC} all)
    add_custom_target(caffe DEPENDS ${CAFFE_SO})
    target_include_directories(backend_obj PUBLIC ${CAFFE_SRC_ROOT}/src)
    target_include_directories(backend_obj PUBLIC ${CAFFE_SRC_ROOT}/include)
    target_link_libraries(backend_obj PUBLIC ${CAFFE_SO})
    add_dependencies(backend_obj caffe)
endif()

## backend: Darknet support ##
if(USE_DARKNET)
    target_compile_definitions(backend PRIVATE USE_DARKNET)
    set(DARKNET_SRC_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/frameworks/darknet)
    set(DARKNET_SO ${DARKNET_SRC_ROOT}/libdarknet.so)
    add_custom_command(
        OUTPUT ${DARKNET_SO}
        WORKING_DIRECTORY ${DARKNET_SRC_ROOT}
        COMMAND make -j${NPROC} all)
    add_custom_target(darknet DEPENDS ${DARKNET_SO})
    target_include_directories(backend_obj PUBLIC ${DARKNET_SRC_ROOT}/src)
    target_include_directories(backend_obj PUBLIC ${DARKNET_SRC_ROOT}/include)
    target_link_libraries(backend_obj PUBLIC ${DARKNET_SO})
    add_dependencies(backend_obj darknet)
endif()



###### tools/profiler ######
add_executable(profiler tools/profiler/profiler.cpp)
target_compile_features(profiler PRIVATE cxx_std_11)
target_link_libraries(profiler PRIVATE common backend_obj)



###### tools/test_pb ######
add_executable(test_pb tools/profiler/test_pb.cpp)
target_compile_features(test_pb PRIVATE cxx_std_11)
target_link_libraries(test_pb PRIVATE common backend_obj)



# FIXME ###### tests ######
# add_executable(runtest
#         tests/cpp/scheduler/backend_delegate_test.cpp
#         tests/cpp/scheduler/scheduler_test.cpp
#         tests/cpp/test_main.cpp)
# target_compile_features(runtest PRIVATE cxx_std_11)
# target_link_libraries(runtest PRIVATE common backend_obj)
