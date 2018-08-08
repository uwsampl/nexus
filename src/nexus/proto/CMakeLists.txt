list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_program(GRPC_CPP_PLUGIN grpc_cpp_plugin)
find_library(GRPC_LIBRARY NAMES grpc)
find_library(GRPCPP_LIBRARY NAMES grpc++)
find_library(GPR_LIBRARY NAMES gpr)
set(GRPC_LIBRARIES ${GRPCPP_LIBRARY} ${GRPC_LIBRARY} ${GPR_LIBRARY})
if(GRPC_LIBRARIES)
    message(STATUS "Found GRPC: ${GRPC_LIBRARIES}; plugin - ${GRPC_CPP_PLUGIN}")
endif()

set(PROTOBUF_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/nnquery.proto ${CMAKE_CURRENT_SOURCE_DIR}/control.proto)
set(GRPC_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/control.proto)

find_package(Protobuf REQUIRED)
find_package(GRPC REQUIRED)

set(GEN_DIR ${PROJECT_BINARY_DIR}/bin/gen/nexus/proto)
file(MAKE_DIRECTORY ${GEN_DIR})
include_directories(${GEN_DIR})


include_directories(${Protobuf_INCLUDE_DIRS})

protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${GEN_DIR} ${PROTOBUF_SRCS})
grpc_generate_cpp(GRPC_SRCS GRPC_HDRS ${GEN_DIR} ${GRPC_SRCS})

set(CUR_PROTO_SRCS ${GEN_DIR}/control.grpc.pb.cc ${GEN_DIR}/control.pb.cc ${GEN_DIR}/nnquery.pb.cc)
message(${CUR_PROTO_SRCS})
add_library(proto ${CUR_PROTO_SRCS})
target_link_libraries(proto
  ${GRPC_LIBRARIES}
    ${Protobuf_LIBRARIES}
)
