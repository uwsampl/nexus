include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_BINARY_DIR}/bin/gen)
include_directories(${PROJECT_SOURCE_DIR}/src)
set(APP_SRCS app_base.cpp frontend.cpp model_handler.cpp request_context.cpp rpc_service.cpp worker.cpp)
add_library(nexus SHARED ${APP_SRCS})
target_link_libraries(nexus proto common ${LD_FLAGS})
set_target_properties(nexus PROPERTIES OUTPUT_NAME "nexus")

