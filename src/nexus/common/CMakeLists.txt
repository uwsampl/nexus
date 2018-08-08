include_directories(${PROJECT_BINARY_DIR}/bin/gen)
include_directories(${PROJECT_SOURCE_DIR}/src)
aux_source_directory(./ COMMON_SRCS)
add_library(common ${COMMON_SRCS})
target_link_libraries(common proto ${LD_FLAGS})
set_target_properties(common PROPERTIES OUTPUT_NAME "common")

