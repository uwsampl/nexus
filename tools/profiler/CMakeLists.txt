include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_BINARY_DIR}/bin/gen)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)   
add_executable(profiler profiler.cpp)
target_link_libraries(profiler backend_lib common ${LD_FLAGS} ${BACKEND_LD_FLAGS})
set_target_properties(profiler PROPERTIES OUTPUT_NAME "profiler")

