add_library(frameworks SHARED ./extra.cpp)

if(USE_CAFFE)
	execute_process(COMMAND make proto
									WORKING_DIRECTORY "${CAFFE_ROOT_DIR}")
	execute_process(COMMAND make all
									WORKING_DIRECTORY "${CAFFE_ROOT_DIR}")
  execute_process(COMMAND make pycaffe
									WORKING_DIRECTORY "${CAFFE_ROOT_DIR}")
									
  add_library(caffe SHARED IMPORTED)
  set_property(TARGET caffe PROPERTY IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/caffe/build/lib/libcaffe.so")
  target_link_libraries(frameworks caffe)
endif() 

if(USE_CAFFE2)
	execute_process(COMMAND mkdir -p ${CAFFE2_BUILD_DIR}
									WORKING_DIRECTORY ${CAFFE2_ROOT_DIR})
	execute_process(COMMAND cmake .. -DUSE_NNPACK=OFF -DUSE_NCCL=OFF -DCMAKE_INSTALL_PREFIX=../install
									WORKING_DIRECTORY ${CAFFE2_BUILD_DIR})
	execute_process(COMMAND make
                  WORKING_DIRECTORY ${CAFFE2_BUILD_DIR})
  execute_process(COMMAND make install
                  WORKING_DIRECTORY ${CAFFE2_BUILD_DIR})						  

  #add_library(caffe2 SHARED IMPORTED)
  add_library(caffe2 SHARED IMPORTED)
  #set_property(TARGET caffe2 PROPERTY IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/caffe2/build/lib/libcaffe2_gpu.so")
  set_property(TARGET caffe2 PROPERTY IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/caffe2/install/lib/libcaffe2_gpu.so")
  target_link_libraries(frameworks caffe2)
endif()

if(USE_DARKNET)
	execute_process(COMMAND make all
									WORKING_DIRECTORY "${DARKNET_ROOT_DIR}"
									)	
  add_library(darknet SHARED IMPORTED)
  set_property(TARGET darknet PROPERTY IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/darknet/libdarknet.so")
  target_link_libraries(frameworks darknet)   
endif()

if(USE_TENSORFLOW)
	execute_process(COMMAND mkdir -p ${TENSORFLOW_BUILD_DIR}
									WORKING_DIRECTORY "${TENSORFLOW_ROOT_DIR}")
	execute_process(COMMAND cmake -DCMAKE_INSTALL_PREFIX=. ..
									WORKING_DIRECTORY "${TENSORFLOW_BUILD_DIR}")
	execute_process(COMMAND make
									WORKING_DIRECTORY "${TENSORFLOW_BUILD_DIR}")
  execute_process(COMMAND make install
									WORKING_DIRECTORY "${TENSORFLOW_BUILD_DIR}")
							                            
  add_library(tensorflow SHARED IMPORTED)
  set_property(TARGET tensorflow PROPERTY IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/tensorflow/build/lib/tensorflow/libtensorflow_cc.so")
  target_link_libraries(frameworks tensorflow)
endif()

set_target_properties(frameworks PROPERTIES OUTPUT_NAME "frameworks")
