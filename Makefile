CONFIG_FILE := Makefile.config
ifeq ($(wildcard $(CONFIG_FILE)),)
	$(error $(CONFIG_FILE) not found. See $(CONFIG_FILE).example.)
endif
include $(CONFIG_FILE)

ROOTDIR = $(CURDIR)

# protobuf srcs and objs
PROTO_SRC_DIR = src/nexus/proto
PROTO_SRCS := $(PROTO_SRC_DIR)/nnquery.proto
PROTO_GEN_HEADERS := ${PROTO_SRCS:src/nexus/%.proto=build/gen/%.pb.h}
PROTO_GEN_CC := ${PROTO_SRCS:src/nexus/%.proto=build/gen/%.pb.cc}
PROTO_OBJS := ${PROTO_SRCS:src/nexus/%.proto=build/obj/%.pb.o}
# gen python code
PROTO_GEN_PY_DIR = python/nexus/proto
PROTO_GEN_PY := $(patsubst $(PROTO_SRC_DIR)/%.proto, $(PROTO_GEN_PY_DIR)/%_pb2.py, $(PROTO_SRCS))
# grpc protobuf
GRPC_PROTO_SRCS := $(PROTO_SRC_DIR)/control.proto
PROTO_GEN_HEADERS += ${GRPC_PROTO_SRCS:src/nexus/%.proto=build/gen/%.pb.h} \
	${GRPC_PROTO_SRCS:src/nexus/%.proto=build/gen/%.grpc.pb.h}
PROTO_GEN_CC += ${GRPC_PROTO_SRCS:src/nexus/%.proto=build/gen/%.pb.cc} \
	${GRPC_PROTO_SRCS:src/nexus/%.proto=build/gen/%.grpc.pb.cc}
PROTO_OBJS += ${GRPC_PROTO_SRCS:src/nexus/%.proto=build/obj/%.pb.o} \
	${GRPC_PROTO_SRCS:src/nexus/%.proto=build/obj/%.grpc.pb.o}

# protoc config
PROTOC = `which protoc`
GRPC_CPP_PLUGIN = grpc_cpp_plugin
GRPC_CPP_PLUGIN_PATH ?= `which $(GRPC_CPP_PLUGIN)`

# c++ srcs and objs
CXX_COMMON_SRCS := $(wildcard src/nexus/common/*.cpp)
CXX_APP_SRCS := $(wildcard src/nexus/app/*.cpp)
CXX_BACKEND_SRCS := $(wildcard src/nexus/backend/*.cpp)
CXX_BACKEND_LIB_SRCS := $(filter-out src/nexus/backend/backend_main.cpp, $(CXX_BACKEND_SRCS))
CXX_SCHEDULER_SRCS := $(wildcard src/nexus/scheduler/*.cpp)

CXX_COMMON_OBJS := $(patsubst src/nexus/%.cpp, build/obj/%.o, $(CXX_COMMON_SRCS)) $(PROTO_OBJS)
CXX_APP_OBJS := $(patsubst src/nexus/%.cpp, build/obj/%.o, $(CXX_APP_SRCS))
CXX_BACKEND_OBJS := $(patsubst src/nexus/%.cpp, build/obj/%.o, $(CXX_BACKEND_SRCS))
CXX_BACKEND_LIB_OBJS := $(patsubst src/nexus/%.cpp, build/obj/%.o, $(CXX_BACKEND_LIB_SRCS))
CXX_SCHEDULER_OBJS := $(patsubst src/nexus/%.cpp, build/obj/%.o, $(CXX_SCHEDULER_SRCS))

OBJS := $(CXX_COMMON_OBJS) $(CXX_APP_OBJS) $(CXX_BACKEND_OBJS) $(CXX_SCHEDULER_OBJS)
DEPS := ${OBJS:.o=.d}

# c++ configs
CXX = g++
WARNING = -Wall -Wfatal-errors -Wno-unused -Wno-unused-result
CXXFLAGS = -std=c++11 -O3 -fPIC $(WARNING) -Isrc/nexus -Ibuild/gen
# Automatic dependency generation
CXXFLAGS += -MMD -MP
LD_FLAGS = -lm -pthread -lglog -lgflags -lboost_system -lboost_thread \
	-lboost_filesystem -lyaml-cpp  `pkg-config --libs protobuf` \
	`pkg-config --libs grpc++ grpc` `pkg-config --libs opencv`
DLL_LINK_FLAGS = -shared
ifeq ($(USE_GPU), 1)
	CXXFLAGS += -I$(CUDA_PATH)/include
	LD_FLAGS += -L$(CUDA_PATH)/lib64 -lcuda -lcudart
endif

# library dependency
DARKNET_BUILD_DIR = $(ROOTDIR)/src/darknet
CAFFE_BUILD_DIR = $(ROOTDIR)/caffe/build
TENSORFLOW_BUILD_DIR = $(ROOTDIR)/tensorflow/build

BACKEND_DEPS =
BACKEND_CXXFLAGS = 
BACKEND_LD_FLAGS = 
ifeq ($(USE_CAFFE), 1)
	BACKEND_DEPS += $(CAFFE_BUILD_DIR)/lib/libcaffe.so
	BACKEND_CXXFLAGS += -I$(ROOTDIR)/caffe/include -I$(CAFFE_BUILD_DIR)/src
	BACKEND_LD_FLAGS += -L$(CAFFE_BUILD_DIR)/lib -lcaffe -Wl,-rpath,$(CAFFE_BUILD_DIR)/lib
endif
ifeq ($(USE_DARKNET), 1)
	BACKEND_DEPS += $(DARKNET_BUILD_DIR)/lib/libdarknet.so
	BACKEND_LD_FLAGS += -L$(DARKNET_BUILD_DIR)/lib -ldarknet -Wl,-rpath,$(DARKNET_BUILD_DIR)/lib
endif
ifeq ($(USE_TENSORFLOW), 1)
	export PKG_CONFIG_PATH:=$(TENSORFLOW_BUILD_DIR)/lib/pkgconfig:${PKG_CONFIG_PATH}
	BACKEND_DEPS += $(TENSORFLOW_BUILD_DIR)/lib/tensorflow/libtensorflow_cc.so
	BACKEND_CXXFLAGS += `pkg-config --cflags tensorflow`
	BACKEND_LD_FLAGS += `pkg-config --libs tensorflow`
endif

all: proto python lib backend scheduler tool

$(CAFFE_BUILD_DIR)/lib/libcaffe.so:
	cd caffe; $(MAKE) proto && $(MAKE) all && $(MAKE) pycaffe; cd -

$(DARKNET_BUILD_DIR)/lib/libdarknet.so:
	cd src/darknet; $(MAKE) all; cd -

$(TENSORFLOW_BUILD_DIR)/lib/tensorflow/libtensorflow_cc.so:
	mkdir -p $(TENSORFLOW_BUILD_DIR)
	cd $(TENSORFLOW_BUILD_DIR); cmake -DCMAKE_INSTALL_PREFIX=. .. && make && make install

proto: $(PROTO_GEN_CC)

python: $(PROTO_GEN_PY)

lib: build/lib/libnexus.so

backend: build/bin/backend

scheduler: build/bin/scheduler

tool: build/bin/profiler

build/lib/libnexus.so: $(CXX_COMMON_OBJS) $(CXX_APP_OBJS)
	@mkdir -p $(@D)
	$(CXX) $(DLL_LINK_FLAGS) $^ -o $@ $(LD_FLAGS)

build/bin/backend: $(CXX_COMMON_OBJS) $(CXX_BACKEND_OBJS)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $^ $(LD_FLAGS) $(BACKEND_LD_FLAGS) -o $@

build/bin/scheduler: $(CXX_COMMON_OBJS) $(CXX_SCHEDULER_OBJS)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $^ $(LD_FLAGS) -o $@

build/bin/profiler: $(CXX_COMMON_OBJS) $(CXX_BACKEND_LIB_OBJS) build/obj/tools/profiler/profiler.o
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $^ $(LD_FLAGS) $(BACKEND_LD_FLAGS) -o $@

build/gen/%.pb.cc build/gen/%.pb.h: src/nexus/%.proto
	@mkdir -p $(@D)
	$(PROTOC) -I$(PROTO_SRC_DIR) --cpp_out=$(@D) $<

build/gen/%.grpc.pb.cc build/gen/%.grpc.pb.h: src/nexus/%.proto | build/gen/%.pb.h
	@mkdir -p $(@D)
	$(PROTOC) -I$(PROTO_SRC_DIR) --grpc_out=$(@D) --plugin=protoc-gen-grpc=$(GRPC_CPP_PLUGIN_PATH) $<

$(PROTO_GEN_PY_DIR)/%_pb2.py: $(PROTO_SRC_DIR)/%.proto
	@mkdir -p $(@D)
	touch $(@D)/__init__.py
	$(PROTOC) --proto_path=$(PROTO_SRC_DIR) --python_out=$(@D) $<

build/obj/%.pb.o: build/gen/%.pb.cc build/gen/%.pb.h
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@

build/obj/backend/%.o: src/nexus/backend/%.cpp | $(PROTO_GEN_HEADERS) $(BACKEND_DEPS)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(BACKEND_CXXFLAGS) -c $< -o $@

build/obj/%.o: src/nexus/%.cpp | $(PROTO_GEN_HEADERS)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@

build/obj/tools/%.o: tools/%.cpp | $(PROTO_GEN_HEADERS)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(BACKEND_CXXFLAGS) -c $< -o $@

.PRECIOUS: %.pb.cc %.pb.h %.grpc.pb.cc %.grpc.pb.h $(PROTO_GEN_HEADERS) $(PROTO_GEN_CC)

.PHONY: proto python lib backend scheduler tool \
	clean clean-darknet clean-caffe clean-tensorflow cleanall

clean:
	rm -rf build $(PROTO_GEN_PY_DIR) $(PROTO_GEN_CC) $(PROTO_GEN_HEADERS)

clean-darknet:
	cd src/darknet; $(MAKE) clean; cd -

clean-caffe:
	cd caffe; $(MAKE) clean; cd -

clean-tensorflow:
	rm -rf $(TENSORFLOW_BUILD_DIR) tensorflow/.tf_configure.bazelrc

cleanall: clean clean-darknet clean-caffe clean-tensorflow

-include $(DEPS)
