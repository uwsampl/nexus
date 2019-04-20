# c++ configs
CXX ?= g++
CUDA ?= /usr/local/cuda-10.0
WARNING = -Wall -Wfatal-errors -Wno-unused -Wno-unused-result
NEXUS = ../..
# CXXFLAGS = -std=c++11 -O3 -fPIC $(WARNING) -I$(NEXUS)/src -I$(NEXUS)/build/gen/src -I$(CUDA)/include
CXXFLAGS = -std=c++11 -g -fPIC $(WARNING) -I$(NEXUS)/src -I$(NEXUS)/build/gen/src -I$(CUDA)/include
LD_FLAGS = -lm -pthread -lglog -lgflags -lboost_system -lboost_thread \
	-lboost_filesystem -lyaml-cpp `pkg-config --libs protobuf` \
	`pkg-config --libs grpc++ grpc` `pkg-config --libs opencv4` \
	-Wl,-rpath,$(CUDA)/lib64  -L$(CUDA)/lib64 -lcuda -lcudart \
	-Wl,-rpath,$(NEXUS)/build -L$(NEXUS)/build -lnexus

all: bin/traffic_complex

bin/%: obj/%.o
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $^ $(LD_FLAGS) -o $@

obj/%.o : src/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf bin obj

.phony: clean
