# c++ configs
CXX = g++
WARNING = -Wall -Wfatal-errors -Wno-unused -Wno-unused-result
CXXFLAGS = -std=c++11 -O3 -fPIC $(WARNING) -I../../src -I../../build/gen \
	-I/usr/local/cuda/include
LD_FLAGS = -lm -pthread -lglog -lgflags -lboost_system -lboost_thread \
	-lboost_filesystem -lyaml-cpp  `pkg-config --libs protobuf` \
	`pkg-config --libs grpc++ grpc` `pkg-config --libs opencv` \
	-L/usr/local/cuda/lib64 -lcuda -lcudart \
	-L../../build/lib -lnexus -Wl,-rpath,../../build/lib

all: bin/simple_app

bin/simple_app : obj/main.o
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $^ $(LD_FLAGS) -o $@

obj/%.o : src/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf bin obj

.phony: clean
