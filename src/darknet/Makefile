GPU=1
CUDNN=1
OPENCV=1
DEBUG=0

ARCH= -gencode arch=compute_30,code=sm_30 \
      -gencode arch=compute_35,code=sm_35 \
      -gencode arch=compute_37,code=sm_37 \
      -gencode arch=compute_50,code=[sm_50,compute_50] \
      -gencode arch=compute_52,code=[sm_52,compute_52]

# This is what I use, uncomment if you know your arch and want to specify
# ARCH=  -gencode arch=compute_52,code=compute_52
# VPATH=./src/
CC=gcc
NVCC=nvcc 
OPTS=-Ofast
LDFLAGS= -lm -pthread 
COMMON=
## change
CFLAGS= -Wall -Wfatal-errors -Wno-unused -Wno-unused-result -fPIC
##change
ifeq ($(DEBUG), 1) 
	OPTS=-O0 -g
endif

CFLAGS+=$(OPTS)
DLL_LINK_FLAGS = -shared

VPATH= ../../darknet/src
COMMON+= -I$(VPATH) -I../../darknet/include
#EXEC=darknet_server

## change ##############################
ifeq ($(ENABLE_CBLAS), 1)
	CFLAGS+=  -DENABLE_CBLAS
	LDFLAGS+=  -lopenblas
#Note openblas assumed to installed in /usr/lib and /usr/include, as done
#e.g. by apt-get
#ENABLE_CBLAS
endif 
########################################

ifeq ($(OPENCV), 1) 
	COMMON += -DOPENCV
	CFLAGS += -DOPENCV
# the following opencv files must match the python dll (cv2.so) in the nnxs
# virtual env
	LDFLAGS += `pkg-config --libs opencv`
	CFLAGS += `pkg-config --cflags opencv`
endif

ifeq ($(GPU), 1)
	COMMON += -DGPU -I/usr/local/cuda/include/ -I/usr/include
	CFLAGS += -DGPU
	LDFLAGS += -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand
endif

ifeq ($(CUDNN), 1) 
	COMMON+= -DCUDNN 
	CFLAGS+= -DCUDNN
	LDFLAGS+= -lcudnn
endif

SRC = $(filter-out %_python_wrap.c, $(wildcard *.c)) \
	gemm.c utils.c cuda.c deconvolutional_layer.c convolutional_layer.c list.c image.c activations.c im2col.c col2im.c blas.c crop_layer.c dropout_layer.c maxpool_layer.c softmax_layer.c data.c matrix.c network.c connected_layer.c cost_layer.c parser.c option_list.c detection_layer.c route_layer.c box.c normalization_layer.c avgpool_layer.c layer.c local_layer.c shortcut_layer.c activation_layer.c rnn_layer.c gru_layer.c crnn_layer.c demo.c batchnorm_layer.c region_layer.c reorg_layer.c tree.c lstm_layer.c
OBJS = $(patsubst %.c, obj/%.o, $(SRC))
CUSRC = $(wildcard *.cu) \
	convolutional_kernels.cu deconvolutional_kernels.cu activation_kernels.cu im2col_kernels.cu col2im_kernels.cu blas_kernels.cu crop_layer_kernels.cu dropout_layer_kernels.cu maxpool_layer_kernels.cu avgpool_layer_kernels.cu
CUOBJS = $(patsubst %.cu, obj/%.o, $(CUSRC))
DEPS = $(wildcard *.h) Makefile

ifeq ($(GPU), 1) 
	LDFLAGS+= -lstdc++
	OBJS += $(CUOBJS)
endif

#$(EXEC): $(OBJS) $(EXEC_DLL) $(EXEC_PY_DLL)
#	$(CC) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS)

all: lib/libdarknet.a lib/libdarknet.so

lib/libdarknet.a: $(OBJS)
	@mkdir -p $(@D)
	ar rcs $@ $^

lib/libdarknet.so: $(OBJS)
	@mkdir -p $(@D)
	$(CC) $(DLL_LINK_FLAGS) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS)

obj/%.o: %.c $(DEPS)
	@mkdir -p $(@D)
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@

obj/%.o: %.cu $(DEPS)
	@mkdir -p $(@D)
	$(NVCC) $(ARCH) $(COMMON) --compiler-options "$(CFLAGS)" -c $< -o $@

.PRECIOUS: %_python_wrap.c

.PHONY: clean

clean:
	rm -rf obj lib $(wildcard *_wrap.c)
