#include <assert.h>
#include "darknet_server.h"
#include "network.h"

// Python wrapper helpers
float *np_to_floatptr(float *inarray, int n){
  return inarray;
}

void cudnn_convolutional_setup_fast(layer *l) {
#ifdef GPU
#ifdef CUDNN
  cudnnSetTensor4dDescriptor(l->srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->c, l->h, l->w);
  cudnnSetTensor4dDescriptor(l->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->out_c, l->out_h, l->out_w);
#endif
#endif
}

void set_batch_network_lightweight(network *net, int batch) {
  net->batch = batch;
  int i;
  for (i = 0; i < net->n; ++i) {
    layer *l = &net->layers[i];
    l->batch = batch;
#ifdef GPU
#ifdef CUDNN
    if (l->type == CONVOLUTIONAL) {
      cudnn_convolutional_setup_fast(l);
    } else if (l->type == DECONVOLUTIONAL) {
      cudnnSetTensor4dDescriptor(l->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l->out_c, l->out_h, l->out_w);
      cudnnSetTensor4dDescriptor(l->normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l->out_c, 1, 1); 
    }
#endif
#endif
  }
}

void reset_batch_network(network *net, int batch) {
  net->batch = batch;
  int i;
  for (i = 0; i < net->n; ++i) {
    net->layers[i].batch = batch;
  }
  resize_network(net, net->w, net->h);
}

void network_predict_gpu_nocopy(network *netp) {
  network net = *netp;
  net.truth = 0;
  net.train = 0;
  net.delta = 0;
  
  cuda_set_device(net.gpu_index);
  int i;
  for(i = 0; i < net.n; ++i){
    net.index = i;
    layer l = net.layers[i];
    l.forward_gpu(l, net);
    net.input_gpu = l.output_gpu;
    net.input = l.output;
    if(l.truth) {
      net.truth_gpu = l.output_gpu;
      net.truth = l.output;
    }
  }
}

void get_detection_output(layer l, float **batch_output, int dim1, int dim2) {
  assert(dim1 >= l.batch);
  assert(dim2 >= l.outputs);
  float *output = l.output;
  int i;
  for (i = 0; i < l.batch; ++i, output += l.outputs) {
    memcpy(batch_output[i], output, l.outputs * sizeof(float));
  }
}

void output_detection_results(float *predictions,
                              layer l,
                              int imagew,
                              int imageh,
                              int netw,
                              int neth,
                              float thresh,
                              float *probs,
                              int nprobs,
                              int *boxes,
                              int nboxes,
                              int only_objectness,
                              int *map,
                              float tree_thresh,
                              int relative,
                              float nms){
  memset(probs, 0, nprobs*sizeof(float));
  memset(boxes, 0, nboxes*sizeof(int));

  int nboxes_rel = l.w*l.h*l.n;
  assert(nboxes_rel == nboxes/4);
  box *boxes_rel = calloc(nboxes_rel, sizeof(box));
  float **pprobs = calloc(nboxes_rel, sizeof(float *));
  int j;
  for(j = 0; j < nboxes_rel; ++j)
    pprobs[j] = probs + j*(l.classes+1);

  l.output = predictions;
  l.batch = 1;
  get_region_boxes(l, imagew, imageh, netw, neth, thresh, pprobs, boxes_rel,
                   0, only_objectness, map, tree_thresh, relative);

  if (l.softmax_tree && nms) {
    do_nms_obj(boxes_rel, pprobs, nboxes_rel, l.classes, nms);
  } else if (nms) {
    do_nms_sort(boxes_rel, pprobs, nboxes_rel, l.classes, nms);
  }
  for(j = 0; j < nboxes_rel; ++j){
    int class = max_index(pprobs[j], l.classes);
    float prob = pprobs[j][class];
    if(prob > thresh){
      box b = boxes_rel[j];
      
      int left  = (b.x-b.w/2.)*imagew;
      int right = (b.x+b.w/2.)*imagew;
      int top   = (b.y-b.h/2.)*imageh;
      int bot   = (b.y+b.h/2.)*imageh;

      if(left < 0) left = 0;
      if(right > imagew-1) right = imagew-1;
      if(top < 0) top = 0;
      if(bot > imageh-1) bot = imageh-1;
      
      boxes[j*4] = left;
      boxes[j*4+1] = right;
      boxes[j*4+2] = top;
      boxes[j*4+3] = bot;
    }

  }

  free(pprobs);
  free(boxes_rel);
  return;
}

// Test this from python with:
// import numpy as np; import sys; sys.path.append('build/lib/python'); import darknet_server as ds; ps, bx = ds.bounder(30, 10)
void bounder(float *probs, int nprobs, float *boxes, int nboxes){
  float **pprobs = calloc(10, sizeof(float *));
  int j;
  for(j = 0; j < 10; ++j)
    pprobs[j] = probs + j*3;

  for(j = 0; j < 10; ++j){
    probs = pprobs[j];
    int i;
    for (i = 0; i < 3; i++) {
      probs[i] = 100*j + i;
    }
  }

  free(pprobs);
  return;
}
