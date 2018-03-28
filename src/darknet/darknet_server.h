#ifndef DARKSERVER_H
#define DARKSERVER_H

#ifndef GPU
#define GPU
#endif

#ifndef CUDNN
#define CUDNN
#endif

#ifndef OPENCV
#define OPENCV
#endif

#include "../../darknet/include/darknet.h"


// Python wrapper helpers
float *np_to_floatptr(float *inarray, int n);
void set_batch_network_lightweight(network *net, int batch);
void reset_batch_network(network *net, int batch);
void network_predict_gpu_nocopy(network *netp);
void get_detection_output(layer l, float **batch_output, int dim1, int dim2);
void bounder(float *probs, int nprobs, float *boxes, int nboxes);
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
                              float nms);
#endif
