#ifndef NEXUS_BACKEND_CAFFE_MODEL_H_
#define NEXUS_BACKEND_CAFFE_MODEL_H_

#ifdef USE_CAFFE

#include <boost/shared_ptr.hpp>

#include "nexus/backend/model_ins.h"

// Caffe headers
// avoid redefined keywords from darknet
#ifdef GPU
#undef GPU
#endif
#ifdef CUDNN
#undef CUDNN
#endif
// flag to include OpenCV related functions in Caffe
#define USE_OPENCV
#include "caffe/caffe.hpp"
#include "caffe/data_transformer.hpp"

namespace nexus {
namespace backend {

class CaffeModel : public ModelInstance {
 public:
  CaffeModel(int gpu_id, const ModelInstanceConfig& config);

  Shape InputShape() final;

  std::unordered_map<std::string, Shape> OutputShapes() final;

  ArrayPtr CreateInputGpuArray() final;

  std::unordered_map<std::string, ArrayPtr> GetOutputGpuArrays() final;

  void Preprocess(std::shared_ptr<Task> task) final;

  void Forward(std::shared_ptr<BatchTask> batch_task) final;

  void Postprocess(std::shared_ptr<Task> task) final;

 private:
  void LoadClassnames(const std::string& filename);

  // Caffe neural network for serving
  std::unique_ptr<caffe::ServeNet<float> > net_;
  // image size
  int image_height_;
  int image_width_;
  // input shape of neural network
  Shape input_shape_;
  // output shape of neural network
  Shape output_shape_;
  // size of input in a single batch
  size_t input_size_;
  // size of output in a single batch
  size_t output_size_;
  int input_blob_idx_;
  std::string output_blob_name_;
  std::vector<std::string> classnames_;
  // transformer for input
  std::unique_ptr<caffe::DataTransformer<float> > transformer_;
  std::vector<boost::shared_ptr<caffe::Blob<float> > > input_blobs_;
  std::string prefix_layer_;
  int prefix_index_;
};

} // namespace backend
} // namespace nexus

#endif // USE_CAFFE

#endif // NEXUS_BACKEND_CAFFE_MODEL_H_
