#ifndef NEXUS_BACKEND_CAFFE_MODEL_H_
#define NEXUS_BACKEND_CAFFE_MODEL_H_

#if USE_CAFFE == 1

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
  CaffeModel(int gpu_id, const std::string& model_name, uint32_t version,
             const std::string& type, uint32_t batch, uint32_t max_batch,
             BlockPriorityQueue<Task>& task_queue, const YAML::Node& info);

  ~CaffeModel() {}

  std::string framework() const final { return "caffe"; }

  std::string profile_id() const final;

 private:
  void InitBatchInputArray() final;

  void PreprocessImpl(std::shared_ptr<Task> task,
                      std::vector<ArrayPtr>* input_arrays) final;

  void ForwardImpl(BatchInput* batch_input, BatchOutput* batch_output) final;

  void PostprocessImpl(std::shared_ptr<Task> task, Output* output) final;

  void LoadClassnames(const std::string& filename);

  void MarshalClassificationResult(
      const QueryProto& query, const float* probs, size_t nprobs,
      float threshold, QueryResultProto* result);

 private:
  // Caffe neural network for serving
  std::unique_ptr<caffe::ServeNet<float> > net_;
  // input shape of neural network
  std::vector<int> input_shape_;
  // output shape of neural network
  std::vector<int> output_shape_;
  // size of input in a single batch
  size_t input_size_;
  // size of output in a single batch
  size_t output_size_;
  // resized image dim
  int image_dim_;
  std::vector<std::string> classnames_;
  // transformer for input
  std::unique_ptr<caffe::DataTransformer<float> > transformer_;
  caffe::Blob<float>* input_blob_;
  std::vector<int> input_blob_indices_;
  std::string prefix_layer_;
  int prefix_index_;
};

} // namespace backend
} // namespace nexus

#endif // USE_CAFFE == 1

#endif // NEXUS_BACKEND_CAFFE_MODEL_H_
