#ifndef NEXUS_BACKEND_CAFFE_DENSECAP_MODEL_H_
#define NEXUS_BACKEND_CAFFE_DENSECAP_MODEL_H_

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
#include "caffe/caffe.hpp"

namespace nexus {
namespace backend {

class CaffeDenseCapModel : public ModelInstance {
 public:
  CaffeDenseCapModel(int gpu_id, const ModelInstanceConfig& config,
                     const YAML::Node& info);

  ArrayPtr CreateInputGpuArray() final;

  std::unordered_map<std::string, size_t> OutputSizes() const final;

  void Preprocess(std::shared_ptr<Task> task) final;

  void Forward(BatchInput* batch_input, BatchOutput* batch_output) final;

  void Postprocess(std::shared_ptr<Task> task) final;

 private:
  void LoadVocabulary(const std::string& filename);

  void TransformBbox(int im_height, int im_width, float scale, int nboxes,
                     const float* rois, const float* bbox_deltas, float* out);

  // parameters
  int max_timestep_;
  int max_boxes_;
  float nms_threshold_;
  float score_threshold_;
  std::vector<float> mean_values_;
  std::vector<float> bbox_mean_;
  std::vector<float> bbox_stds_;
  // networks and data
  std::unique_ptr<caffe::ServeNet<float> > feature_net_;
  std::unique_ptr<caffe::ServeNet<float> > rnn_net_;
  std::unique_ptr<caffe::ServeNet<float> > embed_net_;
  std::vector<std::string> vocabulary_;
  // shapes and sizes of input and output
  int image_height_;
  int image_width_;
  size_t input_size_;
  std::vector<int> input_shape_;
  std::unordered_map<std::string, size_t> output_sizes_;
  //caffe::Blob<float>* input_blob_;
  int feature_net_input_idx_;
  std::vector<boost::shared_ptr<caffe::Blob<float> > > input_blobs_;
  // temporary buffer
  std::vector<float> best_words_;
  std::unique_ptr<caffe::Blob<float> > multiplier_;
};

} // namespace backend
} // namespace nexus

#endif // USE_CAFFE

#endif // NEXUS_BACKEND_CAFFE_DENSECAP_MODEL_H_
