#ifndef NEXUS_BACKEND_CAFFE_DENSECAP_MODEL_H_
#define NEXUS_BACKEND_CAFFE_DENSECAP_MODEL_H_

#include "model_ins.h"

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
  CaffeDenseCapModel(int gpu_id, std::string model_id, std::string model_name,
                     ModelType type, uint32_t batch, uint32_t max_batch,
                     BlockPriorityQueue<Task>& task_queue,
                     const YAML::Node& info);

  ~CaffeDenseCapModel() {}

  Framework framework() const final { return CAFFE; }

 private:
  void InitBatchInputArray() final;

  //void UpdateMaxBatchImpl() final;

  void PreprocessImpl(std::shared_ptr<Task> task,
                      std::vector<ArrayPtr>* input_arrays) final;

  void ForwardImpl(BatchInput* batch_input, BatchOutput* batch_output) final;

  void PostprocessImpl(std::shared_ptr<Task> task, Output* output) final;

  void LoadVocabulary(const std::string& filename);

  void TransformBbox(int im_height, int im_width, float scale, int nboxes,
                     const float* rois, const float* bbox_deltas, float* out);

 private:
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
  std::vector<size_t> output_sizes_;
  caffe::Blob<float>* input_blob_;
  // temporary buffer
  std::vector<float> best_words_;
  std::unique_ptr<caffe::Blob<float> > multiplier_;
};

} // namespace backend
} // namespace nexus

#endif // NEXUS_BACKEND_CAFFE_DENSECAP_MODEL_H_
