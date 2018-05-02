#ifndef NEXUS_BACKEND_CAFFE2_MODEL_H_
#define NEXUS_BACKEND_CAFFE2_MODEL_H_

#if USE_CAFFE2 == 1

#include "nexus/backend/model_ins.h"
// Caffe2 headers
#include "caffe2/core/context_gpu.h"
#include "caffe2/core/predictor.h"

namespace nexus {
namespace backend {

class Caffe2Model : public ModelInstance {
 public:
  Caffe2Model(int gpu_id, const std::string& model_name, uint32_t version,
              const std::string& type, uint32_t batch, uint32_t max_batch,
              BlockPriorityQueue<Task>& task_queue, const YAML::Node& info);

  ~Caffe2Model() {}

  std::string framework() const final { return "caffe2"; }

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
  std::unique_ptr<caffe2::CUDAContext> gpu_ctx_;
  std::string net_name_;
  std::unique_ptr<caffe2::Workspace> workspace_;
  std::string input_blob_name_;
  std::string output_blob_name_;
  // image size
  int image_height_;
  int image_width_;
  // input shape of neural network
  std::vector<int> input_shape_;
  // output shape of neural network
  std::vector<int> output_shape_;
  // size of input in a single input
  size_t input_size_;
  // size of output in a single batch
  size_t output_size_;
  // Input tensor
  caffe2::TensorCUDA* input_tensor_;
  // Output tensor
  const caffe2::TensorCUDA* output_tensor_;

  std::vector<std::string> classnames_;
  bool has_mean_file_;
  std::vector<float> mean_value_;
  std::vector<float> mean_blob_;
  float scale_;
  
  // transformer for input
  //std::unique_ptr<caffe::DataTransformer<float> > transformer_;
};

} // namespace backend
} // namespace nexus

#endif // USE_CAFFE2 == 1

#endif // NEXUS_BACKEND_CAFFE2_MODEL_H_
