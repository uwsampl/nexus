#ifndef NEXUS_BACKEND_TENSORFLOW_MODEL_H_
#define NEXUS_BACKEND_TENSORFLOW_MODEL_H_

#ifdef USE_TENSORFLOW

#include "nexus/backend/model_ins.h"
// Tensorflow headers
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/public/session.h"

namespace tf = tensorflow;

namespace nexus {
namespace backend {

class TensorflowModel : public ModelInstance {
 public:
  TensorflowModel(int gpu_id, const ModelInstanceConfig& config,
                  const YAML::Node& info);

  ~TensorflowModel();

  ArrayPtr CreateInputGpuArray() final;

  std::unordered_map<std::string, size_t> OutputSizes() const final;

  void Preprocess(std::shared_ptr<Task> task) final;

  void Forward(BatchInput* batch_input, BatchOutput* batch_output) final;

  void Postprocess(std::shared_ptr<Task> task) final;

 private:
  tf::Tensor* NewInputTensor();
  
  void LoadClassnames(const std::string& filepath);
  
  tf::SessionOptions gpu_option_;
  tf::SessionOptions cpu_option_;
  std::unique_ptr<tf::Session> session_;
  int image_height_;
  int image_width_;
  size_t input_size_;
  size_t output_size_;
  std::string input_layer_;
  std::string output_layer_;
  std::vector<float> input_mean_;
  std::vector<float> input_std_;
  std::vector<std::string> classnames_;
  tf::Allocator* gpu_allocator_;
  std::vector<std::unique_ptr<tf::Tensor> > input_tensors_;
  bool first_input_array_;
};

} // namespace backend
} // namespace nexus

#endif // USE_TENSORFLOW

#endif // NEXUS_BACKEND_TENSORFLOW_MODEL_H_
