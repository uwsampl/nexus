#ifndef NEXUS_BACKEND_PREFIX_MODEL_H_
#define NEXUS_BACKEND_PREFIX_MODEL_H_

#include "nexus/backend/model_ins.h"

namespace nexus {
namespace backend {

class PrefixModel : public ModelInstance {
 public:
  PrefixModel(int gpu_id, const ModelInstanceConfig& config);

  Shape InputShape() const final;

  std::unordered_map<std::string, Shape> OutputShapes() const final;

  ArrayPtr CreateInputGpuArray() final;

  std::unordered_map<std::string, ArrayPtr> GetOutputGpuArrays() final;

  void Preprocess(std::shared_ptr<Task> task) final;

  void Forward(std::shared_ptr<BatchTask> batch_task) final;

  void Postprocess(std::shared_ptr<Task> task) final;

 private:
  // Prefix model information
  int prefix_length_;
  std::shared_ptr<ModelInstance> prefix_model_;
  std::string prefix_output_name_;
  Shape prefix_output_shape_;
  std::unordered_map<std::string, ArrayPtr> prefix_output_arr_;
  // Suffix models information
  std::unordered_map<std::string, ModelInstancePtr> suffix_models_;
  std::unordered_map<std::string, ArrayPtr> suffix_input_arrays_;
  std::unordered_map<std::string, std::string> suffix_output_names_;
  std::unordered_map<std::string, size_t> suffix_output_sizes_;
  size_t max_suffix_output_size_;
};

} // namespace backend
} // namespace nexus

#endif // NEXUS_BACKEND_PREFIX_MODEL_H_
