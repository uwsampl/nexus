#ifndef NEXUS_BACKEND_SHARE_PREFIX_MODEL_H_
#define NEXUS_BACKEND_SHARE_PREFIX_MODEL_H_

#include <mutex>

#include "nexus/backend/model_ins.h"

namespace nexus {
namespace backend {

class SharePrefixModel : public ModelInstance {
 public:
  SharePrefixModel(int gpu_id, const ModelInstanceConfig& config);

  virtual void set_batch(size_t batch) override;

  Shape InputShape() final;

  std::unordered_map<std::string, Shape> OutputShapes() final;

  ArrayPtr CreateInputGpuArray() final;

  std::unordered_map<std::string, ArrayPtr> GetOutputGpuArrays() final;

  void Preprocess(std::shared_ptr<Task> task) final;

  void Forward(std::shared_ptr<BatchTask> batch_task) final;

  void Postprocess(std::shared_ptr<Task> task) final;

  int num_model_sessions();

  std::vector<std::string> ModelSessions();

  bool HasModelSession(const std::string& model_sess_id);

  bool AddModelSession(const ModelSession& model_sess);

  void RemoveModelSession(const std::string& model_sess_id);

 private:
  // Prefix model information
  int prefix_length_;
  std::unique_ptr<ModelInstance> prefix_model_;
  std::string prefix_output_name_;
  Shape prefix_output_shape_;
  std::unordered_map<std::string, ArrayPtr> prefix_batch_output_arr_;
  // Suffix models information
  std::unordered_map<std::string,
                     std::shared_ptr<ModelInstance> > suffix_models_;
  std::unordered_map<std::string, ArrayPtr> suffix_input_arrays_;
  std::unordered_map<std::string, std::string> suffix_output_names_;
  std::unordered_map<std::string, size_t> suffix_output_sizes_;
  size_t max_suffix_output_size_;
  // Guard suffix_models_, suffix_input_arrays_, suffix_output_names_,
  // suffix_output_sizes_, max_suffix_output_size_
  std::mutex suffix_mu_;
};

} // namespace backend
} // namespace nexus

#endif // NEXUS_BACKEND_SHARE_PREFIX_MODEL_H_
