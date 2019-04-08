#ifndef NEXUS_TFSHAREMODEL_H
#define NEXUS_TFSHAREMODEL_H

#include <mutex>
#include <unordered_set>
#include "nexus/backend/model_ins.h"
#include "nexus/backend/tensorflow_model.h"

namespace nexus {
namespace backend {

class TFShareModel : public ModelInstance {
 public:
  void set_batch(size_t batch) override;
  Shape InputShape() override;
  std::unordered_map<std::string, Shape> OutputShapes() override;
  ArrayPtr CreateInputGpuArray() override;
  std::unordered_map<std::string, ArrayPtr> GetOutputGpuArrays() override;
  void Preprocess(std::shared_ptr<Task> task) override;
  void Forward(std::shared_ptr<BatchTask> batch_task) override;
  void Postprocess(std::shared_ptr<Task> task) override;

  TFShareModel(int gpu_id, const ModelInstanceConfig& config);
  bool AddModelSession(const ModelSession& model_sess);
  bool RemoveModelSession(const ModelSession& model_sess);
  size_t num_model_sessions();

 private:
  size_t num_suffixes_;
  std::shared_ptr<TFShareInfo> tf_share_info_;
  std::unique_ptr<TensorflowModel> tf_model_;
  std::mutex loaded_suffixes_mutex_;
  std::unordered_set<std::string> loaded_suffixes_;
  std::unordered_map<std::string, std::unordered_map<int, std::string>> classnames_;
};

}
}

#endif //NEXUS_TFSHAREMODEL_H
