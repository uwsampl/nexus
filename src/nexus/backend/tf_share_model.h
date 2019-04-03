#ifndef NEXUS_TFSHAREMODEL_H
#define NEXUS_TFSHAREMODEL_H

#include "nexus/backend/model_ins.h"
#include "nexus/backend/tensorflow_model.h"

namespace nexus {
namespace backend {

class TFShareModel : public ModelInstance {
 public:
  TFShareModel(int gpu_id, const ModelInstanceConfig& config);
  void set_batch(size_t batch) override;
  Shape InputShape() override;
  std::unordered_map<std::string, Shape> OutputShapes() override;
  ArrayPtr CreateInputGpuArray() override;
  std::unordered_map<std::string, ArrayPtr> GetOutputGpuArrays() override;
  void Preprocess(std::shared_ptr<Task> task) override;
  void Forward(std::shared_ptr<BatchTask> batch_task) override;
  void Postprocess(std::shared_ptr<Task> task) override;

 private:
  std::unique_ptr<ModelInstance> tf_model_;
};

}
}

#endif //NEXUS_TFSHAREMODEL_H
