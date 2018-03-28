#ifndef NEXUS_BACKEND_DARKNET_MODEL_H_
#define NEXUS_BACKEND_DARKNET_MODEL_H_

#include <memory>
#include <string>

#include "backend/model_ins.h"
// Darknet headers
extern "C" {
#include "../../darknet/darknet_server.h"
}

namespace nexus {
namespace backend {

class DarknetModel : public ModelInstance {
 public:
  DarknetModel(int gpu_id, std::string model_id, std::string model_name,
               ModelType type, uint32_t batch, uint32_t max_batch,
               BlockPriorityQueue<Task>& task_queue, const YAML::Node& info);

  ~DarknetModel();

  Framework framework() const final { return DARKNET; }

 protected:
  void InitBatchInputArray() final;

  //void UpdateMaxBatchImpl() final;

  void PreprocessImpl(std::shared_ptr<Task> task,
                      std::vector<ArrayPtr>* input_arrays) final;

  void ForwardImpl(BatchInput* batch_input, BatchOutput* batch_output) final;

  void PostprocessImpl(std::shared_ptr<Task> task, Output* output) final;

  void LoadClassnames(const std::string& filepath);

  void MarshalDetectionResult(
      const QueryProto& query, const float* probs, size_t nprobs,
      const int* boxes, size_t nboxes, QueryResultProto* result);

  void MarshalClassificationResult(
      const QueryProto& query, const float* prob, size_t nprobs,
      float threshold, QueryResultProto* result);

 private:
  network* net_;
  size_t output_layer_id_;
  size_t input_size_;
  size_t output_size_;
  std::vector<std::string> classnames_;
};

} // namespace backend
} // namespace nexus

#endif // NEXUS_BACKEND_DARKNET_MODEL_H_
