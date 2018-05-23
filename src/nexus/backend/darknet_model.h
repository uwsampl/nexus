#ifndef NEXUS_BACKEND_DARKNET_MODEL_H_
#define NEXUS_BACKEND_DARKNET_MODEL_H_

#ifdef USE_DARKNET

#include <memory>
#include <string>

#include "nexus/backend/model_ins.h"
// Darknet headers
extern "C" {
#include "darknet_server.h"
}

namespace nexus {
namespace backend {

class DarknetModel : public ModelInstance {
 public:
  DarknetModel(int gpu_id, const ModelInstanceConfig& config);

  ~DarknetModel();

  Shape InputShape() final;
  
  std::unordered_map<std::string, Shape> OutputShapes() final;

  ArrayPtr CreateInputGpuArray() final;

  std::unordered_map<std::string, ArrayPtr> GetOutputGpuArrays() final;

  void Preprocess(std::shared_ptr<Task> task) final;

  void Forward(std::shared_ptr<BatchTask> batch_task) final;

  void Postprocess(std::shared_ptr<Task> task) final;

 private:
  void LoadClassnames(const std::string& filepath);

  void MarshalDetectionResult(
      const QueryProto& query, const float* probs, size_t nprobs,
      const int* boxes, size_t nboxes, QueryResultProto* result);

  network* net_;
  int image_height_;
  int image_width_;
  Shape input_shape_;
  Shape output_shape_;
  size_t input_size_;
  size_t output_size_;
  size_t output_layer_id_;
  std::string output_name_;
  std::vector<std::string> classnames_;
  bool first_input_array_;
};

} // namespace backend
} // namespace nexus

#endif // USE_DARKNET

#endif // NEXUS_BACKEND_DARKNET_MODEL_H_
