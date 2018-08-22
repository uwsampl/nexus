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
  TensorflowModel(int gpu_id, const ModelInstanceConfig& config);

  ~TensorflowModel();

  Shape InputShape() final;

  std::unordered_map<std::string, Shape> OutputShapes() final;

  ArrayPtr CreateInputGpuArray() final;

  std::unordered_map<std::string, ArrayPtr> GetOutputGpuArrays() final;

  void Preprocess(std::shared_ptr<Task> task) final;

  void Forward(std::shared_ptr<BatchTask> batch_task) final;

  void Postprocess(std::shared_ptr<Task> task) final;

 private:
  tf::Tensor* NewInputTensor();

  void MarshalDetectionResult(
      const QueryProto& query, std::shared_ptr<Output> output,
      int im_height, int im_width, QueryResultProto* result);
  
  tf::SessionOptions gpu_option_;
  tf::SessionOptions cpu_option_;
  std::unique_ptr<tf::Session> session_;
  int image_height_;
  int image_width_;
  std::string input_layer_;
  Shape input_shape_;
  size_t input_size_;
  DataType input_data_type_;
  std::vector<std::string> output_layers_;
  std::unordered_map<std::string, Shape> output_shapes_;
  std::unordered_map<std::string, size_t> output_sizes_;
  std::vector<float> input_mean_;
  std::vector<float> input_std_;
  std::unordered_map<int, std::string> classnames_;
  tf::Allocator* gpu_allocator_;
  std::vector<std::unique_ptr<tf::Tensor> > input_tensors_;
  bool first_input_array_;
};

} // namespace backend
} // namespace nexus

#endif // USE_TENSORFLOW

#endif // NEXUS_BACKEND_TENSORFLOW_MODEL_H_
