#ifndef NEXUS_BACKEND_CAFFE2_MODEL_H_
#define NEXUS_BACKEND_CAFFE2_MODEL_H_

#ifdef USE_CAFFE2

#include "nexus/backend/model_ins.h"
// Caffe2 headers
#include "caffe2/core/context_gpu.h"
#include "caffe2/core/predictor.h"

namespace nexus {
namespace backend {

class Caffe2Model : public ModelInstance {
 public:
  Caffe2Model(int gpu_id, const ModelInstanceConfig& config);

  Shape InputShape() final;

  std::unordered_map<std::string, Shape> OutputShapes() final;

  ArrayPtr CreateInputGpuArray() final;

  std::unordered_map<std::string, ArrayPtr> GetOutputGpuArrays() final;

  void Preprocess(std::shared_ptr<Task> task) final;

  void Forward(std::shared_ptr<BatchTask> batch_task) final;

  void ForwardAsync(std::shared_ptr<BatchTask> batch_task) final;

  void WaitOutput(std::shared_ptr<BatchTask> batch_task) final;

  void Postprocess(std::shared_ptr<Task> task) final;

 private:
  void LoadModel(const std::string& init_path, const std::string& predict_path,
                 const ModelInstanceConfig& config, caffe2::NetDef* init_net,
                 caffe2::NetDef* predict_net);

  std::pair<std::string, caffe2::Blob*> NewInputBlob();

  void LoadClassnames(const std::string& filename);

  std::unique_ptr<caffe2::CUDAContext> gpu_ctx_;
  std::string net_name_;
  std::unique_ptr<caffe2::Workspace> workspace_;
  caffe2::NetBase* net_;
  std::string input_blob_name_;
  std::string output_blob_name_;
  // image size
  int image_height_;
  int image_width_;
  // input shape of neural network
  Shape input_shape_;
  // output shape of neural network
  Shape output_shape_;
  // size of input in a single input
  size_t input_size_;
  // size of output in a single batch
  size_t output_size_;
  // Input tensor
  std::vector<std::pair<std::string, caffe2::Blob*> > input_blobs_;
  bool first_input_array_;
  // Output tensor
  caffe2::TensorCUDA* output_tensor_;

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

#endif // USE_CAFFE2

#endif // NEXUS_BACKEND_CAFFE2_MODEL_H_
