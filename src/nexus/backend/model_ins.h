#ifndef NEXUS_BACKEND_MODEL_INS_H_
#define NEXUS_BACKEND_MODEL_INS_H_

#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <yaml-cpp/yaml.h>

#include "backend/batch.h"
#include "backend/task.h"
#include "common/block_queue.h"
#include "common/metric.h"
#include "common/model_def.h"
#include "common/spinlock.h"
#include "proto/nnquery.pb.h"
#include "proto/control.pb.h"

namespace nexus {
namespace backend {

class ModelInstance {
 public:
  ModelInstance(int gpu_id, std::string model_id, std::string model_name,
                ModelType type, uint32_t batch, uint32_t max_batch,
                BlockPriorityQueue<Task>& task_queue);

  ~ModelInstance() {}

  std::string model_id() const { return model_id_; }

  std::string model_name() const { return model_name_; }

  uint32_t batch() const { return batch_; }

  uint32_t max_batch() const { return max_batch_; }

  ModelType type() const { return type_; }
  
  virtual Framework framework() const = 0;

  void Init();

  //void UpdateMaxBatch(size_t max_batch);

  bool Preprocess(std::shared_ptr<Task> task);

  void Forward(size_t min_batch = 1);

  void Postprocess(std::shared_ptr<Task> task);

  uint64_t Tick();

  double GetRate();

 protected:
  virtual void InitBatchInputArray() = 0;

  //virtual void UpdateMaxBatchImpl() = 0;

  virtual void PreprocessImpl(std::shared_ptr<Task> task,
                              std::vector<ArrayPtr>* input_arrays) = 0;

  virtual void ForwardImpl(BatchInput* batch_input,
                           BatchOutput* batch_output) = 0;

  virtual void PostprocessImpl(std::shared_ptr<Task> task, Output* ret) = 0;

 private:
  void AppendInputs(std::shared_ptr<Task> task,
                    const std::vector<ArrayPtr>& input_arrays);

  std::unique_ptr<BatchInput> GetBatchInput(size_t min_batch);

  void RemoveOutput(uint64_t batch_id);

 protected:
  int gpu_id_;
  std::string model_id_;
  std::string model_name_;
  ModelType type_;
  uint32_t batch_;
  uint32_t max_batch_;
  bool need_update_max_batch_;
  BlockPriorityQueue<Task>& task_queue_;
  CPUDevice* cpu_device_;
  GPUDevice* gpu_device_;
  ArrayPtr batch_input_array_;
  std::atomic<uint64_t> batch_id_;
  std::priority_queue<std::shared_ptr<Input>,
                      std::vector<std::shared_ptr<Input> >,
                      CompareDeadlineItem> input_queue_;
  std::unordered_map<uint64_t, std::shared_ptr<BatchOutput> > output_pool_;
  std::mutex input_mutex_;
  std::mutex output_mutex_;
  std::shared_ptr<MovingAverage> meter_;
};

using ModelInstancePtr = std::shared_ptr<ModelInstance>;

ModelInstancePtr CreateModelInstance(
    int gpu_id, const ModelInstanceDesc& model_inst_desc, YAML::Node info,
    BlockPriorityQueue<Task>& task_queue);

} // namespace backend
} // namespace nexus


#endif // NEXUS_BACKEND_MODEL_INS_H_
