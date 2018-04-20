#ifndef NEXUS_BACKEND_MODEL_INS_H_
#define NEXUS_BACKEND_MODEL_INS_H_

#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <yaml-cpp/yaml.h>

#include "nexus/backend/batch.h"
#include "nexus/backend/task.h"
#include "nexus/common/block_queue.h"
#include "nexus/common/metric.h"
#include "nexus/common/model_def.h"
#include "nexus/common/spinlock.h"
#include "nexus/proto/nnquery.pb.h"
#include "nexus/proto/control.pb.h"

namespace nexus {
namespace backend {

class ModelProfiler;

class ModelInstance {
 public:
  ModelInstance(int gpu_id, const std::string& model_name,
                uint32_t version, const std::string& type,
                uint32_t batch, uint32_t max_batch,
                BlockPriorityQueue<Task>& task_queue);

  virtual ~ModelInstance();

  std::string model_name() const { return model_name_; }

  std::string type() const { return type_; }
  
  uint32_t batch() const { return batch_; }

  void set_batch(size_t batch);

  uint32_t max_batch() const { return max_batch_; }

  std::shared_ptr<IntervalCounter> counter() const { return counter_; }
      
  virtual std::string framework() const = 0;

  virtual std::string profile_id() const = 0;

  void Setup();

  bool Preprocess(std::shared_ptr<Task> task);

  void Forward(size_t min_batch = 1);

  void Postprocess(std::shared_ptr<Task> task);

 protected:
  virtual void InitBatchInputArray() = 0;

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

  friend class ModelProfiler;

 protected:
  int gpu_id_;
  std::string model_name_;
  uint32_t version_;
  std::string type_;
  uint32_t batch_;
  uint32_t max_batch_;
  BlockPriorityQueue<Task>& task_queue_;
  std::shared_ptr<IntervalCounter> counter_;
  std::atomic<uint64_t> batch_id_;
  CPUDevice* cpu_device_;
  GPUDevice* gpu_device_;
  ArrayPtr batch_input_array_;
  std::priority_queue<std::shared_ptr<Input>,
                      std::vector<std::shared_ptr<Input> >,
                      CompareDeadlineItem> input_queue_;
  std::unordered_map<uint64_t, std::shared_ptr<BatchOutput> > output_pool_;
  std::mutex input_mutex_;
  std::mutex output_mutex_;
};

using ModelInstancePtr = std::shared_ptr<ModelInstance>;

ModelInstancePtr CreateModelInstance(
    int gpu_id, const ModelInstanceConfig& config, YAML::Node info,
    BlockPriorityQueue<Task>& task_queue);

} // namespace backend
} // namespace nexus


#endif // NEXUS_BACKEND_MODEL_INS_H_
