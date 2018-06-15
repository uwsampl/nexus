#ifndef NEXUS_BACKEND_MODEL_EXEC_H_
#define NEXUS_BACKEND_MODEL_EXEC_H_

#include <atomic>
#include <memory>
#include <mutex>

#include "nexus/backend/model_ins.h"
#include "nexus/common/block_queue.h"
#include "nexus/common/model_db.h"

namespace nexus {
namespace backend {

class ModelExecutor {
 public:
  ModelExecutor(std::shared_ptr<ModelInstance> model,
                BlockPriorityQueue<Task>& task_queue);

  std::shared_ptr<ModelInstance> model_instance() const {
    return model_;
  }

  void AddTask(std::shared_ptr<Task> task);

  void Execute();

 private:
  void GetBatchInput(std::shared_ptr<BatchTask> batch_task);

  void RemoveTask(std::shared_ptr<Task> task);

  std::shared_ptr<ModelInstance> model_;
  BlockPriorityQueue<Task>& task_queue_;
  const ModelProfile* profile_;
  std::unordered_map<uint64_t, std::shared_ptr<Task> > processing_tasks_;
  std::priority_queue<std::shared_ptr<Input>,
                      std::vector<std::shared_ptr<Input> >,
                      CompareDeadlineItem> input_queue_;
  std::shared_ptr<Array> input_array_;
  std::atomic<uint64_t> batch_id_;
  std::mutex mu_;
};

} // namespace backend
} // namespace nexus

#endif // NEXUS_BACKEND_MODEL_EXEC_H_
