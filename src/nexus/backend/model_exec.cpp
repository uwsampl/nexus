#include <gflags/gflags.h>

#include "nexus/backend/model_exec.h"
#include "nexus/backend/model_ins.h"
#include "nexus/backend/share_prefix_model.h"
#include "nexus/common/model_db.h"

DEFINE_int32(count_interval, 1, "Interval to count number of requests in sec");

namespace nexus {
namespace backend {

ModelExecutor::ModelExecutor(int gpu_id, const ModelInstanceConfig& config,
                             BlockPriorityQueue<Task>& task_queue) :
    backup_(config.backup()),
    task_queue_(task_queue),
    batch_id_(0),
    open_requests_(0) {
  // Create ModelInstance
  CreateModelInstance(gpu_id, config, &model_);
  auto gpu_device = DeviceManager::Singleton().GetGPUDevice(gpu_id);
  profile_ = ModelDatabase::Singleton().GetModelProfile(
      gpu_device->device_name(), model_->profile_id());
  counter_ = MetricRegistry::Singleton().CreateIntervalCounter(
      FLAGS_count_interval);
  input_array_ = model_->CreateInputGpuArray();
  for (auto const& info : config.backup_backend()) {
    backup_backends_.push_back(info.node_id());
  }
}

ModelExecutor::~ModelExecutor() {
  MetricRegistry::Singleton().RemoveMetric(counter_);
}

bool ModelExecutor::IsSharePrefixModel() const {
  return (dynamic_cast<SharePrefixModel*>(model_.get()) != nullptr);
}

bool ModelExecutor::HasBackup() {
  std::lock_guard<std::mutex> lock(backup_mu_);
  return (backup_backends_.size() > 0);
}

std::vector<uint32_t> ModelExecutor::BackupBackends() {
  std::lock_guard<std::mutex> lock(backup_mu_);
  return backup_backends_;
}

void ModelExecutor::UpdateBackupBackends(const ModelInstanceConfig& config) {
  std::lock_guard<std::mutex> lock(backup_mu_);
  backup_backends_.clear();
  for (auto& info : config.backup_backend()) {
    backup_backends_.push_back(info.node_id());
  }
}

bool ModelExecutor::Preprocess(std::shared_ptr<Task> task, bool force) {
  int cnt = 1;
  if (task->query.window_size() > 0) {
    cnt = task->query.window_size();
  }
  bool limit = !force && HasBackup();
  if (!IncreaseOpenRequests(cnt, limit)) {
    return false;
  }
  counter_->Increase(cnt);
  
  model_->Preprocess(task);
  if (task->result.status() != CTRL_OK) {
    return false;
  }
  std::lock_guard<std::mutex> lock(task_mu_);
  processing_tasks_.emplace(task->task_id, task);
  for (auto input : task->inputs) {
    input_queue_.push(input);
  }
  return true;
}

bool ModelExecutor::AddPreprocessedTask(std::shared_ptr<Task> task,
                                        bool force) {
  int cnt = task->inputs.size();
  bool limit = !force && HasBackup();
  if (!IncreaseOpenRequests(cnt, limit)) {
    return false;
  }
  counter_->Increase(cnt);
  std::lock_guard<std::mutex> lock(task_mu_);
  processing_tasks_.emplace(task->task_id, task);
  for (auto input : task->inputs) {
    input_queue_.push(input);
  }
  return true;
}

void ModelExecutor::Postprocess(std::shared_ptr<Task> task) {
  model_->Postprocess(task);
}

uint64_t ModelExecutor::Execute(uint32_t batch) {
  std::shared_ptr<BatchTask> batch_task;
  int dequeue_cnt;
  if (batch == 0) {
    batch = model_->batch();
  }
  
  auto t1 = std::chrono::high_resolution_clock::now();
  std::tie(batch_task, dequeue_cnt) = GetBatchTask(batch);
  auto t2 = std::chrono::high_resolution_clock::now();
  if (batch_task->batch_size() == 0) {
    DecreaseOpenRequests(dequeue_cnt);
    std::lock_guard<std::mutex> lock(time_mu_);
    last_exec_finish_ = t2;
    return std::chrono::duration_cast<std::chrono::microseconds>(
        t2 - t1).count();
  }
  uint64_t batch_id = batch_id_.fetch_add(1, std::memory_order_relaxed);
  batch_task->set_batch_id(batch_id);
  
  // Each time recompute output sizes because it might change for prefix model
  std::unordered_map<std::string, size_t> output_sizes;
  for (auto iter : model_->OutputShapes()) {
    output_sizes.emplace(iter.first, iter.second.NumElements(1));
  }
  batch_task->CreateOutputArrays(output_sizes,
                                 DeviceManager::Singleton().GetCPUDevice());
  model_->Forward(batch_task);
  auto t3 = std::chrono::high_resolution_clock::now();
  {
    std::lock_guard<std::mutex> lock(time_mu_);
    last_exec_finish_ = t3;
  }
  DecreaseOpenRequests(dequeue_cnt);
  
  auto memcpy_lat = std::chrono::duration_cast<std::chrono::microseconds>(
      t2 - t1).count();
  auto forward_lat = std::chrono::duration_cast<std::chrono::microseconds>(
      t3 - t2).count();
  LOG(INFO) << model_->model_session_id() << " forwards batch " <<
      batch_task->batch_id() << ", size " << batch_task->batch_size() <<
      ", memcpy " << memcpy_lat << " us, forward " << forward_lat << " us";

  auto outputs = batch_task->outputs();
  auto tasks = batch_task->tasks();
  for (int i = 0; i < outputs.size(); ++i) {
    auto output = outputs[i];
    auto task = tasks[i];
    if (task->AddOutput(output)) {
      RemoveTask(task);
    }
  }
  return memcpy_lat + forward_lat;
}

int ModelExecutor::NumberOfOpenRequests() const {
  return open_requests_.load(std::memory_order_relaxed);
}

TimePoint ModelExecutor::LastExecuteFinishTime() {
  std::lock_guard<std::mutex> lock(time_mu_);
  return last_exec_finish_;
}

bool ModelExecutor::IncreaseOpenRequests(int cnt, bool limit_max_batch) {
  if (!limit_max_batch) {
    int prev = open_requests_.fetch_add(cnt, std::memory_order_relaxed);
    return true;
  }
  int curr_val = open_requests_.load();
  while (true) {
    int new_val = curr_val + cnt;
    if (new_val > model_->max_batch()) {
      return false;
    }
    if (open_requests_.compare_exchange_strong(curr_val, new_val)) {
      break;
    }
  }
  return true;
}

void ModelExecutor::DecreaseOpenRequests(int cnt) {
  int prev = open_requests_.fetch_sub(cnt, std::memory_order_relaxed);
  CHECK_GE(prev, cnt) << "Negative value in open requests";
}

std::pair<std::shared_ptr<BatchTask>, int> ModelExecutor::GetBatchTask(
    uint32_t batch_size) {
  auto batch_task = std::make_shared<BatchTask>(model_->max_batch());
  batch_task->SetInputArray(input_array_);
  if (batch_size > model_->max_batch()) {
    batch_size = model_->max_batch();
  }
  if (batch_size > input_queue_.size()) {
    batch_size = input_queue_.size();
  }

  std::unique_lock<std::mutex> lock(task_mu_);
  TimePoint now = Clock::now();
  TimePoint finish;
  if (profile_ != nullptr) {
    float latency = profile_->GetForwardLatency(batch_size);
    finish = now + std::chrono::microseconds(int(latency));
  }
  int cnt = 0;
  while (batch_task->batch_size() < batch_size && !input_queue_.empty()) {
    auto input = std::move(input_queue_.top());
    input_queue_.pop();
    ++cnt;
    auto task = processing_tasks_.at(input->task_id);
    task->timer.Record("exec");
    if (task->result.status() != CTRL_OK ||
        (profile_ != nullptr && input->deadline() < finish)) {
      if (task->AddVirtualOutput(input->index)) {
        lock.unlock();
        RemoveTask(task);
        lock.lock();
      }
    } else {
      batch_task->AppendInput(input, task);
    }
    // Check whether there is enough requests left to fill the batch size
    if (profile_ != nullptr && \
        batch_size > batch_task->batch_size() + input_queue_.size()) {
      batch_size = batch_task->batch_size() + input_queue_.size();
      float latency = profile_->GetForwardLatency(batch_size);
      finish = now + std::chrono::microseconds(int(latency));
    }
  }
  lock.unlock();
  return {batch_task, cnt};
}

void ModelExecutor::RemoveTask(std::shared_ptr<Task> task) {
  std::lock_guard<std::mutex> lock(task_mu_);
  task->stage = kPostprocess;
  task_queue_.push(task);
  processing_tasks_.erase(task->task_id);
}

} // namespace backend
} // namespace nexus

