#ifndef NEXUS_COMMON_BLOCK_QUEUE_H_
#define NEXUS_COMMON_BLOCK_QUEUE_H_

#include <chrono>
#include <condition_variable>
#include <queue>
#include <memory>
#include <mutex>

#include "nexus/common/time_util.h"

namespace nexus {

template <class T>
class BlockQueue {
 public:
  // infinite queue size
  BlockQueue(): max_size_(0) {}

  // queue max size is max_size
  BlockQueue(size_t max_size): max_size_(max_size) {}

  size_t size() const { return queue_.size(); }

  bool push(std::shared_ptr<T> item) {
    std::unique_lock<std::mutex> lock(mutex_);
    not_full_.wait(lock, [this](){
        return max_size_ == 0 || queue_.size() >= max_size_; });
    queue_.push(std::move(item));
    lock.unlock();
    not_empty_.notify_one();
    return true;
  }

  bool push(std::shared_ptr<T> item, const std::chrono::microseconds& timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!not_full_.wait_for(lock, timeout, [this](){
          return max_size_ == 0 || queue_.size() >= max_size_; })) {
      return false;
    }
    queue_.push(std::move(item));
    lock.unlock();
    not_empty_.notify_one();
    return true;
  }

  std::shared_ptr<T> pop() {
    std::unique_lock<std::mutex> lock(mutex_);
    not_empty_.wait(lock, [this](){ return queue_.size() != 0; });
    std::shared_ptr<T> item = queue_.front();
    queue_.pop();
    lock.unlock();
    not_full_.notify_one();
    return item;
  }

  std::shared_ptr<T> pop(const std::chrono::microseconds& timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!not_empty_.wait_for(lock, timeout, [this](){
          return queue_.size() != 0; })) {
      return nullptr;
    }
    std::shared_ptr<T> item = queue_.front();
    queue_.pop();
    lock.unlock();
    not_full_.notify_one();
    return item;
  }

 private:
  size_t max_size_;
  std::queue<std::shared_ptr<T> > queue_;
  std::mutex mutex_;
  std::condition_variable not_full_;
  std::condition_variable not_empty_;
};

class DeadlineItem {
 public:
  DeadlineItem() {
    begin_ = Clock::now();
  }
  
  DeadlineItem(TimePoint deadline) :
      deadline_(deadline) {}
  
  void SetDeadline(std::chrono::milliseconds time_budget) {
    deadline_ = begin_ + time_budget;
  }
  
  void SetDeadline(std::chrono::microseconds time_budget) {
    deadline_ = begin_ + time_budget;
  }

  TimePoint deadline() const { return deadline_; }

 protected:
  TimePoint begin_;
  TimePoint deadline_;
};

class CompareDeadlineItem {
 public:
  bool operator()(std::shared_ptr<DeadlineItem> lhs,
                  std::shared_ptr<DeadlineItem> rhs) {
    return lhs->deadline() > rhs->deadline();
  }
};

template <class T,
          typename = typename std::enable_if<std::is_base_of<DeadlineItem, T>::value>::type>
class BlockPriorityQueue {
 public:
  // infinite queue size
  BlockPriorityQueue(): max_size_(0) {}

  // queue max size is max_size
  BlockPriorityQueue(size_t max_size): max_size_(max_size) {}

  size_t size() const { return queue_.size(); }

  bool push(std::shared_ptr<T> item) {
    std::unique_lock<std::mutex> lock(mutex_);
    not_full_.wait(lock, [this](){
        return max_size_ == 0 || queue_.size() >= max_size_; });
    queue_.push(std::move(item));
    lock.unlock();
    not_empty_.notify_one();
    return true;
  }

  bool push(std::shared_ptr<T> item, const std::chrono::microseconds& timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!not_full_.wait_for(lock, timeout, [this](){
          return max_size_ == 0 || queue_.size() >= max_size_; })) {
      return false;
    }
    queue_.push(std::move(item));
    lock.unlock();
    not_empty_.notify_one();
    return true;
  }

  std::shared_ptr<T> pop() {
    std::unique_lock<std::mutex> lock(mutex_);
    not_empty_.wait(lock, [this](){ return queue_.size() != 0; });
    std::shared_ptr<T> item = queue_.top();
    queue_.pop();
    lock.unlock();
    not_full_.notify_one();
    return item;
  }

  std::shared_ptr<T> pop(const std::chrono::microseconds& timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!not_empty_.wait_for(lock, timeout, [this](){
          return queue_.size() != 0; })) {
      return nullptr;
    }
    std::shared_ptr<T> item = queue_.top();
    queue_.pop();
    lock.unlock();
    not_full_.notify_one();
    return item;
  }

 private:
  size_t max_size_;
  std::priority_queue<std::shared_ptr<T>, std::vector<std::shared_ptr<T> >,
                      CompareDeadlineItem> queue_;
  std::mutex mutex_;
  std::condition_variable not_full_;
  std::condition_variable not_empty_;
};

} // namespace nexus

#endif // NEXUS_COMMON_BLOCK_QUEUE_H_
