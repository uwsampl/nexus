#ifndef NEXUS_COMMON_METRIC_H_
#define NEXUS_COMMON_METRIC_H_

#include <atomic>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "nexus/common/time_util.h"

namespace nexus {

class Metric {
 public:
  virtual void Reset() = 0;
};

class Counter : public Metric {
 public:
  Counter();

  void Increase(uint64_t value);

  void Reset() final;
  
 private:
  std::atomic<uint64_t> count_;
};

class IntervalCounter : public Metric, public Tickable {
 public:
  IntervalCounter(uint32_t interval_sec);

  void Increase(uint64_t value);

  void Reset() override;

  std::vector<uint64_t> GetHistory();

 protected:
  void TickImpl() final;

 private:
  uint32_t tick_interval_sec_;
  TimePoint last_tick_time_;
  std::atomic<uint64_t> count_;
  std::vector<uint64_t> history_;
  std::mutex history_mutex_;
  std::atomic_bool running_;
};

class MovingAverage : public Metric, public Tickable {
 public:
  MovingAverage(uint32_t tick_interval_sec, uint32_t avg_interval_sec);

  void Increase(uint64_t value);

  void Reset() final;

  double rate();

 protected:
  void TickImpl() final;

 private:
  uint32_t avg_interval_sec_;
  std::atomic<uint64_t> count_;
  double alpha_;
  double rate_;
  std::mutex rate_mutex_;
};

class MetricRegistry {
 public:
  static MetricRegistry& Singleton();

  std::shared_ptr<IntervalCounter> CreateIntervalCounter(
      uint32_t interval_sec) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto metric = std::make_shared<IntervalCounter>(interval_sec);
    metrics_.insert(metric);
    return metric;
  }

  std::shared_ptr<MovingAverage> CreateMovingAverage(
      uint32_t tick_interval_sec, uint32_t avg_interval_sec) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto metric = std::make_shared<MovingAverage>(tick_interval_sec,
                                                  avg_interval_sec);
    metrics_.insert(metric);
    return metric;
  }

  void RemoveMetric(std::shared_ptr<Metric> metric);

 private:
  MetricRegistry() {}
  
  std::mutex mutex_;
  std::unordered_set<std::shared_ptr<Metric> > metrics_;
};

} // namespace nexus

#endif // NEXUS_COMMON_METRIC_H_
