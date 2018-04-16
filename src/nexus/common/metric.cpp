#include <cmath>

#include "nexus/common/metric.h"

namespace nexus {

Counter::Counter() :
    count_(0) {
}

void Counter::Increase(uint64_t value) {
  count_.fetch_add(value, std::memory_order_relaxed);
}

void Counter::Reset() {
  count_.exchange(0, std::memory_order_relaxed);
}

IntervalCounter::IntervalCounter(uint32_t interval_sec) :
    Tickable(interval_sec),
    count_(0) {
}

void IntervalCounter::Increase(uint64_t value) {
  count_.fetch_add(value, std::memory_order_relaxed);
}

void IntervalCounter::Reset() {
  std::lock_guard<std::mutex> guard(history_mutex_);
  count_.exchange(0, std::memory_order_relaxed);
  history_.clear();
}

std::vector<uint64_t> IntervalCounter::GetHistory() {
  std::lock_guard<std::mutex> guard(history_mutex_);
  std::vector<uint64_t> ret(std::move(history_));
  history_.clear();
  return ret;
}

void IntervalCounter::TickImpl() {
  uint64_t count = count_.exchange(0, std::memory_order_relaxed);
  std::lock_guard<std::mutex> guard(history_mutex_);
  history_.push_back(count);
}

MovingAverage::MovingAverage(uint32_t tick_interval_sec,
                             uint32_t avg_interval_sec) :
    Tickable(tick_interval_sec),
    avg_interval_sec_(avg_interval_sec),
    count_(0), 
    rate_(-1) {
  alpha_ = 1 - exp(-1. * tick_interval_sec / avg_interval_sec);
}

void MovingAverage::Increase(uint64_t value) {
  count_.fetch_add(value, std::memory_order_relaxed);
}

void MovingAverage::Reset() {
  std::lock_guard<std::mutex> guard(rate_mutex_);
  rate_ = -1;
  count_.store(0, std::memory_order_seq_cst);
}

double MovingAverage::rate() {
  std::lock_guard<std::mutex> guard(rate_mutex_);
  return rate_;
}

void MovingAverage::TickImpl() {
  std::lock_guard<std::mutex> guard(rate_mutex_);
  uint64_t count = count_.exchange(0, std::memory_order_relaxed);
  double current_rate = static_cast<double>(count) / tick_interval_sec_;
  if (rate_ < 0) {
    rate_ = current_rate;
  } else {
    rate_ += (current_rate - rate_) * alpha_;
  }
}

MetricRegistry& MetricRegistry::Singleton() {
    static MetricRegistry metric_registry_;
    return metric_registry_;
}

} // namespace nexus
