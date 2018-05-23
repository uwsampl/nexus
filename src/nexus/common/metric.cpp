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

EWMA::EWMA(uint32_t sample_interval_sec, uint32_t avg_interval_sec) :
    sample_interval_sec_(sample_interval_sec),
    avg_interval_sec_(avg_interval_sec),
    rate_() {
  alpha_ = 1 - exp(-1. * sample_interval_sec_ / avg_interval_sec_);
}

EWMA::EWMA(const EWMA& other) :
    sample_interval_sec_(other.sample_interval_sec_),
    avg_interval_sec_(other.avg_interval_sec_),
    rate_(other.rate_),
    alpha_(other.alpha_) {}

void EWMA::AddSample(uint64_t count) {
  double current_rate = static_cast<double>(count) / sample_interval_sec_;
  if (rate_ < 0) {
    rate_ = current_rate;
  } else {
    rate_ += (current_rate - rate_) * alpha_;
  }
}

EWMA& EWMA::operator=(const EWMA& other) {
  if (this != &other) {
    sample_interval_sec_ = other.sample_interval_sec_;
    avg_interval_sec_ = other.avg_interval_sec_;
    rate_ = other.rate_;
    alpha_ = other.alpha_;
  }
  return *this;
}

MetricRegistry& MetricRegistry::Singleton() {
    static MetricRegistry metric_registry_;
    return metric_registry_;
}

std::shared_ptr<Counter> MetricRegistry::CreateCounter() {
  std::lock_guard<std::mutex> lock(mutex_);
  auto metric = std::make_shared<Counter>();
  metrics_.insert(metric);
  return metric;
}

std::shared_ptr<IntervalCounter> MetricRegistry::CreateIntervalCounter(
    uint32_t interval_sec) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto metric = std::make_shared<IntervalCounter>(interval_sec);
  metrics_.insert(metric);
  return metric;
}

void MetricRegistry::RemoveMetric(std::shared_ptr<Metric> metric) {
  std::lock_guard<std::mutex> lock(mutex_);
  metrics_.erase(metric);
}

} // namespace nexus
