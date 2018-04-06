#include <cmath>
#include "nexus/common/metric.h"
#include "nexus/common/time_util.h"

namespace nexus {

Counter::Counter(std::string name) :
    name_(name),
    count_(0) {
}

void Counter::Increase(uint64_t value) {
  count_.fetch_add(value, std::memory_order_relaxed);
}

void Counter::Reset() {
  count_.exchange(0, std::memory_order_relaxed);
}

Meter::Meter(std::string name, uint32_t tick_interval_sec) :
    name_(name),
    tick_interval_sec_(tick_interval_sec),
    count_(0) {
  running_ = true;
  tick_thread_ = std::thread(&Meter::Tick, this);
}

Meter::~Meter() {
  running_ = false;
  tick_thread_.join();
}

void Meter::Increase(uint64_t value) {
  count_.fetch_add(value, std::memory_order_relaxed);
}

void Meter::Reset() {
  std::lock_guard<std::mutex> guard(history_mutex_);
  history_.clear();
  count_.exchange(0, std::memory_order_relaxed);
}

std::vector<uint64_t> Meter::GetHistory() {
  std::lock_guard<std::mutex> guard(history_mutex_);
  std::vector<uint64_t> ret(std::move(history_));
  history_.clear();
  return ret;
}

void Meter::Tick() {
  std::chrono::seconds tick_interval(tick_interval_sec_);
  auto next_event = Clock::now() + tick_interval;
  while (running_) {
    std::this_thread::sleep_until(next_event);
    std::lock_guard<std::mutex> guard(history_mutex_);
    uint64_t count = count_.exchange(0, std::memory_order_relaxed);
    history_.push_back(count);
    next_event += tick_interval;
  }
}

MovingAverage::MovingAverage(std::string name, uint32_t tick_interval_sec,
                             uint32_t avg_interval_sec) :
    name_(name),
    tick_interval_sec_(tick_interval_sec),
    avg_interval_sec_(avg_interval_sec),
    rate_(-1),
    count_(0) {
  alpha_ = 1 - exp(-1. * tick_interval_sec / avg_interval_sec);
}

void MovingAverage::Increase(uint64_t value) {
  count_.fetch_add(value, std::memory_order_relaxed);
}

uint64_t MovingAverage::Tick() {
  std::lock_guard<std::mutex> guard(rate_mutex_);
  uint64_t count = count_.exchange(0, std::memory_order_relaxed);
  double current_rate = static_cast<double>(count) / tick_interval_sec_;
  if (rate_ < 0) {
    rate_ = current_rate;
  } else {
    rate_ += (current_rate - rate_) * alpha_;
  }
  return count;
}

double MovingAverage::rate() {
  std::lock_guard<std::mutex> guard(rate_mutex_);
  return rate_;
}

void MovingAverage::Reset() {
  std::lock_guard<std::mutex> guard(rate_mutex_);
  rate_ = -1;
  count_.store(0, std::memory_order_seq_cst);
}

} // namespace nexus
