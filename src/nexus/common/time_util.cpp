#include "nexus/common/time_util.h"

namespace nexus {

void Timer::Record(const std::string& tag) {
  time_points_.emplace(tag, Clock::now());
}

uint64_t Timer::GetLatencyMillis(const std::string& beg_tag,
                                 const std::string& end_tag) {
  auto beg = GetTimepoint(beg_tag);
  auto end = GetTimepoint(end_tag);
  if (beg == nullptr || end == nullptr) {
    return 0;
  }
  auto d = std::chrono::duration_cast<std::chrono::milliseconds>(*end - *beg);
  return d.count();
}

uint64_t Timer::GetLatencyMicros(const std::string& beg_tag,
                                 const std::string& end_tag) {
  auto beg = GetTimepoint(beg_tag);
  auto end = GetTimepoint(end_tag);
  if (beg == nullptr || end == nullptr) {
    return 0;
  }
  auto d = std::chrono::duration_cast<std::chrono::microseconds>(*end - *beg);
  return d.count();
}

TimePoint* Timer::GetTimepoint(const std::string& tag) {
  auto itr = time_points_.find(tag);
  if (itr == time_points_.end()) {
    return nullptr;
  }
  return &itr->second;
}

Tickable::Tickable(uint32_t tick_interval_sec) :
      tick_interval_sec_(tick_interval_sec),
      sec_since_last_tick_(0) {
  TimeSystem::Singleton().AddTickable(this);
}

Tickable::~Tickable() {
  TimeSystem::Singleton().RemoveTickable(this);
}

void Tickable::Tick() {
  ++sec_since_last_tick_;
  if (sec_since_last_tick_ == tick_interval_sec_) {
    TickImpl();
    sec_since_last_tick_ = 0;
  }
}

TimeSystem& TimeSystem::Singleton() {
  static TimeSystem time_system_;
  return time_system_;
}

TimeSystem::TimeSystem() :
    running_(true) {
  thread_ = std::thread(&TimeSystem::Run, this);
}

void TimeSystem::Stop() {
  running_ = false;
  thread_.join();
}

bool TimeSystem::AddTickable(Tickable* tickable) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (tickables_.find(tickable) != tickables_.end()) {
    return false;
  }
  tickables_.insert(tickable);
  return true;
}

bool TimeSystem::RemoveTickable(Tickable* tickable) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto iter = tickables_.find(tickable);
  if (iter == tickables_.end()) {
    return false;
  }
  tickables_.erase(iter);
  return true;
}

void TimeSystem::Run() {
  while (running_) {
    auto next_time = Clock::now() + std::chrono::seconds(1);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      for (auto item : tickables_) {
        item->Tick();
      }
    }
    std::this_thread::sleep_until(next_time);
  }
}

} // namespace nexus
