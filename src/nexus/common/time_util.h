#ifndef NEXUS_COMMON_TIME_UTIL_H_
#define NEXUS_COMMON_TIME_UTIL_H_

#include <chrono>
#include <condition_variable>
#include <glog/logging.h>
#include <mutex>
#include <string>
#include <unordered_map>

namespace nexus {

using Clock = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::time_point<Clock>;

class Timer {
 public:
  void Record(const std::string& tag) {
    time_points_.emplace(tag, Clock::now());
  }

  uint64_t GetLatencyMillis(const std::string& beg_tag,
                            const std::string& end_tag) {
    auto beg = GetTimepoint(beg_tag);
    auto end = GetTimepoint(end_tag);
    if (beg == nullptr || end == nullptr) {
      return 0;
    }
    auto d = std::chrono::duration_cast<std::chrono::milliseconds>(*end - *beg);
    return d.count();
  }

  uint64_t GetLatencyMicros(const std::string& beg_tag,
                            const std::string& end_tag) {
    auto beg = GetTimepoint(beg_tag);
    auto end = GetTimepoint(end_tag);
    if (beg == nullptr || end == nullptr) {
      return 0;
    }
    auto d = std::chrono::duration_cast<std::chrono::microseconds>(*end - *beg);
    return d.count();
  }

 private:
  TimePoint* GetTimepoint(const std::string& tag) {
    auto itr = time_points_.find(tag);
    if (itr == time_points_.end()) {
      return nullptr;
    }
    return &itr->second;
  }
  
  std::unordered_map<std::string, TimePoint> time_points_;
};

/*
class ClockSystem {
 public:
  static ClockSystem& Singleton();

  void RegisterTicker(std::string name, uint32_t tick_interval_sec,
                      std::function<void(void)> tick_func);

  class Ticker {
    Ticker(std::string name, uint32_t tick_interval_sec,
           std::function<void(void)> tick_func) :
        name_(name),
        tick_interval_(tick_interval_sec),
        tick_func_(tick_func),
        valid_(true) {
      next_tick_time_ = Clock::now() + tick_interval_;
    }

    bool valid() const { return valid_; }

    TimePoint NextEvent() const { return next_tick_time_; }

    void Tick() {
      tick_func_();
      next_tick_time_ += tick_interval_;
    }

   private:
    std::string name_;
    std::chrono::seconds tick_interval_;
    std::fucntion<void(void)> tick_func_;
    TimePoint next_tick_time_;
    std::atomic_bool valid_;
  };

  class CompareTicker {
   public:
    bool operator()(std::shared_ptr<Ticker> lhs,
                    std::shared_ptr<Ticker> rhs) {
      return lhs->NextEvent() > rhs->NextEvent();
    }
  };


 private:
  ClockSystem();

  void Run();

  std::unordered_map<std::string, std::shared_ptr<Ticker> > tickers_;
  std::priority_queue<std::shared_ptr<Ticker>,
                      std::vector<std::shared_ptr<Ticker> >,
                      CompareTicker> tick_events_;
  std::mutex tick_mutex_;
  std::condition_variable not_empty_;
  std::atomic_bool running_;
  std::thread thread_;
  
}
*/
} // namespace nexus

#endif // NEXUS_COMMON_TIME_UTIL_H_
