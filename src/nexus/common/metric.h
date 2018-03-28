#ifndef NEXUS_COMMON_METRIC_H_
#define NEXUS_COMMON_METRIC_H_

#include <atomic>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace nexus {

class Metric {
 public:
  virtual std::string name() const = 0;

  virtual void Reset() = 0;
};

class Counter : public Metric {
 public:
  Counter(std::string name);

  std::string name() const override { return name_; }

  void Increase(uint64_t value);

  void Reset() final;
  
 private:
  std::string name_;
  std::atomic<uint64_t> count_;
};

class Meter : public Metric {
 public:
  Meter(std::string name, uint32_t tick_interval_sec);

  ~Meter();

  std::string name() const override { return name_; }

  void Increase(uint64_t value);

  void Reset() override;

  std::vector<uint64_t> GetHistory();

 private:
  void Tick();

 private:
  std::string name_;
  uint32_t tick_interval_sec_;
  std::atomic<uint64_t> count_;
  std::vector<uint64_t> history_;
  std::mutex history_mutex_;
  std::atomic_bool running_;
  std::thread tick_thread_;
};

class MovingAverage : public Metric {
 public:
  MovingAverage(std::string name, uint32_t tick_interval_sec,
                uint32_t avg_interval_sec);

  std::string name() const { return name_; }

  void Increase(uint64_t value);

  double rate();

  uint64_t Tick();

  void Reset() final;

 private:
  std::string name_;
  uint32_t tick_interval_sec_;
  uint32_t avg_interval_sec_;
  double alpha_;
  double rate_;
  std::mutex rate_mutex_;
  // counter for current epoch
  std::atomic<uint64_t> count_;
};

class MetricRegistry {
 public:
  static MetricRegistry& Singleton() {
    static MetricRegistry metric_registry;
    return metric_registry;
  }

  std::shared_ptr<MovingAverage> CreateMovingAverage(
      std::string name, uint32_t tick_interval_sec, uint32_t avg_interval_sec) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto metric = std::make_shared<MovingAverage>(name, tick_interval_sec,
                                                  avg_interval_sec);
    metrics_.push_back(metric);
    return metric;
  }
      
 private:
  MetricRegistry() {}

  //~MetricRegistry();
  
  std::mutex mutex_;
  std::vector<std::shared_ptr<Metric> > metrics_;
};

} // namespace nexus

#endif // NEXUS_COMMON_METRIC_H_
