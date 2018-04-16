#ifndef NEXUS_COMMON_TIME_UTIL_H_
#define NEXUS_COMMON_TIME_UTIL_H_

#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>

namespace nexus {

using Clock = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::time_point<Clock>;

/*! \brief Timer helps to record time and count duration between two time
  points */
class Timer {
 public:
  /*!
   * \brief Records the time point with tag
   * \param tag Tag of time point
   */
  void Record(const std::string& tag);
  /*!
   * \brief Get the interval between two tags in millisecond
   * \param beg_tag Tag of begining time point
   * \param end_tag Tag of end time point
   * \return Duration in millisecond
   */
  uint64_t GetLatencyMillis(const std::string& beg_tag,
                            const std::string& end_tag);
  /*!
   * \brief Get the interval between two tags in microsecond
   * \param beg_tag Tag of begining time point
   * \param end_tag Tag of end time point
   * \return Duration in microsecond
   */
  uint64_t GetLatencyMicros(const std::string& beg_tag,
                            const std::string& end_tag);

 private:
  /*!
   * \brief Get the time point given the tag
   * \param tag Tag of time point
   * \return TimePoint pointer
   */
  TimePoint* GetTimepoint(const std::string& tag);
  /*! \brief Map from tag to time points */
  std::unordered_map<std::string, TimePoint> time_points_;
};

class Tickable {
 public:
  Tickable(uint32_t tick_interval_sec);

  virtual ~Tickable();

  void Tick();

 protected:
  virtual void TickImpl() = 0;

 protected:
  uint32_t tick_interval_sec_;
  uint32_t sec_since_last_tick_;
};

class TimeSystem {
 public:
  static TimeSystem& Singleton();

  void Stop();

  bool AddTickable(Tickable* tickable);

  bool RemoveTickable(Tickable* tickable);

 private:
  TimeSystem();

  void Run();

  std::unordered_set<Tickable*> tickables_;
  std::mutex mutex_;
  std::atomic_bool running_;
  std::thread thread_;
};

} // namespace nexus

#endif // NEXUS_COMMON_TIME_UTIL_H_
