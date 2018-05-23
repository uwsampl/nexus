#ifndef NEXUS_SCHEDULER_SCH_INFO_H_
#define NEXUS_SCHEDULER_SCH_INFO_H_

#include <deque>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace nexus {
namespace scheduler {

using SessionGroup = std::vector<ModelSession>;

inline void RemoveFromSessionGroup(SessionGroup* sessions,
                                   const std::string model_session_id) {
  for (auto iter = sessions->begin(); iter != sessions->end(); ++iter) {
    if (ModelSessionToString(*iter) == model_session_id) {
      sessions->erase(iter);
      return;
    }
  }
}

struct SessionInfo {
  SessionGroup model_sessions;
  /*! \brief Mapping from backend id to throughput */
  std::unordered_map<uint32_t, double> backend_throughputs;
  /*! \brief Workload request rate history */
  std::deque<double> rps_history;
  /*! \brief Gap between workload and throughput */
  float unassigned_workload;

  SessionInfo() : unassigned_workload(0) {}

  double total_throughput() const {
    double total = 0.;
    for (auto iter : backend_throughputs) {
      total += iter.second;
    }
    return total;
  }
};

struct InstanceInfo {
  SessionGroup model_sessions;
  uint32_t batch;
  uint32_t max_batch;
  const ModelProfile* profile;
  float fwd_latency_us;
  float max_duty_cycle_us;
  float throughput;
  uint64_t memory_usage;

  InstanceInfo() {}
  
  InstanceInfo(const InstanceInfo& other) :
      model_sessions(other.model_sessions),
      batch(other.batch),
      max_batch(other.max_batch),
      profile(other.profile),
      fwd_latency_us(other.fwd_latency_us),
      max_duty_cycle_us(other.max_duty_cycle_us),
      throughput(other.throughput),
      memory_usage(other.memory_usage) {}
  
  InstanceInfo& operator=(const InstanceInfo& other) {
    if (this != &other) {
      model_sessions = other.model_sessions;
      batch = other.batch;
      max_batch = other.max_batch;
      profile = other.profile;
      fwd_latency_us = other.fwd_latency_us;
      max_duty_cycle_us = other.max_duty_cycle_us;
      throughput = other.throughput;
      memory_usage = other.memory_usage;
    }
    return *this;
  }
};

} // namespace scheduler
} // namespace nexus

#endif // NEXUS_SCHEDULE_SCH_INFO_H_
