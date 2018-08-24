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
  std::unordered_map<uint32_t, double> backend_weights;
  /*! \brief Workload request rate history */
  std::deque<double> rps_history;
  /*! \brief Gap between workload and throughput */
  double unassigned_workload;
  /*! \brief Whether there is a static workload for this session */
  bool has_static_workload;

  std::unordered_set<uint32_t> backup_backends;

  SessionInfo() :
      unassigned_workload(0),
      has_static_workload(false) {}

  double total_throughput() const {
    double total = 0.;
    for (auto iter : backend_weights) {
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
  double fwd_latency_us;
  double max_duty_cycle_us;
  double throughput;
  double weight;
  uint64_t memory_usage;
  bool backup;
  std::unordered_map<uint32_t, BackendInfo> backup_backends;

  InstanceInfo() :
      batch(0),
      max_batch(0),
      profile(nullptr),
      fwd_latency_us(0.),
      max_duty_cycle_us(0.),
      throughput(0.),
      weight(0.),
      memory_usage(0),
      backup(false) {}
  
  InstanceInfo(const InstanceInfo& other) :
      model_sessions(other.model_sessions),
      batch(other.batch),
      max_batch(other.max_batch),
      profile(other.profile),
      fwd_latency_us(other.fwd_latency_us),
      max_duty_cycle_us(other.max_duty_cycle_us),
      throughput(other.throughput),
      weight(other.weight),
      memory_usage(other.memory_usage),
      backup(other.backup) {}
  
  InstanceInfo& operator=(const InstanceInfo& other) {
    if (this != &other) {
      model_sessions = other.model_sessions;
      batch = other.batch;
      max_batch = other.max_batch;
      profile = other.profile;
      fwd_latency_us = other.fwd_latency_us;
      max_duty_cycle_us = other.max_duty_cycle_us;
      throughput = other.throughput;
      weight = other.weight;
      memory_usage = other.memory_usage;
      backup = other.backup;
    }
    return *this;
  }

  double GetWeight() const {
    return (weight > 0) ? weight : throughput;
  }
};

} // namespace scheduler
} // namespace nexus

#endif // NEXUS_SCHEDULE_SCH_INFO_H_
