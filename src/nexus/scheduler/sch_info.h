#ifndef NEXUS_SCHEDULER_SCH_INFO_H_
#define NEXUS_SCHEDULER_SCH_INFO_H_

#include <deque>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <memory>
#include <gflags/gflags.h>
#include "nexus/common/metric.h"
#include "nexus/common/model_db.h"
#include "nexus/common/model_def.h"
#include "nexus/proto/nnquery.pb.h"
#include "nexus/proto/control.pb.h"

DECLARE_int32(avg_interval);

namespace nexus {
namespace scheduler {

using SessionGroup = std::vector<ModelSession>;
using ServerList = std::unordered_set<uint32_t>;

struct SessionInfo {
  SessionInfo() :
      has_static_workload(false),
      unassigned_workload(0) {}

  double TotalThroughput() const;

  void SubscribeModelSession(uint32_t frontend_id,
                             const std::string& model_sess_id);

  bool UnsubscribleModelSession(uint32_t frontend_id, const std::string& model_sess_id);

  void UpdateWorkload(uint32_t frontend_id, const ModelStatsProto& model_stats);

  SessionGroup model_sessions;
  /*! \brief Mapping from backend id to throughput */
  std::unordered_map<uint32_t, double> backend_weights;

  std::unordered_set<uint32_t> backup_backends;
  /*! \brief Whether there is a static workload for this session */
  bool has_static_workload;

  std::unordered_map<std::string, ServerList> session_subscribers;
  /*! \brief Map from frontend id to workload */
  std::unordered_map<uint32_t, std::shared_ptr<EWMA> > workloads;
  /*! \brief Workload request rate history */
  std::deque<double> rps_history;
  /*! \brief Gap between workload and throughput */
  double unassigned_workload;
};

struct InstanceInfo {
  SessionGroup model_sessions;
  uint32_t batch;
  uint32_t max_batch;
  const ModelProfile* profile;
  double fwd_latency_us;
  double max_duty_cycle_us;
  double workload;
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
      workload(0.),
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
      workload(other.workload),
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
      workload = other.workload;
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
