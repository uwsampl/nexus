#include "nexus/scheduler/sch_info.h"
#include <glog/logging.h>

namespace nexus {
namespace scheduler {

void SessionInfo::UpdateWorkload(uint32_t frontend_id, const ModelStatsProto &model_stats) {
  auto iter = workloads.find(frontend_id);
  if (iter == workloads.end()) {
    LOG(ERROR) << "Cannot find rps for " << frontend_id << " in " <<
               model_stats.model_session_id();
    return;
  }
  auto rps = iter->second;
  for (auto num_requests : model_stats.num_requests()) {
    if (rps->rate() < 0 && num_requests == 0) {
      continue;
    }
    rps->AddSample(num_requests);
  }
}
double SessionInfo::TotalThroughput() const {
  double total = 0.;
  for (auto iter : backend_weights) {
    total += iter.second;
  }
  return total;
}
void SessionInfo::SubscribeModelSession(uint32_t frontend_id, const std::string &model_sess_id) {
  if (session_subscribers.count(model_sess_id) == 0) {
    session_subscribers.emplace(model_sess_id, ServerList{frontend_id});
  } else {
    session_subscribers.at(model_sess_id).insert(frontend_id);
  }
  workloads.emplace(frontend_id,
                    std::make_shared<EWMA>(1, FLAGS_avg_interval));
}
bool SessionInfo::UnsubscribleModelSession(uint32_t frontend_id, const std::string &model_sess_id) {
  session_subscribers.at(model_sess_id).erase(frontend_id);
  workloads.erase(frontend_id);
  if (has_static_workload || !session_subscribers.at(model_sess_id).empty()) {
    return false;
  }
  // Remove this model session
  session_subscribers.erase(model_sess_id);
  for (auto iter = model_sessions.begin(); iter != model_sessions.end();
       ++iter) {
    if (ModelSessionToString(*iter) == model_sess_id) {
      model_sessions.erase(iter);
      break;
    }
  }
  return true;
}
}
}