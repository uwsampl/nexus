#ifndef NEXUS_SCHEDULER_COMPLEXQUERY_H
#define NEXUS_SCHEDULER_COMPLEXQUERY_H

#include <string>
#include <memory>
#include <unordered_map>

#include "nexus/scheduler/sch_info.h"

namespace nexus {
namespace scheduler {

class ComplexQuery {
 public:
  struct NodeID {
    std::string framework;
    std::string model_name;
    std::string ToString() const;
  };
  ComplexQuery(std::string cq_id, int slo_us, int segments);
  ~ComplexQuery();
  ComplexQuery(ComplexQuery &&other) noexcept;
  ComplexQuery& operator=(ComplexQuery &&other) noexcept;

  void AddNode(NodeID node_id, std::string current_model_sess_id,
               const ModelProfile& profile);
  void AddChild(const NodeID &parent, const NodeID &child);
  void SetRequestRate(const NodeID &node_id, double request_rate);
  double GetMinimalGPUs();
  void DynamicProgramming();
  void Finalize();
  bool IsFinalized();

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

inline bool operator==(const ComplexQuery::NodeID& lhs, const ComplexQuery::NodeID& rhs) {
  return lhs.framework == rhs.framework && lhs.model_name == rhs.model_name;
}

} // namespace scheduler
} // namespace nexus

namespace std {
template<> struct hash<nexus::scheduler::ComplexQuery::NodeID> {
  std::size_t operator()(const nexus::scheduler::ComplexQuery::NodeID &v) const {
    size_t h = std::hash<std::string>{}(v.framework);
    h = h * 31 + std::hash<std::string>{}(v.model_name);
    return h;
  }
};
}

#endif //NEXUS_SCHEDULER_COMPLEXQUERY_H
