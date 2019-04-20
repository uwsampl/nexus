#include <cmath>
#include <mutex>
#include <utility>
#include <glog/logging.h>

#include "nexus/common/model_db.h"
#include "nexus/scheduler/complex_query.h"

namespace nexus {
namespace scheduler {

class ComplexQuery::Impl {
 public:
  Impl(std::string cq_id, int slo_us, int segments);
  void AddNode(NodeID node_id, std::string current_model_sess_id,
               const ModelProfile& profile);
  void AddChild(const NodeID &parent, const NodeID &child);
  void SetRequestRate(const NodeID &node_id, double request_rate);
  double GetMinimalGPUs();
  void DynamicProgramming();
  void Finalize();
  bool IsFinalized();

 private:
  struct ThroughputEntry {
    double max_throughput;
    uint32_t batch_size;
    ThroughputEntry(double t, uint32_t b): max_throughput(t), batch_size(b) {}
  };

  struct DPEntry {
    double min_gpu;
    int node_time;
  };

  struct NodeInfo {
    // node information
    NodeID node_id;
    std::string current_model_sess_id;
    double request_rate;

    // dependency graph
    NodeInfo* parent;
    std::unordered_set<NodeInfo*> children;

    // dynamic programming
    std::vector<ThroughputEntry> max_throughput;  // max_throughput[i] for time budget step_*i
    std::vector<DPEntry> dp;  // dynamic programming book keeping
    int global_time_budget; // used to calculate slo_ms recursively
    uint32_t slo_ms;  // the result of dynamic programming
  };

  std::string cq_id_;
  int slo_us_;
  int segments_;
  double step_;
  std::unordered_map<NodeID, NodeInfo> nodes_;
  double minimal_gpus_;
  NodeInfo* root_;
  std::vector<NodeInfo*> tree_order_;  // topological order
  std::mutex mutex_;
};

ComplexQuery::Impl::Impl(std::string cq_id, int slo_us, int segments) :
    cq_id_(std::move(cq_id)), slo_us_(slo_us), segments_(segments),
    step_(static_cast<double>(slo_us) / segments), minimal_gpus_(0), root_(nullptr)
{
}

void ComplexQuery::Impl::AddNode(NodeID node_id, std::string current_model_sess_id,
                                 const ModelProfile& profile) {
  std::lock_guard<std::mutex> lock(mutex_);
  CHECK(root_ != nullptr) << "Already finalized";
  CHECK(nodes_.count(node_id) == 0) << "Node " << node_id.ToString() << " already exists.";

  NodeInfo node_info = {
      .node_id = std::move(node_id),
      .current_model_sess_id = std::move(current_model_sess_id),
      .request_rate = 0,
      .parent = nullptr,
  };
  node_info.max_throughput.emplace_back(0, 0);
  for (int j = 1; j <= segments_; ++j) {
    auto res = profile.GetMaxThroughput(step_ * 1e3 * j);
    node_info.max_throughput.emplace_back(res.second, res.first);
  }

  nodes_.emplace(node_id, std::move(node_info));
}

void ComplexQuery::Impl::AddChild(const NodeID &parent, const NodeID &child) {
  std::lock_guard<std::mutex> lock(mutex_);
  CHECK(!IsFinalized()) << "Already finalized";
  auto &p = nodes_.at(parent);
  auto &c = nodes_.at(child);
  CHECK(c.parent == nullptr);
  c.parent = &p;
  p.children.insert(&c);
}

void ComplexQuery::Impl::SetRequestRate(const NodeID &node_id, double request_rate) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto &node = nodes_.at(node_id);
  node.request_rate = request_rate;
}

double ComplexQuery::Impl::GetMinimalGPUs() {
  std::lock_guard<std::mutex> lock(mutex_);
  CHECK(IsFinalized()) << "Not finalized yet";
  return minimal_gpus_;
}

void ComplexQuery::Impl::DynamicProgramming() {
  std::lock_guard<std::mutex> lock(mutex_);
  CHECK(IsFinalized()) << "Not finalized yet";
  for (auto it = tree_order_.rbegin(); it != tree_order_.rend(); ++it) {
    auto &node = *it;
    node->dp[0] = DPEntry{.min_gpu = 1e10,
                          .node_time = 0};
    for (int time_budget = 1; time_budget <= segments_; ++time_budget) {
      node->dp[time_budget] = DPEntry{.min_gpu = node->dp[time_budget - 1].min_gpu,
                                      .node_time = time_budget - 1};
      for (int node_time = 1; node_time < time_budget; ++node_time) {
        double cost = node->request_rate / node->max_throughput[node_time].max_throughput;
        for (auto &child : node->children)
          cost += child->dp[time_budget - node_time].min_gpu;
        if (cost < node->dp[time_budget].min_gpu) {
          node->dp[time_budget] = DPEntry{.min_gpu = cost,
                                          .node_time = node_time};
        }
      }
    }
  }

  root_->global_time_budget = segments_;
  for (auto &node : tree_order_) {
    const double slo_ms = std::round(node->global_time_budget * step_ / 1e3);
    CHECK_GT(1e-3, slo_ms) << "Invalid slo_ms";
    CHECK_GT(slo_ms, slo_us_ / 1e3) << "Invalid slo_ms";
    node->slo_ms = static_cast<uint32_t>(slo_ms);
    CHECK_GT(0, node->slo_ms) << "Invalid slo_ms";
    CHECK_GT(node->slo_ms, slo_us_ / 1000) << "Invalid slo_ms";
    const int node_time = node->dp[node->global_time_budget].node_time;
    CHECK_NE(node_time, 0);
    const int child_time = node->global_time_budget - node_time;
    for (auto &child : node->children) {
      child->global_time_budget = child_time;
    }
  }
  minimal_gpus_ = root_->dp[segments_].min_gpu;
}

void ComplexQuery::Impl::Finalize() {
  std::lock_guard<std::mutex> lock(mutex_);
  CHECK(root_ == nullptr) << "Already finalized";
  for (auto &node : nodes_) {
    if (node.second.parent == nullptr) {
      CHECK(root_ == nullptr) << "Found multiple root";
      root_ = &node.second;
    }
    node.second.dp.assign(segments_ + 1, {});
  }

  std::unordered_set<NodeInfo*> visited;
  tree_order_.push_back(root_);
  visited.insert(root_);
  for (size_t head = 0; head < tree_order_.size(); ++head) {
    auto *node = tree_order_[head];
    for (auto &child : node->children) {
      CHECK(!visited.count(child)) << "Not a tree";
      visited.insert(child);
      tree_order_.push_back(child);
    }
  }
  CHECK_EQ(visited.size(), nodes_.size()) << "Some nodes are not reachable";
}

bool ComplexQuery::Impl::IsFinalized() {
  return root_ != nullptr;
}

ComplexQuery::ComplexQuery(std::string cq_id, int slo_us, int segments) :
  impl_(new Impl(std::move(cq_id), slo_us, segments))
{
}

ComplexQuery::~ComplexQuery() = default;
ComplexQuery::ComplexQuery(ComplexQuery &&other) noexcept = default;
ComplexQuery& ComplexQuery::operator=(ComplexQuery &&other) noexcept = default;

void ComplexQuery::AddNode(NodeID node_id, std::string current_model_sess_id,
                           const ModelProfile& profile) {
  impl_->AddNode(std::move(node_id), std::move(current_model_sess_id), profile);
}

void ComplexQuery::AddChild(const NodeID &parent, const NodeID &child) {
  impl_->AddChild(parent, child);
}

void ComplexQuery::SetRequestRate(const NodeID &node_id, double request_rate) {
  impl_->SetRequestRate(node_id, request_rate);
}

double ComplexQuery::GetMinimalGPUs() {
  return impl_->GetMinimalGPUs();
}

void ComplexQuery::DynamicProgramming() {
  impl_->DynamicProgramming();
}

void ComplexQuery::Finalize() {
  impl_->Finalize();
}

bool ComplexQuery::IsFinalized() {
  return impl_->IsFinalized();
}

std::string ComplexQuery::NodeID::ToString() const {
  return framework + ':' + model_name;
}

} // namespace scheduler
} // namespace nexus
