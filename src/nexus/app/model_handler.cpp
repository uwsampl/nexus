#include <glog/logging.h>
#include <typeinfo>

#include "nexus/app/model_handler.h"
#include "nexus/app/request_context.h"
#include "nexus/common/model_def.h"

DEFINE_int32(count_interval, 1, "Interval to count number of requests in sec");
DEFINE_int32(load_balance, 1, "Load balance policy (1: random, 2: choice of 2, "
             "3: deficit round robin)");

namespace nexus {
namespace app {

QueryResult::QueryResult(uint64_t qid) :
    qid_(qid),
    ready_(false) {
}

uint32_t QueryResult::status() const {
  CheckReady();
  return status_;
}

std::string QueryResult::error_message() const {
  CheckReady();
  return error_message_;
}

void QueryResult::ToProto(ReplyProto* reply) const {
  CheckReady();
  reply->set_status(status_);
  if (status_ != CTRL_OK) {
    reply->set_error_message(error_message_);
  } else {
    for (auto record : records_) {
      auto rec_p = reply->add_output();
      record.ToProto(rec_p);
    }
  }
}

const Record& QueryResult::operator[](uint32_t idx) const {
  CheckReady();
  return records_.at(idx);
}

uint32_t QueryResult::num_records() const {
  CheckReady();
  return records_.size();
}

void QueryResult::CheckReady() const {
  CHECK(ready_) << "Rpc reply for query " << qid_ << " is not ready yet";
}

void QueryResult::SetResult(const QueryResultProto& result) {
  status_ = result.status();
  if (status_ != CTRL_OK) {
    error_message_ = result.error_message();
  } else {
    for (auto record : result.output()) {
      records_.emplace_back(record);
    }
  }
  ready_ = true;
}

void QueryResult::SetError(uint32_t status, const std::string& error_msg) {
  status_ = status;
  error_message_ = error_msg;
  ready_ = true;
}

std::atomic<uint64_t> ModelHandler::global_query_id_(0);

ModelHandler::ModelHandler(const std::string& model_session_id,
                           BackendPool& pool) :
    model_session_id_(model_session_id),
    backend_pool_(pool),
    lb_policy_(LoadBalancePolicy(FLAGS_load_balance)),
    total_throughput_(0.),
    rand_gen_(rd_()) {
  ParseModelSession(model_session_id, &model_session_);
  counter_ = MetricRegistry::Singleton().CreateIntervalCounter(
      FLAGS_count_interval);
  LOG(INFO) << "Load balance policy: " << lb_policy_;
  if (lb_policy_ == LB_DeficitRR) {
    running_ = true;
    deficit_thread_ = std::thread(&ModelHandler::DeficitDaemon, this);
  }
}

ModelHandler::~ModelHandler() {
  MetricRegistry::Singleton().RemoveMetric(counter_);
  if (deficit_thread_.joinable()) {
    running_ = false;
    deficit_thread_.join();
  }
}

std::shared_ptr<QueryResult> ModelHandler::Execute(
    std::shared_ptr<RequestContext> ctx, const ValueProto& input,
    std::vector<std::string> output_fields, uint32_t topk,
    std::vector<RectProto> windows) {
  uint64_t qid = global_query_id_.fetch_add(1, std::memory_order_relaxed);
  counter_->Increase(1);
  auto reply = std::make_shared<QueryResult>(qid);
  auto backend = GetBackend();
  if (backend == nullptr) {
    ctx->HandleError(SERVICE_UNAVAILABLE, "Service unavailable");
    return reply;
  }
  QueryProto query;
  query.set_query_id(qid);
  query.set_model_session_id(model_session_id_);
  query.mutable_input()->CopyFrom(input);
  for (auto field : output_fields) {
    query.add_output_field(field);
  }
  if (topk > 0) {
    query.set_topk(topk);
  }
  for (auto rect : windows) {
    query.add_window()->CopyFrom(rect);
  }
  if (ctx->slack_ms() > 0) {
    query.set_slack_ms(int(floor(ctx->slack_ms())));
  }
  ctx->RecordQuerySend(qid);
  {
    std::lock_guard<std::mutex> lock(query_ctx_mu_);
    query_ctx_.emplace(qid, ctx);
  }
  auto msg = std::make_shared<Message>(kBackendRequest, query.ByteSizeLong());
  msg->EncodeBody(query);
  backend->Write(std::move(msg));
  return reply;
}

void ModelHandler::HandleReply(const QueryResultProto& result) {
  std::lock_guard<std::mutex> lock(query_ctx_mu_);
  uint64_t qid = result.query_id();
  auto iter = query_ctx_.find(qid);
  if (iter == query_ctx_.end()) {
    LOG(FATAL) << model_session_id_ << " cannot find query context for query " <<
        qid;
    return;
  }
  auto ctx = iter->second;
  ctx->HandleQueryResult(result);
  query_ctx_.erase(qid);
}

void ModelHandler::UpdateRoute(const ModelRouteProto& route) {
  std::lock_guard<std::mutex> lock(route_mu_);
  backends_.clear();
  backend_rates_.clear();
  total_throughput_ = 0.;
  
  for (auto itr : route.backend_rate()) {
    uint32_t backend_id = itr.info().node_id();
    backends_.push_back(backend_id);
    backend_rates_.emplace(backend_id, itr.throughput());
    total_throughput_ += itr.throughput();
    LOG(INFO) << "- backend " << backend_id << ": " << itr.throughput();
    if (backend_quantums_.count(backend_id) == 0) {
      backend_quantums_.emplace(backend_id, 0.);
    }
  }
  LOG(INFO) << "Total throughput: " << total_throughput_;
  std::sort(backends_.begin(), backends_.end());
  for (auto iter = backend_quantums_.begin(); iter != backend_quantums_.end();) {
    if (backend_rates_.count(iter->first) == 0) {
      iter = backend_quantums_.erase(iter);
    } else {
      ++iter;
    }
  }
}

std::vector<uint32_t> ModelHandler::BackendList() {
  std::vector<uint32_t> ret;
  std::lock_guard<std::mutex> lock(route_mu_);
  for (auto iter : backend_rates_) {
    ret.push_back(iter.first);
  }
  return ret;
}

std::shared_ptr<BackendSession> ModelHandler::GetBackend() {
  switch (lb_policy_) {
    case LB_WeightedRR: {
      return GetBackendWeightedRoundRobin();
    }
    case LB_DeficitRR: {
      auto backend = GetBackendDeficitRoundRobin();
      if (backend != nullptr) {
        return backend;
      }
      return GetBackendWeightedRoundRobin();
    }
    case LB_Query: {
      auto candidate1 = GetBackendWeightedRoundRobin();
      if (candidate1 == nullptr) {
        return nullptr;
      }
      auto candidate2 = GetBackendWeightedRoundRobin();
      if (candidate1 == candidate2) {
        return candidate1;
      }
      if (candidate1->GetUtilization() <= candidate2->GetUtilization()) {
        return candidate1;
      }
      return candidate2;
    }
    default:
      return nullptr;
  }
}

std::shared_ptr<BackendSession> ModelHandler::GetBackendWeightedRoundRobin() {
  std::uniform_real_distribution<float> dis(0, total_throughput_);
  float select = dis(rand_gen_);
  uint i = 0;
  for (; i < backends_.size(); ++i) {
    uint32_t backend_id = backends_[i];
    float rate = backend_rates_.at(backend_id);
    select -= rate;
    if (select < 0) {
      auto backend_sess = backend_pool_.GetBackend(backend_id);
      if (backend_sess != nullptr) {
        return backend_sess;
      }
      break;
    }
  }
  ++i;
  for (uint j = 0; j < backends_.size(); ++j, ++i) {
    auto backend_sess = backend_pool_.GetBackend(backends_[i]);
    if (backend_sess != nullptr) {
      return backend_sess;
    }
  }
  return nullptr;
}

std::shared_ptr<BackendSession> ModelHandler::GetBackendDeficitRoundRobin() {
  std::lock_guard<std::mutex> lock(route_mu_);
  for (int i = 0; i < backends_.size(); ++i) {
    uint32_t idx = backend_idx_.fetch_add(1, std::memory_order_relaxed) %
                   backends_.size();
    uint32_t backend_id = backends_[idx];
    if (backend_quantums_.at(backend_id) >= 1) {
      auto backend = backend_pool_.GetBackend(backend_id);
      if (backend != nullptr) {
        --backend_quantums_[backend_id];
        return backend;
      }
    }
  }
  return nullptr;
}

void ModelHandler::DeficitDaemon() {
  std::chrono::milliseconds gap(200); // 200 ms
  std::unique_lock<std::mutex> lock(route_mu_, std::defer_lock);
  while (running_) {
    lock.lock();
    for (auto backend_id : backends_) {
      backend_quantums_[backend_id] = backend_rates_[backend_id] * .2;
    }
    lock.unlock();
    std::this_thread::sleep_for(gap);
  }
}

} // namespace app
} // namespace nexus
