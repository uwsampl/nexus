#include <glog/logging.h>
#include <typeinfo>

#include "nexus/app/model_handler.h"
#include "nexus/app/request_context.h"
#include "nexus/common/model_def.h"

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
    total_throughput_(0.),
    rand_gen_(rd_()) {
  ParseModelSession(model_session_id, &model_session_);
}

std::shared_ptr<QueryResult> ModelHandler::Execute(
    std::shared_ptr<RequestContext> ctx, const ValueProto& input,
    std::vector<std::string> output_fields, uint32_t topk,
    std::vector<RectProto> windows) {
  uint64_t qid = global_query_id_.fetch_add(1, std::memory_order_relaxed);
  auto reply = std::make_shared<QueryResult>(qid);
  auto backend = GetBackend();
  if (backend == nullptr) {
    //reply->SetError(SERVICE_UNAVAILABLE, "Service unavailable");
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
  auto ctx = query_ctx_.at(qid);
  ctx->HandleQueryResult(result);
  query_ctx_.erase(qid);
}

void ModelHandler::UpdateRoute(const ModelRouteProto& route) {
  std::lock_guard<std::mutex> lock(route_mu_);
  backend_rates_.clear();
  total_throughput_ = 0.;
  LOG(INFO) << "Update model route for " << model_session_id_;
  
  for (auto itr : route.backend_rate()) {
    backend_rates_.emplace_back(itr.info().node_id(), itr.throughput());
    total_throughput_ += itr.throughput();
    LOG(INFO) << "- backend " << itr.info().node_id() << ": " <<
        itr.throughput();
  }
  LOG(INFO) << "Total throughput: " << total_throughput_;
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
  std::lock_guard<std::mutex> lock(route_mu_);
  std::uniform_real_distribution<float> dis(0, total_throughput_);
  float select = dis(rand_gen_);
  uint i = 0;
  for (; i < backend_rates_.size(); ++i) {
    float rate = backend_rates_[i].second;
    select -= rate;
    if (select < 0) {
      auto backend_sess = backend_pool_.GetBackend(backend_rates_[i].first);
      if (backend_sess != nullptr) {
        return backend_sess;
      }
      break;
    }
  }
  ++i;
  for (uint j = 0; j < backend_rates_.size(); ++j, ++i) {
    auto backend_sess = backend_pool_.GetBackend(backend_rates_[i].first);
    if (backend_sess != nullptr) {
      return backend_sess;
    }
  }
  return nullptr;
}

// void ModelHandler::RemoveOutput(uint64_t qid) {
//   std::lock_guard<std::mutex> lock(rpc_replies_mu_);
//   auto iter = rpc_replies_.find(qid);
//   if (iter == rpc_replies_.end()) {
//     return;
//   }
//   rpc_replies_.erase(iter);
// }

} // namespace app
} // namespace nexus
