#include <glog/logging.h>
#include <typeinfo>

#include "common/model_def.h"
#include "common/model_handler.h"

namespace nexus {

Output::Output(uint32_t timeout_ms) :
    ready_(false),
    timeout_(timeout_ms) {
}

uint32_t Output::status() {
  WaitForReadyOrTimeout();
  return status_;
}

std::string Output::error_message() {
  WaitForReadyOrTimeout();
  return error_message_;
}

void Output::FillReply(ReplyProto* reply) {
  WaitForReadyOrTimeout();
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

const Record& Output::operator[](uint32_t idx) {
  WaitForReadyOrTimeout();
  return records_.at(idx);
}

void Output::SetResult(const QueryResultProto& result) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
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
  cv_.notify_all();
}

void Output::SetResult(uint32_t status, const std::string& error_msg) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    status_ = status;
    error_message_ = error_msg;
    ready_ = true;
  }
  cv_.notify_all();
}

void Output::WaitForReadyOrTimeout() {
  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait_for(lock, timeout_, [this](){ return ready_; });
  if (!ready_) {
    status_ = TIMEOUT;
    error_message_ = "Timeout";
  }
}

ModelHandler::ModelHandler(const ModelSession& model_session,
                           BackendPool& pool) :
    model_session_(model_session),
    backend_pool_(pool),
    query_id_(0),
    rand_gen_(rd_()) {
  model_session_id_ = ModelSessionToString(model_session);
}

std::shared_ptr<Output> ModelHandler::Execute(
    const ValueProto& input, std::vector<std::string> output_fields,
    uint32_t topk, std::vector<RectProto> windows) {
  auto output = std::make_shared<Output>(model_session_.latency_sla());
  auto backend = GetBackend();
  if (backend == nullptr) {
    output->SetResult(SERVICE_UNAVAILABLE, "Service unavailable");
    return output;
  }
  uint32_t qid = query_id_.fetch_add(1);
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
  
  auto msg = std::make_shared<Message>(kBackendRequest, query.ByteSizeLong());
  msg->EncodeBody(query);
  backend->Write(std::move(msg));
  {
    std::lock_guard<std::mutex> lock(outputs_mu_);
    outputs_.emplace(qid, output);
  }
  return output;
}

void ModelHandler::HandleResult(const QueryResultProto& result) {
  uint32_t qid = result.query_id();
  std::shared_ptr<Output> output;
  {
    std::lock_guard<std::mutex> lock(outputs_mu_);
    auto iter = outputs_.find(qid);
    if (iter == outputs_.end()) {
      LOG(ERROR) << "Cannot find output for query " << qid;
      return;
    }
    output = iter->second;
    outputs_.erase(iter);
  }
  output->SetResult(result);
}

void ModelHandler::UpdateRoute(const ModelRoute& route) {
  std::lock_guard<std::mutex> lock(route_mu_);
  backend_rates_.clear();
  for (auto itr : route.backend_rate()) {
    backend_rates_.emplace_back(itr.node_id(), itr.rate());
  }
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

} // namespace nexu
