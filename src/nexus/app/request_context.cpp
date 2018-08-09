#include "nexus/app/exec_block.h"
#include "nexus/app/request_context.h"

namespace nexus {
namespace app {

RequestContext::RequestContext(std::shared_ptr<UserSession> user_sess,
                               std::shared_ptr<Message> msg,
                               RequestPool& req_pool) :
    user_session_(user_sess),
    req_pool_(req_pool),
    state_(kUninitialized) {
  beg_ = Clock::now();
  msg->DecodeBody(&request_);
}

bool RequestContext::finished() {
  std::lock_guard<std::mutex> lock(mu_);
  return (pending_blocks_.empty() && ready_blocks_.empty());
}

void RequestContext::SetState(RequestState state) {
  if (state_ == kError) {
    return;
  }
  RequestState prev_state = state_.exchange(state);
  if (prev_state == state) {
    return;
  }
  if (prev_state == kBlocking) {
    if (state == kRunning || state == kError) {
      req_pool_.MoveToReady(shared_from_this());
    }
  } else if (prev_state == kRunning) {
    if (state == kBlocking) {
      req_pool_.AddBlockRequest(shared_from_this());
    }
  }
}

void RequestContext::SetExecBlocks(std::vector<ExecBlock*> blocks) {
  CHECK_EQ(state_, kUninitialized) << "Request context is alrealdy initialized";
  for (auto block : blocks) {
    auto deps = block->dependency();
    if (deps.empty()) {
      ready_blocks_.push_back(block);
    } else {
      pending_blocks_.emplace(block->id(), block);
      block_deps_.emplace(block->id(), block->dependency());
    }
  }
  state_.store(kRunning);
}

ExecBlock* RequestContext::NextReadyBlock() {
  std::lock_guard<std::mutex> lock(mu_);
  if (ready_blocks_.empty()) {
    return nullptr;
  }
  auto block = ready_blocks_.front();
  ready_blocks_.pop_front();
  // LOG(INFO) << "Ready blocks: " << ready_blocks_.size() <<
  //     ", pending blocks: " << pending_blocks_.size();
  return block;
}

VariablePtr RequestContext::GetVariable(const std::string& var_name) {
  std::lock_guard<std::mutex> lock(mu_);
  auto itr = vars_.find(var_name);
  CHECK(itr != vars_.end()) << "Variable " << var_name << " doesn't exist " <<
      " or is not ready";
  return itr->second;
}

void RequestContext::AddBlockReturn(std::vector<VariablePtr> vars) {
  std::lock_guard<std::mutex> lock(mu_);
  for (auto var : vars) {
    auto var_name = var->name();
    for (auto qid : var->query_ids()) {
      auto itr = dangling_results_.find(qid);
      if (itr != dangling_results_.end()) {
        var->AddQueryResult(itr->second);
        dangling_results_.erase(itr);
      } else {
        qid_var_map_.emplace(qid, var_name);
      }
    }
    if (var->ready()) {
      AddReadyVariable(var);
    } else {
      waiting_vars_.emplace(var_name, var);
    }
  }
}

void RequestContext::HandleQueryResult(const QueryResultProto& result) {
  if (state_ == kError) {
    return;
  }
  if (result.status() != CTRL_OK) {
    // LOG(INFO) << request_.user_id() << ":" << request_.req_id() << ":" <<
    //     result.query_id() << " error: " << result.status();
    HandleError(result.status(), result.error_message());
    return;
  }
  std::lock_guard<std::mutex> lock(mu_);
  uint64_t qid = result.query_id();
  auto qid_itr = qid_var_map_.find(qid);
  if (qid_itr == qid_var_map_.end()) {
    dangling_results_.emplace(qid, result);
    return;
  }
  std::string var_name = qid_itr->second;
  qid_var_map_.erase(qid_itr);
  
  auto var = waiting_vars_.at(var_name);
  if (var->AddQueryResult(result)) {
    waiting_vars_.erase(var_name);
    AddReadyVariable(var);
  }
}

void RequestContext::HandleError(uint32_t status,
                                 const std::string& error_msg) {
  std::lock_guard<std::mutex> lock(mu_);
  reply_.set_status(status);
  reply_.set_error_message(error_msg);
  ready_blocks_.clear();
  pending_blocks_.clear();
  SetState(kError);
}

void RequestContext::SendReply() {
  reply_.set_user_id(request_.user_id());
  reply_.set_req_id(request_.req_id());
  auto end = Clock::now();
  auto latency = std::chrono::duration_cast<std::chrono::microseconds>(
      end - beg_).count();
  reply_.set_latency_us(latency);
  auto reply_msg = std::make_shared<Message>(kUserReply,
                                             reply_.ByteSizeLong());
  reply_msg->EncodeBody(reply_);
  user_session_->Write(std::move(reply_msg));
}

void RequestContext::AddReadyVariable(std::shared_ptr<Variable> var) {
  vars_.emplace(var->name(), var);
  std::vector<int> ready_blocks;
  for (auto& block_itr : block_deps_) {
    block_itr.second.erase(var->name());
    if (block_itr.second.empty()) {
      ready_blocks.push_back(block_itr.first);
    }
  }
  for (auto block_id : ready_blocks) {
    auto block = pending_blocks_.at(block_id);
    pending_blocks_.erase(block_id);
    block_deps_.erase(block_id);
    ready_blocks_.push_back(block);
  }
  if (!ready_blocks_.empty()) {
    SetState(kRunning);
  }
}

} // namespace app
} // namespace nexus