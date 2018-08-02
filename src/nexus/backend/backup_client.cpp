#include <glog/logging.h>

#include "nexus/backend/backup_client.h"

namespace nexus {
namespace backend {

BackupClient::BackupClient(const BackendInfo& info,
                           boost::asio::io_service& io_service,
                           MessageHandler* handler) :
    BackendSession(info, io_service, handler),
    utilization_(-1.) {
  std::stringstream rpc_addr;
  rpc_addr << ip_ << ":" << rpc_port_;
  auto channel = grpc::CreateChannel(rpc_addr.str(),
                                     grpc::InsecureChannelCredentials());
  stub_ = BackendCtrl::NewStub(channel);
}

double BackupClient::GetUtilization() {
  std::lock_guard<std::mutex> lock(util_mu_);
  if (utilization_ >= 0 && Clock::now() <= expire_) {
    return utilization_;
  }
  UtilizationRequest request;
  UtilizationReply reply;
  request.set_node_id(node_id_);
  grpc::ClientContext ctx;
  grpc::Status status = stub_->CurrentUtilization(&ctx, request, &reply);
  if (!status.ok()) {
    LOG(ERROR) << status.error_code() << ": " << status.error_message();
    utilization_ = -1.;
    return -1.;
  }
  utilization_ = reply.utilization();
  expire_ = Clock::now() + std::chrono::milliseconds(reply.valid_ms());
  LOG(INFO) << "Backup " << node_id_ << " utilization " << utilization_;
  return utilization_;
}

void BackupClient::Forward(std::shared_ptr<Task> task) {
  uint64_t qid = task->query.query_id();
  task->query.set_query_id(task->task_id);
  auto msg = std::make_shared<Message>(kBackendRelay,
                                       task->query.ByteSizeLong());
  msg->EncodeBody(task->query);
  Write(std::move(msg));
  std::lock_guard<std::mutex> lock(relay_mu_);
  qid_lookup_.emplace(task->task_id, qid);
  conns_.emplace(task->task_id, task->connection);
}

void BackupClient::Reply(std::shared_ptr<Message> message) {
  QueryResultProto result;
  message->DecodeBody(&result);
  uint64_t tid = result.query_id();
  std::lock_guard<std::mutex> lock(relay_mu_);
  auto qid_iter = qid_lookup_.find(tid);
  if (qid_iter == qid_lookup_.end()) {
    LOG(ERROR) << "Cannot find query ID for task " << tid;
    return;
  }
  uint64_t qid = qid_iter->second;
  result.set_query_id(qid);
  message->EncodeBody(result);
  message->set_type(kBackendReply);
  auto conn_iter = conns_.find(tid);
  conn_iter->second->Write(std::move(message));
  qid_lookup_.erase(qid_iter);
  conns_.erase(conn_iter);
}

} // namespace backend
} // namespace nexus
