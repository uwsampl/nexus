#include <glog/logging.h>

#include "nexus/backend/backup_client.h"

namespace nexus {
namespace backend {

BackupClient::BackupClient(const BackendInfo& info,
                           boost::asio::io_context& io_context,
                           MessageHandler* handler) :
    BackendSession(info, io_context, handler) {}

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
  // LOG(INFO) << "Convert " << result.model_session_id() << " tid " << tid <<
  //     " to qid " << qid;
  auto reply_msg = std::make_shared<Message>(kBackendReply,
                                             result.ByteSizeLong());
  reply_msg->EncodeBody(result);
  auto conn_iter = conns_.find(tid);
  conn_iter->second->Write(std::move(reply_msg));
  qid_lookup_.erase(qid_iter);
  conns_.erase(conn_iter);
}

} // namespace backend
} // namespace nexus
