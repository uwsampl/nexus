#include "app/frontend.h"
#include "app/worker.h"

namespace nexus {
namespace app {

Worker::Worker(Frontend* frontend, BlockQueue<Message>& req_queue) :
    frontend_(frontend),
    request_queue_(req_queue),
    running_(false) {
}

void Worker::Start() {
  running_ = true;
  thread_ = std::thread(&Worker::Run, this);
}

void Worker::Stop() {
  running_ = false;
  if (thread_.joinable()) {
    thread_.join();
  }
}

void Worker::Run() {
  LOG(INFO) << "Worker started";
  auto timeout = std::chrono::milliseconds(50);
  while (running_) {
    std::shared_ptr<Message> msg = request_queue_.pop(timeout);
    if (msg == nullptr) {
      continue;
    }
    RequestProto request;
    ReplyProto reply;
    msg->DecodeBody(&request);
    auto user_sess = frontend_->GetUserSession(request.user_id());
    if (user_sess == nullptr) {
      LOG(ERROR) << "No user session for " << request.user_id();
      continue;
    }
    reply.set_user_id(request.user_id());
    reply.set_req_id(request.req_id());
    frontend_->Process(request, &reply);
    auto reply_msg = std::make_shared<Message>(kUserReply,
                                               reply.ByteSizeLong());
    reply_msg->EncodeBody(reply);
    user_sess->Write(std::move(reply_msg));
  }
  LOG(INFO) << "Worker stopped";
}

} // namespace app
} // namespace nexus
