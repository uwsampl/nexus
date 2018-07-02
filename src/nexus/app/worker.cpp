#include "nexus/app/frontend.h"
#include "nexus/app/worker.h"

namespace nexus {
namespace app {

Worker::Worker(QueryProcessor* qp, RequestPool& req_pool) :
    qp_(qp),
    req_pool_(req_pool),
    running_(false) {
}

void Worker::Start() {
  running_ = true;
  thread_ = std::thread(&Worker::Run, this);
}

void Worker::Stop() {
  running_ = false;
}

void Worker::Join() {
  if (thread_.joinable()) {
    thread_.join();
  }
}

void Worker::Run() {
  auto timeout = std::chrono::milliseconds(50);
  while (running_) {
    auto req = req_pool_.GetRequest(timeout);
    if (req == nullptr) {
      continue;
    }
    qp_->Process(req);
  }
}

} // namespace app
} // namespace nexus
