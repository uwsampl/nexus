#include <glog/logging.h>

#include "nexus/common/backend_pool.h"
#include "nexus/common/util.h"

namespace nexus {

BackendSession::BackendSession(const BackendInfo& info,
                               boost::asio::io_service& io_service,
                               MessageHandler* handler):
    Connection(io_service, handler),
    io_service_(io_service),
    node_id_(info.node_id()),
    ip_(info.ip()),
    server_port_(info.server_port()),
    rpc_port_(info.rpc_port()),
    running_(false) {}

BackendSession::~BackendSession() {
  Stop();
}

void BackendSession::Start() {
  // Connect to backend server
  DoConnect();
}

void BackendSession::Stop() {
  if (running_) {
    LOG(INFO) << "Disconnect to backend " << node_id_;
    running_ = false;
    Connection::Stop();
  }
}

void BackendSession::DoConnect() {
  boost::asio::ip::tcp::resolver::iterator endpoint;
  boost::asio::ip::tcp::resolver resolver(io_service_);
  endpoint = resolver.resolve({ ip_, server_port_ });
  boost::asio::async_connect(
      socket_, endpoint,
      [this](boost::system::error_code ec,
             boost::asio::ip::tcp::resolver::iterator) {
        if (ec) {
          handler_->HandleError(shared_from_this(), ec);
        } else {
          running_ = true;
          LOG(INFO) << "Connected to backend " << node_id_;
          DoReadHeader();
        }
      });
}

std::shared_ptr<BackendSession> BackendPool::GetBackend(uint32_t backend_id) {
  std::lock_guard<std::mutex> lock(mu_);
  auto iter = backends_.find(backend_id);
  if (iter == backends_.end()) {
    return nullptr;
  }
  return iter->second;
}

void BackendPool::AddBackend(std::shared_ptr<BackendSession> backend) {
  std::lock_guard<std::mutex> lock(mu_);
  backend->Start();
  backends_.emplace(backend->node_id(), backend);
}

void BackendPool::RemoveBackend(std::shared_ptr<BackendSession> backend) {
  std::lock_guard<std::mutex> lock(mu_);
  LOG(INFO) << "Remove backend " << backend->node_id();
  backend->Stop();
  backends_.erase(backend->node_id());
}

void BackendPool::RemoveBackend(uint32_t backend_id) {
  std::lock_guard<std::mutex> lock(mu_);
  auto iter = backends_.find(backend_id);
  if (iter == backends_.end()) {
    return;
  }
  LOG(INFO) << "Remove backend " << backend_id;
  iter->second->Stop();
  backends_.erase(iter);
}

std::vector<uint32_t> BackendPool::UpdateBackendList(
    std::unordered_set<uint32_t> list) {
  std::lock_guard<std::mutex> lock(mu_);
  // Remove backends that are not on the list
  for (auto iter = backends_.begin(); iter != backends_.end(); ) {
    if (list.count(iter->first) == 0) {
      auto backend_id = iter->first;
      iter->second->Stop();
      iter = backends_.erase(iter);
      LOG(INFO) << "Remove backend " << backend_id;
    } else {
      ++iter;
    }
  }
  // Find out new backends
  std::vector<uint32_t> missing;
  for (auto backend_id : list) {
    if (backends_.count(backend_id) == 0) {
      missing.push_back(backend_id);
    }
  }
  return missing;
}

void BackendPool::StopAll() {
  std::lock_guard<std::mutex> lock(mu_);
  for (auto iter : backends_) {
    iter.second->Stop();
  }
  backends_.clear();
}

} // namespace nexus
