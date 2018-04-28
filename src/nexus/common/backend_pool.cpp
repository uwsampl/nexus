#include <glog/logging.h>

#include "nexus/common/backend_pool.h"
#include "nexus/common/util.h"

namespace nexus {

BackendSession::BackendSession(BackendPool* pool, const BackendInfo& info,
                               boost::asio::io_service& io_service,
                               MessageHandler* handler):
    Connection(io_service, handler),
    pool_(pool) {
  // init backend node_id and address
  node_id_ = info.node_id();
  address_ = info.server_address();
  // connect to backend server
  DoConnect(io_service);
}

BackendSession::~BackendSession() {
  Stop();
}

void BackendSession::Stop() {
  if (running_) {
    running_ = false;
    Connection::Stop();
  }
}

void BackendSession::AddModelSession(const std::string& model_session) {
  model_sessions_.insert(model_session);
}

bool BackendSession::RemoveModelSession(const std::string& model_session) {
  model_sessions_.erase(model_session);
  return model_sessions_.empty();
}

void BackendSession::DoConnect(boost::asio::io_service& io_service) {
  boost::asio::ip::tcp::resolver::iterator endpoint;
  std::vector<std::string> tokens;
  SplitString(address_, ':', &tokens);
  boost::asio::ip::tcp::resolver resolver(io_service);
  endpoint = resolver.resolve({ tokens[0], tokens[1] });
  boost::asio::async_connect(
      socket_, endpoint,
      [this](boost::system::error_code ec,
             boost::asio::ip::tcp::resolver::iterator) {
        if (ec) {
          LOG(ERROR) << "Failed to connect to backend " << node_id_ <<
              " (" << ec << "): " << ec.message();
          pool_->RemoveBackend(node_id_);
        } else {
          running_ = true;
          LOG(INFO) << "Connected to backend " << node_id_;
          Start();
        }
      });
}

BackendPool::BackendPool(boost::asio::io_service& io_service,
                         MessageHandler* handler) :
    io_service_(io_service),
    handler_(handler) {
}

std::shared_ptr<BackendSession> BackendPool::GetBackend(uint32_t backend_id) {
  std::lock_guard<std::mutex> lock(pool_mu_);
  auto iter = backends_.find(backend_id);
  if (iter == backends_.end()) {
    return nullptr;
  }
  return iter->second;
}

void BackendPool::AddBackend(const BackendInfo& backend_info,
                             const std::string& model_session) {
  std::lock_guard<std::mutex> lock(pool_mu_);
  uint32_t node_id = backend_info.node_id();
  if (backends_.find(node_id) == backends_.end()) {
    LOG(INFO) << "New connection to backend " << node_id;
    auto backend = std::make_shared<BackendSession>(this, backend_info,
                                                    io_service_, handler_);
    backends_.emplace(node_id, backend);
  }
  backends_.at(node_id)->AddModelSession(model_session);
}

void BackendPool::RemoveBackend(uint32_t backend_id) {
  std::lock_guard<std::mutex> lock(pool_mu_);
  auto iter = backends_.find(backend_id);
  if (iter == backends_.end()) {
    return;
  }
  iter->second->Stop();
  backends_.erase(iter);
}

void BackendPool::RemoveModelSessionFromBackend(
    uint32_t backend_id, const std::string& model_session) {
  std::lock_guard<std::mutex> lock(pool_mu_);
  auto iter = backends_.find(backend_id);
  if (iter == backends_.end()) {
    return;
  }
  auto backend = iter->second;
  if (backend->RemoveModelSession(model_session)) {
    // No model session loaded by this backend
    LOG(INFO) << "Remove connection to backend " << backend_id;
    backend->Stop();
    backends_.erase(iter);
  }
}

void BackendPool::StopAll() {
  std::lock_guard<std::mutex> lock(pool_mu_);
  for (auto iter : backends_) {
    auto backend = iter.second;
    backend->Stop();
  }
  backends_.clear();
}

} // namespace nexus
