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
          Start();
        }
      });
}

BackendPool::BackendPool(boost::asio::io_service& io_service,
                         MessageHandler* handler) :
    io_service_(io_service),
    handler_(handler),
    version_(0) {
}

void BackendPool::UpdateBackends(const BackendsUpdate& request,
                                 BackendsUpdateReply* reply) {
  std::lock_guard<std::mutex> lock(pool_mu_);
  uint32_t base_version = request.base_version();
  if (base_version > 0 && base_version != version_) {
    reply->set_status(CTRL_ROUTE_TABLE_VERSION_MISMATCH);
    reply->set_version(version_);
    return;
  }
  if (base_version == 0) {
    std::unordered_map<uint32_t, BackendInfo&> infos;
    for (auto info : request.add_backend()) {
      uint32_t node_id = info.node_id();
      infos.emplace(node_id, info);
      if (backends_.find(node_id) == backends_.end()) {
        auto backend = std::make_shared<BackendSession>(this, info, io_service_,
                                                        handler_);
        backends_.emplace(node_id, backend);
      }
    }
    std::vector<uint32_t> remove_list;
    for (auto iter : backends_) {
      uint32_t node_id = iter.first;
      if (infos.find(node_id) == infos.end()) {
        remove_list.push_back(node_id);
      }
    }
    for (uint32_t node_id : remove_list) {
      backends_.erase(node_id);
    }
  } else {
    for (auto info : request.add_backend()) {
      auto backend = std::make_shared<BackendSession>(this, info, io_service_,
                                                      handler_);
      backends_.emplace(backend->node_id(), backend);
    }
    for (auto info : request.remove_backend()) {
      auto iter = backends_.find(info.node_id());
      if (iter == backends_.end()) {
        return;
      }
      backends_.erase(iter);
    }
  }
  version_ = request.curr_version();
  reply->set_status(CTRL_OK);
  reply->set_version(version_);
  LOG(INFO) << "Backend pool version: " << version_;
}
/*
void BackendPool::AddBackend(const BackendInfo& info) {
  std::lock_guard<std::mutex> lock(pool_mu_);
  auto backend = std::make_shared<BackendSession>(this, info, io_service_,
                                                  handler_);
  backends_.emplace(backend->node_id(), backend);
}
*/
std::shared_ptr<BackendSession> BackendPool::GetBackend(uint32_t backend_id) {
  std::lock_guard<std::mutex> lock(pool_mu_);
  auto iter = backends_.find(backend_id);
  if (iter == backends_.end()) {
    return nullptr;
  }
  return iter->second;
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

void BackendPool::StopAll() {
  std::lock_guard<std::mutex> lock(pool_mu_);
  for (auto iter : backends_) {
    auto backend = iter.second;
    backend->Stop();
  }
  backends_.clear();
}

} // namespace nexus
