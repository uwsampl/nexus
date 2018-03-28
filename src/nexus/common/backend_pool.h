#ifndef NEXUS_COMMON_BACKEND_POOL_H_
#define NEXUS_COMMON_BACKEND_POOL_H_

#include <unordered_map>

#include "common/connection.h"
#include "proto/control.pb.h"

namespace nexus {

class BackendPool;

class BackendSession : public Connection {
 public:
  explicit BackendSession(BackendPool* pool, const BackendInfo& info,
                          boost::asio::io_service& io_service,
                          MessageHandler* handler);

  ~BackendSession();

  uint32_t node_id() const { return node_id_; }

  std::string address() const { return address_; }

  void Stop() final;

 private:
  void DoConnect(boost::asio::io_service& io_service);

 protected:
  BackendPool* pool_;
  uint32_t node_id_;
  std::string address_;
  std::atomic_bool running_;
};

class BackendPool {
 public:
  BackendPool(boost::asio::io_service& io_service, MessageHandler* handler);

  void UpdateBackends(const BackendsUpdate& request,
                      BackendsUpdateReply* reply);

  std::shared_ptr<BackendSession> GetBackend(uint32_t backend_id);

  void RemoveBackend(uint32_t backend_id);

  void StopAll();

 protected:
  boost::asio::io_service& io_service_;
  MessageHandler* handler_;
  std::unordered_map<uint32_t, std::shared_ptr<BackendSession> > backends_;
  std::mutex pool_mu_;
  std::uint32_t version_;
};

} // namespace nexus

#endif // NEXUS_COMMON_BACKEND_POOL_H_
