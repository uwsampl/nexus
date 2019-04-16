#ifndef NEXUS_COMMON_BACKEND_POOL_H_
#define NEXUS_COMMON_BACKEND_POOL_H_

#include <sstream>
#include <unordered_map>

#include "nexus/common/connection.h"
#include "nexus/common/time_util.h"
#include "nexus/proto/control.grpc.pb.h"

namespace nexus {

class BackendPool;

class BackendSession : public Connection {
 public:
  explicit BackendSession(const BackendInfo& info,
                          boost::asio::io_context& io_context,
                          MessageHandler* handler);

  ~BackendSession();

  inline uint32_t node_id() const { return node_id_; }

  inline std::string ip() const { return ip_; }

  inline std::string server_port() const { return server_port_; }

  inline std::string rpc_port() const { return rpc_port_; }

  virtual void Start();

  virtual void Stop();

  double GetUtilization();

 protected:
  /*! \brief Asynchronously connect to backend server. */
  void DoConnect();

  /*! \brief Boost io service */
  boost::asio::io_context& io_context_;
  uint32_t node_id_;
  std::string ip_;
  std::string server_port_;
  std::string rpc_port_;
  std::atomic_bool running_;
  std::unique_ptr<BackendCtrl::Stub> stub_;
  double utilization_;
  TimePoint expire_;
  std::mutex util_mu_;
};

class BackendPool {
 public:
  BackendPool() {}

  std::shared_ptr<BackendSession> GetBackend(uint32_t backend_id);

  void AddBackend(std::shared_ptr<BackendSession> backend);

  void RemoveBackend(std::shared_ptr<BackendSession> backend);

  void RemoveBackend(uint32_t backend_id);

  std::vector<uint32_t> UpdateBackendList(std::unordered_set<uint32_t> list);

  void StopAll();

 protected:
  // brief from backend.node_id() to BackendSession
  std::unordered_map<uint32_t, std::shared_ptr<BackendSession> > backends_;
  std::mutex mu_;
};

} // namespace nexus

#endif // NEXUS_COMMON_BACKEND_POOL_H_
