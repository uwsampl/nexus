#ifndef NEXUS_BACKEND_BACKUP_CLIENT_H_
#define NEXUS_BACKEND_BACKUP_CLIENT_H_

#include <atomic>
#include <grpc++/grpc++.h>

#include "nexus/backend/task.h"
#include "nexus/common/backend_pool.h"
#include "nexus/common/time_util.h"
#include "nexus/proto/control.grpc.pb.h"

namespace nexus {
namespace backend {

class BackupClient : public BackendSession {
 public:
  explicit BackupClient(const BackendInfo& info,
                        boost::asio::io_service& io_service,
                        MessageHandler* handler);

  double GetUtilization();

  void Forward(std::shared_ptr<Task> task);

  void Reply(std::shared_ptr<Message> message);

 private:
  std::unique_ptr<BackendCtrl::Stub> stub_;
  double utilization_;
  TimePoint expire_;
  std::mutex util_mu_;
  /*! \brief Map from task id to frontend connection. Guarded by relay_mu_. */
  std::unordered_map<uint64_t, std::shared_ptr<Connection> > conns_;
  /*! \brief Map from task id to query id. Guarded by relay_mu_. */
  std::unordered_map<uint64_t, uint64_t> qid_lookup_;
  std::mutex relay_mu_;
};

} // namespace backend
} // namespace nexus

#endif // NEXUS_BACKEND_BACKUP_CLIENT_H_
