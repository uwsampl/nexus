#ifndef NEXUS_COMMON_MODEL_HANDLER_H_
#define NEXUS_COMMON_MODEL_HANDLER_H_

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "common/backend_pool.h"
#include "common/data_type.h"
#include "proto/nnquery.pb.h"

namespace nexus {

class ModelHandler;

class Output {
 public:
  Output(uint32_t timeout);

  uint32_t status();

  std::string error_message();

  void FillReply(ReplyProto* reply);

  const Record& operator[](uint32_t idx);

 private:
  void SetResult(const QueryResultProto& result);

  void SetResult(uint32_t error, const std::string& error_msg);

  void WaitForReadyOrTimeout();

 private:
  bool ready_;
  std::chrono::milliseconds timeout_;
  uint32_t status_;
  std::string error_message_;
  std::vector<Record> records_;
  std::mutex mutex_;
  std::condition_variable cv_;

  friend class ModelHandler;
};

class ModelHandler {
 public:
  ModelHandler(const ModelSession& model_session, BackendPool& pool);

  std::string model_session_id() const { return model_session_id_; }

  std::shared_ptr<Output> Execute(
      const ValueProto& input, std::vector<std::string> output_fields={},
      uint32_t topk=1, std::vector<RectProto> windows={});

  void HandleResult(const QueryResultProto& result);

  void UpdateRoute(const ModelRoute& route);

 private:
  std::shared_ptr<BackendSession> GetBackend();

 private:
  ModelSession model_session_;
  std::string model_session_id_;
  BackendPool& backend_pool_;
  std::atomic<uint32_t> query_id_;
  /*! \brief map from backend id to its serving rate, protected by route_mu_ */
  std::vector<std::pair<uint32_t, float> > backend_rates_;
  float total_throughput_;
  std::unordered_map<uint32_t, std::shared_ptr<Output> > outputs_;
  std::mutex route_mu_;
  std::mutex outputs_mu_;
  /*! \brief random number generator */
  std::random_device rd_;
  std::mt19937 rand_gen_;
};

} // namespace nexus

#endif // NEXUS_COMMON_MODEL_HANDLER_H_
