#ifndef NEXUS_COMMON_MODEL_HANDLER_H_
#define NEXUS_COMMON_MODEL_HANDLER_H_

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "nexus/common/backend_pool.h"
#include "nexus/common/data_type.h"
#include "nexus/common/metric.h"
#include "nexus/proto/nnquery.pb.h"

namespace nexus {
namespace app {

/*!
 * \brief QueryResult provides a mechanism to access the result of
 *   ansynchronous model execution.
 */
class QueryResult {
 public:
  /*!
   * \brief Constructor of OutputFuture
   * \param timeout_ms Timeout for output future in millisecond
   */
  QueryResult(uint64_t qid);

  bool ready() const { return ready_; }
  
  uint64_t query_id() const { return qid_; }
  /*! \brief Gets the status of output result */
  uint32_t status() const;
  /*! \brief Gets the error message if any error happens in the execution */
  std::string error_message() const;
  /*!
   * \brief Output the result to reply protobuf
   * \param reply ReplyProto to be filled
   */
  void ToProto(ReplyProto* reply) const;
  /*!
   * \brief Get the record given then index
   * \param idx Index of record
   * \return Record at idx
   */
  const Record& operator[](uint32_t idx) const;
  /*! \brief Get number of records in the output */
  uint32_t num_records() const;

  void SetResult(const QueryResultProto& result);

 private:
  void CheckReady() const;

  void SetError(uint32_t error, const std::string& error_msg);

 private:
  uint64_t qid_;
  std::atomic<bool> ready_;
  uint32_t status_;
  std::string error_message_;
  std::vector<Record> records_;
};

class RequestContext;

enum LoadBalancePolicy {
  // Weighted round robin
  LB_WeightedRR = 1,
  // Query 2 backends and pick one with lowest utilization
  LB_Query = 2,
  // Deficit round robin
  LB_DeficitRR = 3,
};

class ModelHandler {
 public:
  ModelHandler(const std::string& model_session_id, BackendPool& pool,
               LoadBalancePolicy lb_policy);

  ~ModelHandler();

  std::string model_session_id() const { return model_session_id_; }

  std::shared_ptr<IntervalCounter> counter() const { return counter_; }

  std::shared_ptr<QueryResult> Execute(
      std::shared_ptr<RequestContext> ctx, const ValueProto& input,
      std::vector<std::string> output_fields={}, uint32_t topk=1,
      std::vector<RectProto> windows={});

  void HandleReply(const QueryResultProto& result);

  void UpdateRoute(const ModelRouteProto& route);

  std::vector<uint32_t> BackendList();

 private:
  std::shared_ptr<BackendSession> GetBackend();
  
  std::shared_ptr<BackendSession> GetBackendWeightedRoundRobin();

  std::shared_ptr<BackendSession> GetBackendDeficitRoundRobin();

  void DeficitDaemon();

  ModelSession model_session_;
  std::string model_session_id_;
  BackendPool& backend_pool_;
  LoadBalancePolicy lb_policy_;
  static std::atomic<uint64_t> global_query_id_;

  std::vector<uint32_t> backends_;
  /*!
   * \brief Mapping from backend id to its serving rate,
   *
   *   Guarded by route_mu_
   */
  std::unordered_map<uint32_t, double> backend_rates_;

  std::unordered_map<uint32_t, double> backend_quantums_;
  float total_throughput_;
  /*! \brief Interval counter to count number of requests within each
   *  interval.
   */
  std::shared_ptr<IntervalCounter> counter_;

  std::unordered_map<uint64_t, std::shared_ptr<RequestContext> > query_ctx_;
  std::mutex route_mu_;
  std::mutex query_ctx_mu_;
  /*! \brief random number generator */
  std::atomic<uint32_t> backend_idx_;
  std::random_device rd_;
  std::mt19937 rand_gen_;

  std::atomic<bool> running_;
  std::thread deficit_thread_;
};

} // namespace app
} // namespace nexus

#endif // NEXUS_COMMON_MODEL_HANDLER_H_
