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
#include "nexus/proto/nnquery.pb.h"

namespace nexus {

class ModelHandler;

/*!
 * \brief OutputFuture provides a mechanism to access the result of
 *   ansynchronous model execution.
 */
class OutputFuture {
 public:
  /*!
   * \brief Constructor of OutputFuture
   * \param timeout_ms Timeout for output future in millisecond
   */
  OutputFuture(uint32_t timeout_ms);
  /*! \brief Gets the status of output result */
  uint32_t status();
  /*! \brief Gets the error message if any error happens in the execution */
  std::string error_message();
  /*!
   * \brief Fill the result to reply
   * \param reply ReplyProto to fill in
   */
  void FillReply(ReplyProto* reply);
  /*!
   * \brief Get the record given then index
   * \param idx Index of record
   * \return Record at idx
   */
  const Record& operator[](uint32_t idx);
  /*! \brief Get number of records in the output */
  uint32_t num_records();

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
  ModelHandler(const std::string& model_session_id, BackendPool& pool);

  std::string model_session_id() const { return model_session_id_; }

  std::shared_ptr<OutputFuture> Execute(
      const ValueProto& input, std::vector<std::string> output_fields={},
      uint32_t topk=1, std::vector<RectProto> windows={});

  void HandleResult(const QueryResultProto& result);

  void UpdateRoute(const ModelRouteProto& route);

  std::vector<uint32_t> BackendList();

 private:
  std::shared_ptr<BackendSession> GetBackend();

 private:
  ModelSession model_session_;
  std::string model_session_id_;
  BackendPool& backend_pool_;
  std::atomic<uint32_t> query_id_;
  /*!
   * \brief Mapping from backend id to its serving rate,
   *
   *   Guarded by route_mu_
   */
  std::vector<std::pair<uint32_t, float> > backend_rates_;
  float total_throughput_;
  std::unordered_map<uint32_t, std::shared_ptr<OutputFuture> > outputs_;
  std::mutex route_mu_;
  std::mutex outputs_mu_;
  /*! \brief random number generator */
  std::random_device rd_;
  std::mt19937 rand_gen_;
};

} // namespace nexus

#endif // NEXUS_COMMON_MODEL_HANDLER_H_
