#ifndef NEXUS_APP_REQUEST_CONTEXT_H_
#define NEXUS_APP_REQUEST_CONTEXT_H_

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <mutex>

#include "nexus/app/model_handler.h"
#include "nexus/app/user_session.h"
#include "nexus/common/block_queue.h"
#include "nexus/proto/nnquery.pb.h"

namespace nexus {
namespace app {

class Variable {
 public:
  Variable(std::string name, std::shared_ptr<QueryResult> result) :
      name_(name),
      data_{result} {
    if (!result->ready()) {
      pending_results_.emplace(result->query_id(), result);
    }
  }
  
  Variable(std::string name,
           std::vector<std::shared_ptr<QueryResult> > results) :
      name_(name),
      data_(results) {
    for (auto& r : results) {
      if (!r->ready()) {
        pending_results_.emplace(r->query_id(), r);
      }
    }
  }

  bool ready() const { return pending_results_.empty(); }

  std::string name() const { return name_; }

  std::shared_ptr<QueryResult> result() const { return data_.at(0); }

  std::shared_ptr<QueryResult> operator[](int idx) const {
    return data_.at(idx);
  }

  std::vector<uint64_t> query_ids() const {
    std::vector<uint64_t> qids;
    for (auto itr : pending_results_) {
      qids.push_back(itr.first);
    }
    return qids;
  }

  bool AddQueryResult(const QueryResultProto& result) {
    auto reply = pending_results_.at(result.query_id());
    reply->SetResult(result);
    pending_results_.erase(result.query_id());
    return pending_results_.empty();
  }

 private:
  std::string name_;
  std::vector<std::shared_ptr<QueryResult> > data_;
  std::unordered_map<uint64_t, std::shared_ptr<QueryResult> > pending_results_;
};

using VariablePtr = std::shared_ptr<Variable>;

enum RequestState {
  kUninitialized = 0,
  kRunning = 1,
  kBlocking = 2,
  kError = 3,
};

class ExecBlock;
class RequestPool;

class RequestContext : public DeadlineItem,
                       public std::enable_shared_from_this<RequestContext> {
 public:
  RequestContext(std::shared_ptr<UserSession> user_sess,
                 std::shared_ptr<Message> msg, RequestPool& req_pool);

  RequestProto* request() { return &request_; }

  const RequestProto& const_request() const { return request_; }

  ReplyProto* reply() { return &reply_; }

  const ReplyProto& const_reply() const { return reply_; }

  RequestState state() const { return state_; }

  bool finished();

  void SetState(RequestState state);

  void SetExecBlocks(std::vector<ExecBlock*> blocks);

  ExecBlock* NextReadyBlock();

  VariablePtr GetVariable(const std::string& var_name);

  void AddBlockReturn(std::vector<VariablePtr> vars);

  void HandleQueryResult(const QueryResultProto& result);

  void HandleError(uint32_t status, const std::string& error_msg);

  void RecordQuerySend(uint64_t qid);

  void SendReply();

 private:
  void AddReadyVariable(std::shared_ptr<Variable> var);

 protected:
  std::shared_ptr<UserSession> user_session_;
  RequestPool& req_pool_;
  RequestProto request_;
  ReplyProto reply_;
  //TimePoint beg_;
  std::atomic<RequestState> state_;
  
  std::deque<ExecBlock*> ready_blocks_;
  std::unordered_map<int, ExecBlock*> pending_blocks_;
  std::unordered_map<int, std::unordered_set<std::string> > block_deps_;
  
  std::unordered_map<std::string, VariablePtr> vars_;
  std::unordered_map<std::string, VariablePtr> waiting_vars_;
  std::unordered_map<uint64_t, std::string> qid_var_map_;
  std::unordered_map<uint64_t, QueryResultProto> dangling_results_;
  std::unordered_map<uint64_t, uint64_t> query_send_;
  std::mutex mu_;
};

class RequestPool {
 public:
  void AddNewRequest(std::shared_ptr<RequestContext> req) {
    ready_requests_.push(req);
  }

  void AddBlockRequest(std::shared_ptr<RequestContext> req) {
    std::lock_guard<std::mutex> lock(mu_);
    block_requests_.insert(req);
  }

  void MoveToReady(std::shared_ptr<RequestContext> req) {
    ready_requests_.push(req);
    std::lock_guard<std::mutex> lock(mu_);
    block_requests_.erase(req);
  }

  std::shared_ptr<RequestContext> GetRequest(std::chrono::milliseconds timeout) {
    return ready_requests_.pop(timeout);
  }

 private:
  BlockPriorityQueue<RequestContext> ready_requests_;
  std::unordered_set<std::shared_ptr<RequestContext> > block_requests_;
  std::mutex mu_;
};

} // namespace app
} // namespace nexus

#endif // NEXUS_APP_REQUEST_CONTEXT_H_
