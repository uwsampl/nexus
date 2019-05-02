#ifndef NEXUS_APP_WORKER_H_
#define NEXUS_APP_WORKER_H_

#include <atomic>
#include <thread>

#include "nexus/app/query_processor.h"
#include "nexus/app/request_context.h"

namespace nexus {
namespace app {

class Frontend;

class Worker {
 public:
  Worker(QueryProcessor* qp, RequestPool& req_pool);

  void Start();

  void Stop();

  void Join();

  void Run();

 private:
  QueryProcessor* qp_;
  RequestPool& req_pool_;
  volatile std::atomic_bool running_;
  std::thread thread_;
};

} // namespace app
} // namespace nexus

#endif // NEXUS_APP_MESSAGE_PROCESSOR_H_
