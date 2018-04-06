#ifndef NEXUS_APP_WORKER_H_
#define NEXUS_APP_WORKER_H_

#include <atomic>
#include <thread>

#include "nexus/common/block_queue.h"

namespace nexus {
namespace app {

class Frontend;

class Worker {
 public:
  Worker(Frontend* frontend, BlockQueue<Message>& req_queue);

  void Start();

  void Stop();

  void Run();

 private:
  Frontend* frontend_;
  BlockQueue<Message>& request_queue_;
  volatile std::atomic_bool running_;
  std::thread thread_;
};

} // namespace app
} // namespace nexus

#endif // NEXUS_APP_MESSAGE_PROCESSOR_H_
