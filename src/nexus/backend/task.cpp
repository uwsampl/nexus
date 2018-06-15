#include "nexus/backend/task.h"
#include "nexus/common/model_def.h"

namespace nexus {
namespace backend {

Input::Input(TimePoint deadline, uint64_t tid, int idx, ArrayPtr arr) :
    DeadlineItem(deadline),
    tid(tid),
    index(idx),
    array(arr) {}

Output::Output(uint64_t tid, int idx,
               const std::unordered_map<std::string, ArrayPtr>& arrs) :
    tid(tid),
    index(idx),
    arrays(arrs) {}

std::atomic<uint64_t> Task::global_tid_(0);

Task::Task() : Task(nullptr) {}

Task::Task(std::shared_ptr<Connection> conn) :
    DeadlineItem(),
    connection(conn),
    model(nullptr),
    stage(kPreprocess),
    filled_outputs(0) {
  tid = global_tid_.fetch_add(1, std::memory_order_relaxed);
  timer.Record("begin");
}

void Task::DecodeQuery(std::shared_ptr<Message> message) {
  message->DecodeBody(&query);
  ModelSession sess;
  ParseModelSession(query.model_session_id(), &sess);
  SetDeadline(std::chrono::milliseconds(sess.latency_sla()));
}

void Task::AppendInput(ArrayPtr arr) {
  auto input = std::make_shared<Input>(deadline(), tid, inputs.size(), arr);
  inputs.push_back(input);
  // Put a placeholder in the outputs
  outputs.push_back(nullptr);
}

bool Task::AddOutput(std::shared_ptr<Output> output) {
  outputs[output->index] = output;
  uint32_t filled = ++filled_outputs;
  if (filled == outputs.size()) {
    return true;
  }
  return false;
}

bool Task::AddVirtualOutput(int index) {
  result.set_status(TIMEOUT);
  uint32_t filled = ++filled_outputs;
  if (filled == outputs.size()) {
    return true;
  }
  return false;
}

} // namespace backend
} // namespace nexus

