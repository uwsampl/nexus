#include "nexus/backend/task.h"

namespace nexus {
namespace backend {

Task::Task() : Task(nullptr) {}

Task::Task(std::shared_ptr<Connection> conn) :
    DeadlineItem(),
    connection(conn),
    model(nullptr),
    stage(kPreprocess),
    filled_outputs(0) {
  timer.Record("begin");
}

void Task::DecodeQuery(std::shared_ptr<Message> message) {
  message->DecodeBody(&query);
  ModelSession sess;
  ParseModelSession(query.model_session_id(), &sess);
  SetDeadline(std::chrono::milliseconds(sess.latency_sla()));
}

bool Task::AddOutput(int index, std::unique_ptr<Output> output) {
  outputs[index] = std::move(output);
  ++filled_outputs;
  if (filled_outputs == outputs.size()) {
    return true;
  }
  return false;
}

bool Task::AddVirtualOutput(int index) {
  result.set_status(TIMEOUT);
  outputs[index] = nullptr;
  ++filled_outputs;
  if (filled_outputs == outputs.size()) {
    return true;
  }
  return false;
}

} // namespace backend
} // namespace nexus

