#include "nexus/backend/task.h"
#include "nexus/common/model_def.h"

namespace nexus {
namespace backend {

Input::Input(ArrayPtr arr, std::shared_ptr<Task> task, int idx) :
    DeadlineItem(task->deadline()),
    array(arr),
    task(task),
    index_in_task(idx) {}

Output::Output(const std::unordered_map<std::string, ArrayPtr>& arrs,
               std::shared_ptr<Task> task, int idx) :
    arrays(arrs),
    task(task),
    index_in_task(idx) {}

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

void Task::AppendInput(ArrayPtr arr) {
  auto input = std::make_shared<Input>(arr, shared_from_this(),
                                       inputs.size());
  inputs.push_back(input);
  // Put a placeholder in the outputs
  outputs.push_back(nullptr);
}

bool Task::AddOutput(std::shared_ptr<Output> output) {
  outputs[output->index_in_task] = output;
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

