#ifndef NEXUS_APP_EXEC_BLOCK_H_
#define NEXUS_APP_EXEC_BLOCK_H_

#include <memory>
#include <unordered_set>

#include "nexus/app/model_handler.h"
#include "nexus/app/request_context.h"

namespace nexus {
namespace app {

using ExecFunc = std::function<std::vector<VariablePtr>(
    std::shared_ptr<RequestContext> ctx)>;

class ExecBlock {
 public:
  ExecBlock(int id, ExecFunc func, std::vector<std::string> required_vars) :
      id_(id),
      func_(func) {
    for (auto var_name : required_vars) {
      dependency_.insert(var_name);
    }
  }

  int id() const { return id_; }

  std::unordered_set<std::string> dependency() const { return dependency_; }

  std::vector<VariablePtr> Run(std::shared_ptr<RequestContext> ctx) {
    return func_(ctx);
  }

 private:
  int id_;
  ExecFunc func_;
  std::unordered_set<std::string> dependency_;
};

} // namespace app
} // namespace nexus

#endif // NEXUS_APP_EXEC_BLOCK_H_
