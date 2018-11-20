#ifndef NEXUS_APP_APP_BASE_H_
#define NEXUS_APP_APP_BASE_H_

#include "nexus/app/frontend.h"

namespace nexus {
namespace app {

class AppBase : public Frontend {
 public:
  AppBase(std::string port, std::string rpc_port, std::string sch_addr,
          size_t nthreads);

  virtual ~AppBase();

  void Start();

  virtual void Setup() {}

  //virtual void Process(const RequestProto& request, ReplyProto* reply) {}

 protected:
  std::shared_ptr<ModelHandler> GetModelHandler(const bool complex_query,
      const std::string& framework, const std::string& model_name,
      uint32_t version, uint64_t latency_sla, float estimate_workload=0.,
      std::vector<uint32_t> image_size={});
  size_t nthreads_;
  QueryProcessor* qp_;
};

void LaunchApp(AppBase* app);

} // namespace app
} // namespace nexus

#endif // NEXUS_APP_APP_BASE_H_
