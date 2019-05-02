#ifndef NEXUS_APP_APP_BASE_H_
#define NEXUS_APP_APP_BASE_H_

#include <gflags/gflags.h>
#include "nexus/app/frontend.h"

DECLARE_int32(load_balance);

namespace nexus {
namespace app {

class AppBase : public Frontend {
 public:
  AppBase(const std::string& port,
          const std::string& rpc_port,
          const std::string& sch_addr,
          size_t nthreads);

  ~AppBase() override;

  void Start();

  virtual void Setup() {}

  bool IsComplexQuery() const;

  void ComplexQuerySetup(const std::string &cq_name, uint32_t slo_us, uint32_t step_us);

  void ComplexQueryAddEdge(const std::shared_ptr<ModelHandler>& source,
                           const std::shared_ptr<ModelHandler>& target);

 protected:
  std::shared_ptr<ModelHandler> GetModelHandler(
      const std::string& framework, const std::string& model_name,
      uint32_t version, uint64_t latency_sla, float estimate_workload=0.,
      std::vector<uint32_t> image_size={},
      LoadBalancePolicy lb_policy=LoadBalancePolicy(FLAGS_load_balance));
  size_t nthreads_;
  QueryProcessor* qp_;

  std::string cq_id_;
  uint32_t slo_us_;
  uint32_t step_us_;
};

void LaunchApp(AppBase* app);

} // namespace app
} // namespace nexus

#endif // NEXUS_APP_APP_BASE_H_
