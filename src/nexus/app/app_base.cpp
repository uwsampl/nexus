#include "nexus/app/app_base.h"

namespace nexus {
namespace app {

AppBase::AppBase(const std::string& port,
                 const std::string& rpc_port,
                 const std::string& sch_addr,
                 size_t nthreads) :
    Frontend(port, rpc_port, sch_addr),
    nthreads_(nthreads),
    qp_(nullptr),
    step_us_(0)
{
}

AppBase::~AppBase() {
  if (qp_ != nullptr) {
    delete qp_;
  }
}

void AppBase::Start() {
  CHECK(qp_ != nullptr) << "Query processor is not initialized";
  Run(qp_, nthreads_);
}

std::shared_ptr<ModelHandler> AppBase::GetModelHandler(
    const std::string& framework, const std::string& model_name,
    uint32_t version, uint64_t latency_sla, float estimate_workload,
    std::vector<uint32_t> image_size, LoadBalancePolicy lb_policy) {
  LoadModelRequest req;
  req.set_node_id(node_id());
  auto model_sess = req.mutable_model_session();
  model_sess->set_framework(framework);
  model_sess->set_model_name(model_name);
  model_sess->set_version(version);
  model_sess->set_latency_sla(latency_sla);
  if (image_size.size() > 0) {
    if (image_size.size() != 2) {
      LOG(ERROR) << "Image size is not 2";
      return nullptr;
    }
    model_sess->set_image_height(image_size[0]);
    model_sess->set_image_width(image_size[1]);
  }
  if (estimate_workload < 0) {
    LOG(ERROR) << "Estimate workload must be non-negative value";
    return nullptr;
  }
  if (estimate_workload > 0) {
    req.set_estimate_workload(estimate_workload);
  }

  auto model_handler = LoadModel(req, lb_policy);
  if (model_handler == nullptr) {
    // TODO: load model failed, should retry after some time,
    // or wait for callback from scheduler
    LOG(FATAL) << "Load model failed";
  }
  return model_handler;
}

bool AppBase::IsComplexQuery() const {
  return slo_us_ != 0;
}

void AppBase::ComplexQuerySetup(const std::string &cq_id, uint32_t slo_us, uint32_t step_us) {
  CHECK(!IsComplexQuery()) << "The complex query has been set up.";
  CHECK(!cq_id.empty()) << "cq_id cannot be empty.";
  CHECK(slo_us != 0) << "slo_us cannot be 0.";
  CHECK(step_us != 0) << "step_us cannot be 0.";
  cq_id_ = cq_id;
  slo_us_ = slo_us;
  step_us_ = step_us;

  ComplexQuerySetupRequest req;
  req.set_cq_id(cq_id_);
  req.set_slo_us(slo_us_);
  req.set_step_us(step_us);
  Frontend::ComplexQuerySetup(req);
}

void AppBase::ComplexQueryAddEdge(const std::shared_ptr<ModelHandler>& source,
                                  const std::shared_ptr<ModelHandler>& target) {
  ComplexQueryAddEdgeRequest req;
  req.set_cq_id(cq_id_);
  req.mutable_source()->CopyFrom(source->model_session());
  req.mutable_target()->CopyFrom(target->model_session());
  Frontend::ComplexQueryAddEdge(req);
}

void LaunchApp(AppBase* app) {
  app->Setup();
  app->Start();
}

} // namespace app
} // namespace nexus
