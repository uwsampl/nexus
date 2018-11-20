#include "nexus/app/complex_query_app.h"

namespace nexus{
namespace app {

ComplexQueryApp::ComplexQueryApp(std::string port, std::string rpc_port, 
                                 std::string sch_addr, size_t nthreads) :
    AppBase(port, rpc_port, sch_addr, nthreads) {
  SetComplexQuery();
}

ComplexQueryApp::Initialize(size_t n, size_t m, int latency_sla) {
  n_ = n;
  m_ = m;
  latency_sla_ = latency_sla;
  proto_ = request_.mutable_dependency();
  proto_->set_n(2);
  proto_->set_m(1);
  proto_->set_latency(latency_slo_);
  
}

std::shared_ptr<ModelHandler> ComplexQueryApp::AddModel(std::string &framework, 
                       std::string &model, uint version, 
                       uint estimate_latency, uint estimate_workload, 
                       std::vector<uint32_t> image_size) {
  auto model_handler = GetModelHandler(true, framework, model, version, 
                                       estimate_latency, estimate_workload, 
                                       image_size);
  model_handlers_[model_handler.VirtualModelSessionId()] = model_handler;
  auto model_sess = model_handler.GetModelSession();
  model_sess.set_estimate_latency(estimate_latency);
  proto_->add_models()->CopyFrom(model_sess);
}

  void ComplexQueryApp::AddEdge(std::shared_ptr<ModelHandler> model1, 
                        std::shared_ptr<ModelHandler> model2); {
  auto* edge = proto->add_edges();
  edge->mutable_v1()->CopyFrom(model1.GetModelSession());
  edge->mutable_v2()->CopyFrom(model2.GetModelSession());
}

void ComplexQueryApp::LoadDependency() {
  LoadDependency(request_);
}

void ComplexQueryApp::BuildQueryProcessor() {
  qp_ = new QueryProcessor(exec_blocks_);
}

void ComplexQueryApp::AddExecBlock(ExecFunc func, std::vector<std::string> variables) {
  int id = exec_blocks_.size();
  ExecBlock* exec_block = new ExecBlock(id, func, variables);
  exec_blocks_.push_back(exec_block);
}

RectProto ComplexQueryApp::GetRect(int left, int right, int top, int bottom) {
  RectProto rect;
  rect.set_left(left);
  rect.set_right(right);
  rect.set_top(top);
  rect.set_bottom(bottom);
  return rect;
}
 
} //namespace app
} //namespace nexus
