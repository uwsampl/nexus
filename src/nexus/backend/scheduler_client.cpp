#include <glog/logging.h>

#include "nexus/backend/backend_server.h"
#include "nexus/backend/scheduler_client.h"
#include "nexus/common/time_util.h"

namespace nexus {
namespace backend {

SchedulerClient::SchedulerClient(BackendServer* backend,
                                     const std::string& scheduler_address):
    backend_(backend),
    running_(false) {
  auto channel = grpc::CreateChannel(scheduler_address,
                                     grpc::InsecureChannelCredentials());
  stub_ = SchedulerCtrl::NewStub(channel);
}

bool SchedulerClient::Start() {
  running_ = true;
  thread_ = std::thread(&SchedulerClient::Run, this);
  return true;
}

void SchedulerClient::Stop() {
  running_ = false;
  if (thread_.joinable()) {
    thread_.join();
  }
}

void SchedulerClient::Run() {
  std::chrono::seconds tick_interval(1);
  TimePoint last_tick_time = Clock::now();
  while (running_) {
    TimePoint next_tick_time = last_tick_time + tick_interval;
    auto models = backend_->GetAllModelInstances();
    for (auto model : models) {
      uint64_t count = model->Tick();
      double rate = model->GetRate();
      if (rate < 1) {
        continue;
      }
      LOG(INFO) << Framework_name(model->framework()) << ":" <<
          model->model_name() << " " << count << " reqs, avg rate: " << rate;
    }
    std::this_thread::sleep_until(next_tick_time);
    last_tick_time = next_tick_time;
  }
}

CtrlStatus SchedulerClient::Register() {
  // Prepare the request
  RegisterRequest request;
  request.set_node_type(BACKEND_NODE);
  request.set_node_id(backend_->node_id());
  request.set_server_port(backend_->port());
  request.set_rpc_port(backend_->rpc_port());
  GPUDevice* gpu_device = DeviceManager::Singleton().GetGPUDevice(
      backend_->gpu_id());
  request.set_gpu_device_name(gpu_device->device_name());
  request.set_gpu_available_memory(gpu_device->FreeMemory());

  grpc::ClientContext context;
  RegisterReply reply;
  grpc::Status status = stub_->Register(&context, request, &reply);
  if (!status.ok()) {
    LOG(ERROR) << status.error_code() << ": " << status.error_message();
    return CTRL_SERVER_UNREACHABLE;
  }
  CtrlStatus ret = reply.status();
  if (ret == CTRL_OK) {
    RpcReply r;
    backend_->UpdateModelTable(reply.init_model_table(), &r);
  }
  return ret;
}

void SchedulerClient::Unregister() {
  UnregisterRequest request;
  request.set_node_type(BACKEND_NODE);
  request.set_node_id(backend_->node_id());

  grpc::ClientContext context;
  RpcReply reply;
  grpc::Status status = stub_->Unregister(&context, request, &reply);
  if (!status.ok()) {
    LOG(ERROR) << status.error_code() << ": " << status.error_message();
    return;
  }
  CtrlStatus ret = reply.status();
  if (ret != CTRL_OK) {
    LOG(ERROR) << "Unregister error: " << CtrlStatus_Name(ret);
  }
}

} // namespace backend
} // namespace nexus
