#include <chrono>
#include <gtest/gtest.h>
#include <memory>

#include "nexus/common/config.h"
#include "nexus/common/model_db.h"
#include "nexus/proto/control.pb.h"
#include "nexus/proto/nnquery.pb.h"
#include "nexus/scheduler/backend_delegate.h"
#include "nexus/scheduler/frontend_delegate.h"
#include "nexus/scheduler/scheduler.h"

DECLARE_string(model_db);
DECLARE_int32(beacon);
DECLARE_int32(epoch);

namespace nexus {
namespace scheduler {

class SchedulerTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    gpu_device_ = "TITAN_X_(Pascal)";
    gpu_available_memory_ = 12L * 1024L * 1024L * 1024L;
    FLAGS_beacon = 1;
    FLAGS_epoch = 5;
    scheduler_.reset(new Scheduler("10001", 1));
    for (int i = 0; i < 5; ++i) {
      auto backend = std::make_shared<BackendDelegate>(
          i + 1, "127.0.0.1", "8001", "8002", gpu_device_,
          gpu_available_memory_, FLAGS_beacon, FLAGS_epoch);
      RegisterReply reply;
      scheduler_->RegisterBackend(backend, &reply);
      ASSERT_EQ(reply.status(), CTRL_OK);
      backends_.push_back(backend);
    }
    for (int i = 0; i < 3; ++i) {
      auto frontend = std::make_shared<FrontendDelegate>(
          i + 1, "127.0.0.1", "9001", "9002", FLAGS_beacon);
      RegisterReply reply;
      scheduler_->RegisterFrontend(frontend, &reply);
      ASSERT_EQ(reply.status(), CTRL_OK);
      frontends_.push_back(frontend);
    }
  }

  std::string LoadModel(int frontend_id, std::string framework,
                        std::string model_name, uint32_t latency_sla,
                        float workload) {
    LoadModelRequest request;
    request.set_node_id(frontend_id);
    request.set_estimate_workload(workload);
    auto model_sess = request.mutable_model_session();
    model_sess->set_framework(framework);
    model_sess->set_model_name(model_name);
    model_sess->set_version(1);
    model_sess->set_latency_sla(latency_sla);
    LoadModelReply reply;
    grpc::ServerContext ctx;
    scheduler_->LoadModel(ctx, request, &reply);
    LOG(INFO) << reply.DebugString();
    EXPECT_EQ(reply.status(), CTRL_OK);
    return reply.model_route().model_session_id();
  }

  void UpdateBackendStats(const std::string& model_sess_id,
                          std::vector<uint64_t> num_requests) {
    auto iter_session_info = scheduler_->session_table_.find(model_sess_id);
    ASSERT_NE(iter_session_info, scheduler_->session_table_.end());
    BackendStatsProto request;
    auto session_info = iter_session_info->second;
    for (auto iter : session_info->backend_throughputs) {
      request.set_node_id(iter.first);
      break;
    }
    auto model_stats = request.add_model_stats();
    model_stats->set_model_session_id(model_sess_id);
    for (auto n : num_requests) {
      model_stats->add_num_requests(n);
    }
    RpcReply reply;
    grpc::ServerContext ctx;
    scheduler_->UpdateBackendStats(ctx, request, &reply);
    ASSERT_EQ(reply.status(), CTRL_OK);
  }
  
  void TickAll() {
    for (auto backend : backends_) {
      backend->Tick();
    }
    for (auto frontend : backends_) {
      frontend->Tick();
    }
  }

  std::string gpu_device_;
  size_t gpu_available_memory_;
  std::unique_ptr<Scheduler> scheduler_;
  std::vector<std::shared_ptr<BackendDelegate> > backends_;
  std::vector<std::shared_ptr<FrontendDelegate> > frontends_;
};

TEST_F(SchedulerTest, LoadModel) {
  LoadModel(1, "caffe", "vgg16", 200, 500.);
  LoadModel(2, "caffe", "vgg_s", 100, 500.);
}

TEST_F(SchedulerTest, EpochSchedule) {
  auto model1_id = LoadModel(1, "caffe", "vgg16", 200, 500.);
  auto model2_id = LoadModel(2, "caffe", "vgg_s", 100, 500.);
  scheduler_->DisplayModelTable();
  TickAll();
  UpdateBackendStats(model1_id, {550});
  UpdateBackendStats(model2_id, {450});
  scheduler_->BeaconCheck();
  UpdateBackendStats(model1_id, {550});
  UpdateBackendStats(model2_id, {450});
  scheduler_->BeaconCheck();
  UpdateBackendStats(model1_id, {550});
  UpdateBackendStats(model2_id, {450});
  scheduler_->BeaconCheck();
  UpdateBackendStats(model1_id, {550});
  UpdateBackendStats(model2_id, {450});
  scheduler_->BeaconCheck();
  UpdateBackendStats(model1_id, {550});
  UpdateBackendStats(model2_id, {450});
  scheduler_->BeaconCheck();
  scheduler_->EpochSchedule();
}

} // namespace scheduler
} // namespace nexus
