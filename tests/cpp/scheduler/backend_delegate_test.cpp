#include <chrono>
#include <gtest/gtest.h>
#include <memory>

#include "nexus/common/config.h"
#include "nexus/common/model_db.h"
#include "nexus/proto/control.pb.h"
#include "nexus/proto/nnquery.pb.h"
#include "nexus/scheduler/backend_delegate.h"

DECLARE_string(model_db);

namespace nexus {
namespace scheduler {

class BackendDelegateTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    ModelDatabase::Singleton().Init(FLAGS_model_db);
    gpu_device_ = "TITAN_X_(Pascal)";
    gpu_available_memory_ = 100;
    backend_.reset(new BackendDelegate(
        1, "127.0.0.1:8001", "127.0.0.1:8002", gpu_device_,
        gpu_available_memory_, 1, EPOCH_INTERVAL_SEC));
  }

  std::string gpu_device_;
  size_t gpu_available_memory_;
  std::unique_ptr<BackendDelegate> backend_;
};

TEST_F(BackendDelegateTest, PrepareLoadModel) {
  ModelSession vgg16_sess;
  vgg16_sess.set_framework("caffe");
  vgg16_sess.set_model_name("vgg16");
  vgg16_sess.set_version(1);
  vgg16_sess.set_latency_sla(500);

  ModelSession vgg_face_sess;
  vgg_face_sess.set_framework("caffe");
  vgg_face_sess.set_model_name("vgg_face");
  vgg_face_sess.set_version(1);
  vgg_face_sess.set_latency_sla(300);

  // Residue workload
  for (float workload : {50., 100., 150., 200., 250.}) {
    ModelInstanceConfig config;
    float occupancy;
    bool ret = backend_->PrepareLoadModel(vgg16_sess, workload, &config,
                                          &occupancy);
    ASSERT_TRUE(ret);
    ASSERT_GE(config.throughput(), workload);
    ASSERT_GT(config.batch(), 0);
    ASSERT_LE(occupancy, 1.);
  }

  // Saturate entire gpu when workload > 298
  for (float workload : {300., 400., 500.}) {
    ModelInstanceConfig config;
    float occupancy;
    bool ret = backend_->PrepareLoadModel(vgg16_sess, workload, &config,
                                          &occupancy);
    ASSERT_TRUE(ret);
    ASSERT_GT(config.batch(), 0);
    ASSERT_EQ(occupancy, 1.);
  }

  ModelInstanceConfig vgg16_cfg;
  float occupancy;
  backend_->PrepareLoadModel(vgg16_sess, 150., &vgg16_cfg, &occupancy);
  backend_->LoadModel(vgg16_cfg);
  ASSERT_NEAR(backend_->Occupancy(), occupancy, 1e-3);

  // Try load second model
  for (float workload : {50, 100, 125}) {
    ModelInstanceConfig config;
    float occupancy;
    bool ret = backend_->PrepareLoadModel(vgg_face_sess, workload, &config,
                                          &occupancy);
    LOG(INFO) << config.DebugString();
    LOG(INFO) << occupancy;
    ASSERT_TRUE(ret);
    ASSERT_GE(config.throughput(), workload);
    ASSERT_GT(config.batch(), 0);
    ASSERT_LE(occupancy, 1.);
  }

  for (float workload : {150, 200, 250}) {
    ModelInstanceConfig config;
    float occupancy;
    bool ret = backend_->PrepareLoadModel(vgg_face_sess, workload, &config,
                                          &occupancy);
    ASSERT_FALSE(ret);
  }
  
  ModelInstanceConfig vgg_face_cfg;
  backend_->PrepareLoadModel(vgg_face_sess, 125., &vgg_face_cfg, &occupancy);
  
  backend_->LoadModel(vgg_face_cfg);
  ASSERT_NEAR(backend_->Occupancy(), occupancy, 1e-3);
}

TEST_F(BackendDelegateTest, CheckAlive) {
  std::this_thread::sleep_for(std::chrono::milliseconds(2100));
  ASSERT_FALSE(backend_->IsAlive());
  backend_->Tick();
  ASSERT_TRUE(backend_->IsAlive());
}

} // namespace scheduler
} // namespace nexus
