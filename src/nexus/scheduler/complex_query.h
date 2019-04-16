#ifndef NEXUS_SCHEDULER_COMPLEXQUERY_H_
#define NEXUS_SCHEDULER_COMPLEXQUERY_H_
#include <chrono>
#include <deque>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <grpc++/grpc++.h>
#include <gtest/gtest.h>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <yaml-cpp/yaml.h>
#include <sstream>
#include <utility>

#include "nexus/common/rpc_call.h"
#include "nexus/common/rpc_service_base.h"
#include "nexus/common/config.h"
#include "nexus/common/model_db.h"
#include "nexus/common/model_def.h"
#include "nexus/common/util.h"
#include "nexus/proto/control.grpc.pb.h"
#include "nexus/proto/nnquery.pb.h"

namespace nexus {
namespace scheduler {

class RpsRecord {
 public:
  RpsRecord() {}
  void init(std::map<std::string, uint32_t> models_id);
  void add(const CurRpsProto& request);
  float sqr(float x) {
    return x * x;
  }
  std::vector<float> getRecord();
 private:
  std::map<std::string, uint32_t> models_id_;
  std::vector<std::vector<float> > models_rps_;
  uint32_t max_size_, begin_, end_, len_;
};
class QuerySplit {

 public:
  QuerySplit() {}
  void init(int n);
  void addModel(ModelSession model, float lat);
  void updateLatencys(std::vector<uint32_t> latencys);
  
  void updateWorkloads(std::vector<float> workloads);
  std::vector<ModelSession> constructSplit(std::vector<uint32_t> latencys);
  
  std::vector<ModelSession> last_subscribe_models();
  
  std::vector<ModelSession> cur_subscribe_models();
  
  std::vector<float> cur_workloads();
  
  void setState(bool state) {
    state_ = state;
  }
  bool getState() {
    return state_;
  }
 private:
  std::vector<ModelSession> models_;
  std::vector<uint32_t> latencys_;
  std::vector<uint32_t> last_latencys_;
  std::vector<float> workloads_;
  bool state_;
  
};

class ComplexQuery {
 public:
  ComplexQuery() {}
  float structure(int n);
  float gpu_need(std::vector<ModelSession> sess, std::vector<float> rps);
  QuerySplit* split();
  CtrlStatus init(const LoadDependencyProto& request, std::string common_gpu);
  void addRecord(const CurRpsProto& request);
 private:
  uint32_t ori_latency_;
  uint32_t n_;
  std::vector<ModelSession> model_sessions_;
  std::map<std::string, uint32_t> models_id_;
  std::vector<std::vector<uint32_t> > edges_;
  std::vector<std::vector<uint32_t> > redges_;
  std::vector<std::vector<uint32_t> > layers_;
  std::vector<std::string> models_;
  std::vector<float> intervals_;
  std::vector<uint32_t> degrees_;
  std::vector<uint32_t> depth_;
  std::vector<uint32_t> node_; 
  std::string common_gpu_;
  uint32_t latency_;
  uint32_t step_;
  uint32_t diameter_;  
  std::vector<std::vector<std::pair<float, uint32_t>> > max_throughputs_;
  RpsRecord rps_record_;
  QuerySplit query_split_;
};
}
}
#endif 
