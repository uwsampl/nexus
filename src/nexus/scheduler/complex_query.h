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
  void init(std::map<std::string, uint32_t> models_id, double split);
  void add(const CurRpsProto& request);
  double sqr(double x) {
    return x * x;
  }
  std::vector<double> getRecord();
 private:
  std::map<std::string, uint32_t> models_id;
  std::vector<std::vector<double> > models_rps;
  //std::queue<double> intervals;
  uint32_t max_size, begin, end, len;
};
class QuerySplit {

 public:
  QuerySplit() {}
  void addModel(ModelSession model, double lat);
  void updateLatencys(std::vector<uint32_t> latencys);
  std::vector<ModelSession> constructSplit(std::vector<uint32_t> latencys);
  
  std::vector<ModelSession> last_subscribe_models();
  
  std::vector<ModelSession> cur_subscribe_models();
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
  bool state_;
  
};

class ComplexQuery {
 public:
  ComplexQuery() {}
  double structure(int n);
  double gpu_need(std::vector<ModelSession> sess, std::vector<double> rps);
  QuerySplit* split();
  CtrlStatus init(const LoadDependencyProto& request, std::string common_gpu);
  void addRecord(const CurRpsProto& request);
 private:
  uint32_t n;
  double latency;
  std::vector<ModelSession> model_sessions_;
  std::map<std::string, uint32_t> models_id_;
  std::vector<std::vector<uint32_t> > edges_;
  std::vector<std::vector<uint32_t> > redges_;
  std::vector<std::string> models_;
  std::vector<double> intervals_;
  std::vector<uint32_t> degrees_;
  std::vector<uint32_t> depth_;
  std::vector<uint32_t> node_; 
  std::string common_gpu_;
  double latency_;
  int step_;  
  std::vector<std::vector<std::pair<float, uint32_t>> > max_throughputs_;
  RpsRecord rps;
  QuerySplit qs;
};
}
}
#endif 
