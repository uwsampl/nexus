#ifndef NEXUS_SCHEDULER_RPSRECORD_H_
#define NEXUS_SCHEDULER_RPSRECORD_H_

#include <chrono>
#include <deque>
#include <gflags/gflags.h>
#include <grpc++/grpc++.h>
#include <gtest/gtest.h>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <yaml-cpp/yaml.h>
#include <cmath>

#include "nexus/common/rpc_call.h"
#include "nexus/common/rpc_service_base.h"
#include "nexus/proto/control.grpc.pb.h"
#include "nexus/scheduler/backend_delegate.h"
#include "nexus/scheduler/frontend_delegate.h"
#include "nexus/scheduler/sch_info.h"
class RpsRecord {
 public:
  RpsRecord() {}
  void init(std::map<string, uint32_t> models_id, double split) {
    this.models_id = models_id;
    max_size = 30;
    begin = end = 0;
    int n = models_id.size();
    models_rps.resize(n + 1);
    len = 1;
    for (int i = 0; i <= n; i++) {
      models_rps[i].resize(max_size * 2 + 1);
      if(i == 0) {
        models_rps[i][0] = 100; //ms 
      }
      else {
        models_rps[i][0] = split;
      }
    }
  }
  void add(const CurRpsRequest& request) {
    double interval = request.interval;
    uint32_t n = request.n;
    models_rps[0].push(interval);
    end += 1;
    for (int i = 0; i < n; i++) {
      uint32_t id = models_id[request.model_rps(i).model];
      models_rps[id][end] = request.model_rps(i).rps;
    }
    len++;
    if(len > max_size) {
      begin ++;
    }
    if(begin >= max_size) {
      int n = models_id.size();
      for (int i = 0; i <= n; i++) {
        for (int j = 0; j < len; j++) {
          models_rps[i][j] = models_rps[i][j + begin];
          models_rps[i][j + begin] = 0;
        }
      }
      begin = 0;
      end = len - 1;
    }
    //return true;
  }
  double sqr(double x) {
    return x * x;
  }
  std::vector<double> getRecord() {
    std::vector<double> ret;
    double total = 0.0;
    for (int i = begin; i <= end; i++) {
      total += models_rps[0][i];
    }
    int n = models_id.size();
    for (int i = 1; i <= n; i++) {
      double mean = 0.0, std = 0.0;
      for (int j = begin; j <= end; j++) {
        mean += models_rps[i][j] * models[0][j];
      }
      mean /= total;
      for (int j = begin; j <= end; j++) {
        std += models_rps[0][i] * sqr(models_rps[i][j] / models_rps[0][i] - mean);
      }
      std /= total;
      std = std::sqrt(std);
      ret.push_back(mean + std);
    }
    return ret;
  }
 private:
  std::map<std::string, uint32_t> models_id;
  std::vector<std::vector<double> > models_rps;
  //std::queue<double> intervals;
  uint32_t max_size, begin, end, len;
}
