#include "nexus/scheduler/complex_query.h"
#include <cmath>
namespace nexus {
namespace scheduler {
void RpsRecord::init(std::map<std::string, uint32_t> models_id) {
  models_id_ = models_id;
  max_size_ = 30;
  begin_ = end_ = 0;
  uint n = models_id_.size();
  models_rps_.resize(n + 1);
  len_ = 1;
  for (int i = 0; i <= n; i++) {
    models_rps_[i].resize(max_size_ * 2 + 1);
    if(i == 0) {
      models_rps_[i][0] = 100; //ms 
    }
    else {
      models_rps_[i][0] = 0;
    }
  }
}
void RpsRecord::add(const CurRpsProto& request) {
  float interval = request.interval();
  uint32_t n = request.n();
  models_rps_[0].push_back(interval);
  end_ += 1;
  for (uint i = 0; i < n; i++) {
    uint32_t id = models_id_[request.model_rps(i).model()];
    models_rps_[id][end_] = request.model_rps(i).rps();
  }
  len_++;
  if(len_ > max_size_) {
    begin_ ++;
  }
  if(begin_ >= max_size_) {
    int n = models_id_.size();
    for (uint i = 0; i <= n; i++) {
      for (uint j = 0; j < len_; j++) {
        models_rps_[i][j] = models_rps_[i][j + begin_];
        models_rps_[i][j + begin_] = 0;
      }
    }
    begin_ = 0;
    end_ = len_ - 1;
  }
    //return true;
}
std::vector<float> RpsRecord::getRecord() {
  LOG(INFO) << "[---GetRecord start---]";
  std::vector<float> ret;
  float total = 0.0;
  for (uint i = begin_; i <= end_; i++) {
    total += models_rps_[0][i];
  }
  LOG(INFO) << "[---got total---]" << total;
  uint n = models_id_.size();
  for (uint i = 1; i <= n; i++) {
    float mean = 0.0, std = 0.0;
    for (uint j = begin_; j <= end_; j++) {
      mean += models_rps_[i][j] * models_rps_[0][j];
    }
    mean /= total;
    for (uint j = begin_; j <= end_; j++) {
      std += models_rps_[0][i] * sqr(models_rps_[i][j] / models_rps_[0][i] - mean);
    }
    std /= total;
    std = std::sqrt(std);
    ret.push_back(mean + std);
  }
  return ret;
}

void QuerySplit::init(int n) {
  workloads_.resize(n);
}
void QuerySplit::addModel(ModelSession model, float lat) {
  models_.push_back(model);
  latencys_.push_back(lat);
}
void QuerySplit::updateLatencys(std::vector<uint32_t> latencys) {
  for (int i = 0; i < models_.size(); i++) {
    last_latencys_[i] = latencys_[i];
    latencys_[i] = latencys[i];
  }
}

void QuerySplit::updateWorkloads(std::vector<float> workloads) {
  for (int i = 0; i < models_.size(); i++) {
    workloads_[i] = workloads[i];
  }
}
std::vector<ModelSession> QuerySplit::constructSplit(std::vector<uint32_t> latencys) {
  std::vector<ModelSession> ret;
  for (int i = 0; i < models_.size(); i++) {
    ModelSession tmp = models_[i];
    tmp.set_latency_sla(latencys[i]);
    ret.push_back(tmp);
  }
  return ret;
} 
std::vector<ModelSession> QuerySplit::last_subscribe_models() {
  std::vector<ModelSession> ret;
  for (int i = 0; i < models_.size(); i++) {
    ModelSession tmp = models_[i];
    tmp.set_latency_sla(last_latencys_[i]);
    ret.push_back(tmp);
  }
  return ret;
}
  
std::vector<ModelSession> QuerySplit::cur_subscribe_models() {
  std::vector<ModelSession> ret;
  for (int i = 0; i < models_.size(); i++) {
    ModelSession tmp = models_[i];
    tmp.set_latency_sla(latencys_[i]);
    ret.push_back(tmp);
  }
  return ret;
}
std::vector<float> QuerySplit::cur_workloads() {
  return workloads_;
}

float ComplexQuery::structure(int n) {  
  LOG(INFO) << "[---Structure---]"; 
  for (int i = 0; i <= n; i++) {
    depth_.push_back(0);
    if(i != 0 && degrees_[i] == 0) {
      redges_[i].push_back(0);
      edges_[0].push_back(i);
      degrees_[i] ++;
    }
  }
  LOG(INFO) << "[---Data---degrees_---]";
  for (int i = 0; i <= n; i++) {
    LOG(INFO) << "[---deg["<<i<<"]---]" << degrees_[i];
  }
  LOG(INFO) << "[---start bfs---]";
  int l = -1, r = 0;
  std::vector<uint32_t> deg(degrees_);
  node_.resize(n + 1);
  node_[0] = 0;
  uint32_t maxn = 0;
  while(l != r) {
    uint32_t x = node_[++l];
    LOG(INFO) << "[---bfs node_id---]" << x;
    maxn = std::max(depth_[x], maxn);
    for (int i = 0; i < edges_[x].size(); i++) {
      uint32_t y = edges_[x][i];
      deg[y] --;
      if(deg[y] == 0) {
        r++;
        depth_[y] = std::max(depth_[y], depth_[x] + 1); 
        node_[r] = y;
      }
    }
  }
  LOG(INFO) << "[---construct layers_---]"; 
  for (uint i = 0; i <= maxn; i++) {
    layers_.push_back(std::vector<uint32_t>());
  }
  for (uint i = 0; i <= n; i++) {
    layers_[depth_[i]].push_back(i);
  }
  diameter_ = maxn;
  LOG(INFO) << "[---diameter_---]" << diameter_; 
}
float ComplexQuery::gpu_need(std::vector<ModelSession> sess, std::vector<float> rps) {
  float ret = 0.0, min_float = 1e-7, max_float = 1e7;
  for (int i = 0; i < sess.size(); i++) {
    float throughput = max_throughputs_[i + 1][sess[i].latency_sla()].first;
    if(throughput <= min_float) {
      ret += max_float;
    }
    else {
      ret += rps[i] / throughput;
    }
  }
  return ret;
}
QuerySplit* ComplexQuery::split() {
  LOG(INFO) << "[---complex query split---]";
  float max_float = 10000000.0;
  std::vector<float> cur_rps = rps_record_.getRecord();
  LOG(INFO) << "[---Got rpsRecord---]";
  uint m = latency_ / step_;
  LOG(INFO) << "[---m = lat / step---]" << m;
  std::vector<std::vector<float> > layer_gpus, dp;
  std::vector<std::vector<uint32_t> > last_layer_lats;
  LOG(INFO) << "[---diameter_---]" << diameter_;
  for (uint i = 0; i <= diameter_; i++) {
    dp.push_back(std::vector<float>());
    last_layer_lats.push_back(std::vector<uint32_t>());
    layer_gpus.push_back(std::vector<float>());
    LOG(INFO) << "[---i---]" << i;
    for (uint j = 0; j <= m; j++) {
      LOG(INFO) << "[---j---]" << i;
      last_layer_lats[i].push_back(0);
      if(i == 0) {
        dp[i].push_back(0.0);
      }
      else {
        dp[i].push_back(max_float);
      }
      if(j == 0) {
        layer_gpus[i].push_back(max_float);
        continue;
      }
      if(i == 0) {
        layer_gpus[i].push_back(0.0);
        continue;
      }
      layer_gpus[i].push_back(0.0);
      for (uint k = 0; k < layers_[i].size(); k++) {
        uint node = layers_[i][k];
        LOG(INFO) << "[---node---]" << node;
        LOG(INFO) << "[---throughput---]" << max_throughputs_[node][j].first;
        float gpu = cur_rps[node - 1] / max_throughputs_[node][j].first;
        layer_gpus[i][j] += gpu;
      }
    }
  }
  LOG(INFO) << "[---initialized---]";
  std::vector<uint32_t> split;
  for (uint i = 0; i <= n_; i++) {
    split.push_back(0);
  }
  for (uint i = 1; i <= diameter_; i++) {
    for(uint j = 1; j <= m; j++) {
      for (uint k = 0; k < j; k++) {
        float tmp = dp[i - 1][k] + layer_gpus[i][j - k];
        if(dp[i][j] > tmp) {
          dp[i][j] = tmp;
          last_layer_lats[i][j] = j - k;
        }
      }
      if(dp[i][j] > dp[i][j - 1]) {
        dp[i][j] = dp[i][j - 1];
        last_layer_lats[i][j] = last_layer_lats[i][j - 1];
      }
    }
  }
  LOG(INFO) << "[---dp finished---]";
  uint32_t last_lat = m, cur_lat;
  for (uint i = diameter_; i > 0; i--) {
    cur_lat = last_layer_lats[i][last_lat];
    for (uint j = 0; j < layers_[i].size(); j++) {
      split[layers_[i][j] - 1] = cur_lat;
    }
    last_lat -= cur_lat;
  }
  LOG(INFO) << "[---get latencies finished---]";
  //check: if current split is 10% better than last split
  float n1 = gpu_need(query_split_.cur_subscribe_models(), cur_rps);
  float n2 = gpu_need(query_split_.constructSplit(split), cur_rps);
  LOG(INFO) << "[---get gpu need finished---]";
  query_split_.updateWorkloads(rps_record_.getRecord());
  if(n1 > n2 * 1.1) {
    query_split_.setState(true);
    query_split_.updateLatencys(split);
  }
  else {
    query_split_.setState(false);
  }
  return &query_split_;
}

CtrlStatus ComplexQuery::init(const LoadDependencyProto& request, std::string common_gpu) {
  LOG(INFO) << "complex query init";
  common_gpu_ = common_gpu;
  models_.push_back("");
  n_ = request.n();
  
  int m = request.m();
  ori_latency_ = request.latency();
  for (uint i = 0; i < n_; i++) {
    std::string model = ModelSessionToString(request.models(i));
    model_sessions_.push_back(request.models(i));
    models_id_[model] = i + 1;
    models_.push_back(model);
  }
  edges_.resize(n_ + 1);
  redges_.resize(n_ + 1);

  degrees_.resize(n_ + 1);
  for (uint i = 0; i <= n_; i++) {
    degrees_[i] = 0;
  }
  for (uint i = 0; i < m; i++) {
    std::string model1 = ModelSessionToString(request.edges(i).v1());
    std::string model2 = ModelSessionToString(request.edges(i).v2());
    uint32_t v1 = models_id_[model1], v2 = models_id_[model2];
    edges_[v1].push_back(v2);
    redges_[v2].push_back(v1);
    degrees_[v2] ++;
  }
  step_ = std::max(std::ceil(std::sqrt(1.0 * n_ / 100000000.0) * ori_latency_), 1.0);
  LOG(INFO) << "[---complex query step_ = step_---]" << step_;
  m = ori_latency_ / step_;
  latency_ = step_ * m;
  max_throughputs_.push_back(std::vector<std::pair<float, uint32_t>>());
  for (uint i = 0; i < n_; i++) {
    std::string profile_id = ModelSessionToProfileID(model_sessions_[i]);
    auto profile = ModelDatabase::Singleton().GetModelProfile(common_gpu_, profile_id);
    if (profile == nullptr) {
      // Cannot find model profile
      return CTRL_PROFILER_UNAVALIABLE;
    }
    max_throughputs_.push_back(std::vector<std::pair<float, uint32_t>>());
    max_throughputs_[i + 1].push_back(std::make_pair(0.0, 0));
    for (uint j = 1; j <= m; j++) {
      uint32_t lat = j * step_;
      max_throughputs_[i + 1].push_back(profile->GetMaxThroughput(lat));
    }
  }
  structure(n_);
  query_split_.init(n_);
  for (uint i = 0; i < n_; i++) {
    query_split_.addModel(request.models(i), request.models(i).estimate_latency());
  }
  rps_record_.init(models_id_);
  return CTRL_OK;
}
void ComplexQuery::addRecord(const CurRpsProto& request) {
  rps_record_.add(request);
}
}
}
