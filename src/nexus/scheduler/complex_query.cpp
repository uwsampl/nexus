#include "nexus/scheduler/complex_query.h"
namespace nexus {
namespace scheduler {
void RpsRecord::init(std::map<std::string, uint32_t> models_id, double split) {
  this->models_id = models_id;
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
void RpsRecord::add(const CurRpsProto& request) {
  double interval = request.interval();
  uint32_t n = request.n();
  models_rps[0].push_back(interval);
  end += 1;
  for (int i = 0; i < n; i++) {
    uint32_t id = models_id[request.model_rps(i).model()];
    models_rps[id][end] = request.model_rps(i).rps();
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
std::vector<double> RpsRecord::getRecord() {
  std::vector<double> ret;
  double total = 0.0;
  for (int i = begin; i <= end; i++) {
    total += models_rps[0][i];
  }
  int n = models_id.size();
  for (int i = 1; i <= n; i++) {
    double mean = 0.0, std = 0.0;
    for (int j = begin; j <= end; j++) {
      mean += models_rps[i][j] * models_rps[0][j];
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

void QuerySplit::addModel(ModelSession model, double lat) {
  models_.push_back(model);
  latencys_.push_back(lat);
}
void QuerySplit::updateLatencys(std::vector<uint32_t> latencys) {
  for (int i = 0; i < models_.size(); i++) {
    last_latencys_[i] = latencys_[i];
    latencys_[i] = latencys[i];
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

double ComplexQuery::structure(int n) {   
  for (int i = 0; i <= n; i++) {
    depth_.push_back(0);
    if(i != 0 && degrees_[i] == 0) {
      redges_[i].push_back(0);
      edges_[0].push_back(i);
      degrees_[i] ++;
    }
  }
  int l = -1, r = 0;
  std::vector<uint32_t> deg(degrees_);
  node_.resize(n + 1);
  node_[0] = 0;
  uint32_t maxn = 0;
  while(l != r) {
    uint32_t x = node_[++l];
    maxn = std::max(depth_[x], maxn);
    for (int i = 0; i < edges_[x].size(); i++) {
      uint32_t y = edges_[x][i];
      deg[y] --;
      if(deg[y] == 0) {
        r++;
        depth_[y] = std::max(depth_[y], depth_[x] + 1);
         
        node_.push_back(y);
      }
    }
  }
  return latency_ / maxn;
}
double ComplexQuery::gpu_need(std::vector<ModelSession> sess, std::vector<double> rps) {
  double ret = 0;
  for (int i = 0; i < sess.size(); i++) {
    double throughput = max_throughputs_[i + 1][sess[i].latency_sla()].first;
    ret += rps[i] / throughput;
  }
  return ret;
}
QuerySplit* ComplexQuery::split() {
  double max_float = 10000000.0;
  std::vector<double>alpha = rps.getRecord();
  int m = latency_ / step_;
  std::vector<std::vector<double> > f, g, last_batch, last_lat;
  std::vector<int> q;
  std::vector<bool> visit;
  std::vector<double> lats;
  std::vector<uint32_t> split;
  for (int i = 0; i <= n; i++) {
    visit.push_back(false);
    split.push_back(0.0);
    f.push_back(std::vector<double>());
    g.push_back(std::vector<double>());
    //last_batch.push_back(std::vector<double>());
    last_lat.push_back(std::vector<double>());
    
    if(i == 0) {
      for (int j = 0; j <= m; j++) {
        f[i].push_back(0.0);
      }        
    }
    else {
      for (int j = 0; j <= m; j++) {
        f[i].push_back(max_float);
      }
    }
    for (int j = 0; j <= m; j++) {
      g[i].push_back(0.0);
      //last_batch[i].push_back(0.0);
      lats.push_back(0.0);
    }
  }
  for (int i = 1; i <= n; i++) {
    int now = node_[i];
    for(int j = 0; j <= m; j++) {
      for (int k = 0; k < redges_[now].size(); k++) {
        int x = redges_[now][k];
        g[now][j] = std::max(g[now][j], f[x][j]);
          
      }
      for (int k = 0; k <= m; k++) {
        double throughput = max_throughputs_[now][j - k].first;
        uint32_t batch = max_throughputs_[now][j - k].second;
        double ngpu = throughput > 0 ? alpha[now] / throughput : max_float;
        double tmp = g[now][k] + ngpu;
        if(f[now][j] > tmp) {
          f[now][j] = tmp;
          //last_batch[now][j] = batch;
          last_lat[now][j] = j - k;
        }
      }
      if(j > 0 && f[now][j] > f[now][j - 1]) {
        f[now][j] = f[now][j - 1];
        //last_batch[now][j] = last_batch[now][j - 1];
        last_lat[now][j] = last_lat[now][j - 1];
      }
    }
  }
  int l = -1, r = -1;
  for (int i = 1; i <= n; i++) {
    if (edges_[i].size() == 0) {
      r ++;
      q.push_back(i);
      lats.push_back(m);
      visit[i] = true;
    }
  }
  
  while(l < r) {
    l ++;
    int now = q[l], lat = lats[l];
    if(now == 0) {
      break;
    }
    split[now - 1] = last_lat[now][lat];
    for (int i = 0; i < redges_[now].size(); i++) {
      int x = redges_[now][i];
      if(visit[x] == false) {
        visit[x] = true;
        r ++;
        q.push_back(x);
        lats.push_back(lat - split[now]);
      }
    }
  }
  //check: if current split is 10% better than last split
  double n1 = gpu_need(qs.cur_subscribe_models(), alpha);
  double n2 = gpu_need(qs.constructSplit(split), alpha);
  if(n1 > n2 * 1.1) {
    qs.setState(true);
    qs.updateLatencys(split);
  }
  else {
    qs.setState(false);
  }
  return &qs;
}

CtrlStatus ComplexQuery::init(const LoadDependencyProto& request, std::string common_gpu) {
  common_gpu_ = common_gpu;
  models_.push_back("");
  n = request.n();
  int m = request.m();
  latency = request.latency();
  for (int i = 0; i < n; i++) {
    std::string model = ModelSessionToString(request.models(i));
    model_sessions_.push_back(request.models(i));
    models_id_[model] = i + 1;
    models_.push_back(model);
  }
  edges_.resize(n + 1);
  redges_.resize(n + 1);

  degrees_.resize(n + 1);
  for (int i = 0; i <= n; i++) {
    degrees_[i] = 0;
  }
  for (int i = 0; i < m; i++) {
    std::string model1 = ModelSessionToString(request.edges(i).v1());
    std::string model2 = ModelSessionToString(request.edges(i).v2());
    int v1 = models_id_[model1], v2 = models_id_[model2];
    edges_[v1].push_back(v2);
    redges_[v2].push_back(v1);
    degrees_[v2] ++;
  }
  step_ = std::max(std::ceil(100000000.0 / latency / latency), 1.0);
  m = latency / step_;
  latency_ = step_ * m;
  max_throughputs_.push_back(std::vector<std::pair<float, uint32_t>>());
  for (int i = 0; i < n; i++) {
    std::string profile_id = ModelSessionToProfileID(model_sessions_[i]);
    auto profile = ModelDatabase::Singleton().GetModelProfile(common_gpu_, profile_id);
    if (profile == nullptr) {
      // Cannot find model profile
      return CTRL_PROFILER_UNAVALIABLE;
    }
    max_throughputs_.push_back(std::vector<std::pair<float, uint32_t>>());
    max_throughputs_[i].push_back(std::make_pair(0.0, 0));
    for (int j = 1; j <= m; j++) {
      int lat = j * step_;
      max_throughputs_[i].push_back(profile->GetMaxThroughput(lat));
    }
  }
  double split = structure(n);
  for (int i = 0; i < n; i++) {
    qs.addModel(request.models(i), split);
  }
  rps.init(models_id_, split);
  return CTRL_OK;
}
void ComplexQuery::addRecord(const CurRpsProto& request) {
  rps.add(request);
}
}
}
