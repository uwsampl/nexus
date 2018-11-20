#ifndef NEXUS_APP_COMPLEX_QUERY_APP_H_
#define NEXUS_APP_COMPLEX_QUERY_APP_H_

#include <gflags/gflags.h>

#include "nexus/app/app_base.h"

namespace nexus{
namespace app {

class ComplexQueryApp : public AppBase {
 public:
  /*!
   * \brief Create App for a complex query.
   * \param port Port of socket.
   * \param rpc_port Port of rpc call.
   * \param sch_addr Address of scheduler.
   * \param nthreads Number of threads in frontend.
   */
  ComplexQueryApp(std::string port, std::string rpc_port, std::string sch_addr,
                  size_t nthreads);
  /*!
   * \brief Iinitialize complex query app.
   * \param n Number of models in the complex query.
   * \param m Number of edges in dataflow graph. 
   * \param latency_sla Latency sla requirement of the app.
   */                
  void Initialize(uint n, uint m, uint latency_sla);
  /*!
   * \brief Get model handler and add model to LoadDependency Request.
   * \param n Number of models in the complex query.
   * \param m Number of edges in dataflow graph. 
   * \param latency_sla Latency sla requirement of the app.
   */   
  std::shared_ptr<ModelHandler> AddModel(std::string &framework, std::string &model, uint version,
                       uint estimate_latency, uint estimate_workload, 
                       std::vector<uint32_t> image_size);
  /*!
   * \brief Add edge to LoadDependency Request.
   * \params There is a edge in the dataflow graph from model1 to model2. 
   */                     
  void AddEdge(std::shared_ptr<ModelHandler> model1, 
                        std::shared_ptr<ModelHandler> model2);
  /*!
   * \brief LoadDependencyRequest construction finished.
   */ 
  void LoadDependency();
  /*!
   * \brief Construct an ExecBlock and add it to query processor construction.
   * \param func Function in the ExecBlock.
   * \params variables variables needed (must have gotten value) in the Function.
   */ 
  void AddExecBlock(ExecFunc func, std::vector<std::string> variables);
  /*!
   * \brief Construct queryProcessor according to Exec Blocks added.
   */
  void BuildQueryProcessor();
   /*! \brief Construct RectProto */
  RectProto GetRect(int left, int right, int top, int bottom);
  
 private:
  /*! \brief Number of models */
  size_t n_;
  /*! \brief Number of edges */
  size_t m_;
  /*! \brief Latency required of the complex query*/
  int latency_sla_;
  /*! \brief Map from model session id (lat = 0) to model handler */
  std::unordered_map<std::string, std::shared_ptr<ModelHandler> > models_;
  /*! \brief request for LoadDependency */
  LoadDependencyRequest request_;
  /*! \brief ptoro for LoadDependencyRequest */
  LoadDependencyProto* proto_;
  /*! \brief exec blocks of query processor */
  std::vector<ExecBlock*> exec_blocks_;
};

} //namespace app
} //namespace nexus

#endif // NEXUS_APP_COMPLEX_QUERY_APP_H_
