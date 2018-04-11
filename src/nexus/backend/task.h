#ifndef NEXUS_BACKEND_TASK_H_
#define NEXUS_BACKEND_TASK_H_

#include <atomic>
#include <chrono>
#include <memory>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include "nexus/backend/batch.h"
#include "nexus/common/block_queue.h"
#include "nexus/common/connection.h"
#include "nexus/common/model_def.h"
#include "nexus/proto/nnquery.pb.h"
#include "nexus/proto/control.pb.h"

namespace nexus {
namespace backend {

class ModelInstance;

/*! \brief Stage indicates the context processing stage */
enum Stage {
  /* !\brief Task at the pre-processing stage */
  kPreprocess = 0,
  /* !\brief Task at the forwarding model stage */
  kForward,
  /* !\brief Task at the post-processing stage */
  kPostprocess,
};

class Task : public DeadlineItem {
 public:
  Task();

  Task(std::shared_ptr<Connection> conn);

  void DecodeQuery(std::shared_ptr<Message> message);
  /*!
   * \brief Add output at index location
   * \param index Index of the output
   * \param output Output content
   * \return whether all output has been filled in
   */
  bool AddOutput(int index, std::unique_ptr<Output> output);
  /*!
   * \brief Add virtual output at index location due to error such as timeout
   * \param index Index of the virtual output
   * \return whether all output has been filled in
   */
  bool AddVirtualOutput(int index);

  std::shared_ptr<Connection> connection;
  QueryProto query;
  QueryResultProto result;
  /*! \brief Model instance to execute for the task */
  std::shared_ptr<ModelInstance> model;
  /*! \brief Current task processing stage */
  volatile Stage stage;
  /*! \brief Outputs of the context */
  std::vector<std::unique_ptr<Output> > outputs;
  /*! \brief Number of outputs that has been filled in */
  uint32_t filled_outputs;
  /*! \brief Attributes that needs to be kept during the task */
  YAML::Node attrs;
  /*! \brief Timer that counts the time spent in each stage */
  Timer timer;
};

} // namespace backend
} // namespace nexus

#endif // NEXUS_BACKEND_TASK_H_
