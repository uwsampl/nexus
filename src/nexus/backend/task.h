#ifndef NEXUS_BACKEND_TASK_H_
#define NEXUS_BACKEND_TASK_H_

#include <atomic>
#include <chrono>
#include <memory>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include "nexus/common/block_queue.h"
#include "nexus/common/connection.h"
#include "nexus/common/data_type.h"
#include "nexus/proto/nnquery.pb.h"
#include "nexus/proto/control.pb.h"

namespace nexus {
namespace backend {

class ModelInstance;
class Task;

/*!
 * \brief Input contains input data of a single input and related information
 *   to neural networks.
 */
class Input : public DeadlineItem {
 public:
  /*!
   * \brief Construct a Input
   * \param task Task that input belongs to.
   * \param arr Input array that contains the input data
   * \param idx Index in the inputs of task
   */
  Input(ArrayPtr arr, std::shared_ptr<Task> task, int idx);
  /*! \brief Input array that contains the data. */
  std::shared_ptr<Array> array;
  /*! \brief Task that input belongs to. */
  std::shared_ptr<Task> task;
  /*! \brief Index in the input vector of task. */
  int index_in_task;
};

/*!
 * \brief Output contains the data of a single output.
 */
class Output {
 public:
  /*!
   * \brief Construct an Output.
   * \param output_batch Pointer to the BatchOutput class.
   * \param arrays Map from name to arrays.
   */
  Output(const std::unordered_map<std::string, ArrayPtr>& arrs,
         std::shared_ptr<Task> task, int idx);
  /*! \brief Map from array name to array. */
  std::unordered_map<std::string, ArrayPtr> arrays;
  /*! \brief Task that input belongs to. */
  std::shared_ptr<Task> task;
  /*! \brief Index in the output vector of task. */
  int index_in_task;
};

/*! \brief Stage indicates the context processing stage */
enum Stage {
  /* !\brief Task at the pre-processing stage */
  kPreprocess = 0,
  /* !\brief Task at the forwarding model stage */
  kForward,
  /* !\brief Task at the post-processing stage */
  kPostprocess,
};

class Task : public DeadlineItem, public std::enable_shared_from_this<Task> {
 public:
  Task();

  Task(std::shared_ptr<Connection> conn);

  void DecodeQuery(std::shared_ptr<Message> message);
  
  void AppendInput(ArrayPtr arr);
  /*!
   * \brief Add output at index location
   * \param index Index of the output
   * \param output Output content
   * \return whether all output has been filled in
   */
  bool AddOutput(std::shared_ptr<Output> output);
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
  std::vector<std::shared_ptr<Input> > inputs;
  /*! \brief Outputs of the context */
  std::vector<std::shared_ptr<Output> > outputs;
  /*! \brief Number of outputs that has been filled in */
  std::atomic<uint32_t> filled_outputs;
  /*! \brief Attributes that needs to be kept during the task */
  YAML::Node attrs;
  /*! \brief Timer that counts the time spent in each stage */
  Timer timer;
};

} // namespace backend
} // namespace nexus

#endif // NEXUS_BACKEND_TASK_H_
