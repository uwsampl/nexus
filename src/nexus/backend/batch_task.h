#ifndef NEXUS_BACKEND_BATCH_TASK_H_
#define NEXUS_BACKEND_BATCH_TASK_H_

#include <memory>

#include "nexus/backend/slice.h"
#include "nexus/backend/task.h"

namespace nexus {
namespace backend {

/*!
 * \brief BatchTask holds a batch of inputs and outputs, and is used for
 * batch forwarding through a DNN model.
 */
class BatchTask {
 public:
  /*!
   * \brief Construct a batch task.
   * \param max_batch Max batch size.
   */
  BatchTask(uint32_t max_batch);
  /*!
   * \brief Set batch id
   * \param batch_id Batch id
  */
  inline void set_batch_id(uint64_t batch_id) { batch_id_ = batch_id; }
  /*! \brief Return batch id */
  inline uint64_t batch_id() const { return batch_id_; }
  /*! \brief Return batch size */
  inline uint32_t batch_size() const { return inputs_.size(); }
  /*! \brief Return max batch size */
  inline uint32_t max_batch() const { return max_batch_; }
  /*!
   * \brief Set input array for holding the batch input data.
   * \param arr Array pointer.
   */
  void SetInputArray(ArrayPtr arr);
  /*!
   * \brief Create input arrays to hold the batch input data.
   * \param input_size Number of elements of a single input.
   * \param device Device for allocation of input array.
   */
  void CreateInputArray(size_t input_size, Device* device);
  /*!
   * \brief Set output arrays for holding the batch output results.
   * \param arrays Map from name to arrays.
   */
  void SetOutputArrays(const std::unordered_map<std::string, ArrayPtr>& arrays);
  /*!
   * \brief Create output arrays to hold the batch output results.
   * \param sizes Map from name to output sizes in float for a single batch.
   * \param device Device for allocation of output arrays.
   */
  void CreateOutputArrays(const std::unordered_map<std::string, size_t>& sizes,
                          Device* device);
  /*! \brief Return input batch array */
  inline ArrayPtr GetInputArray() const { return input_array_; }
  /*!
   * \brief Get the output batch array given name.
   * \param name Name of array.
   * \return Array corresponding to the name
   */
  ArrayPtr GetOutputArray(const std::string& name) const;
  /*!
   * \brief Append a new input into the batch input.
   * \param input A single input.
   */
  void AppendInput(std::shared_ptr<Input> input, std::shared_ptr<Task> task);
  /*!
   * \brief Slice the batch output into individual outputs.
   * \param slices Slices for all arrays.
   */
  void SliceOutputBatch(const std::unordered_map<std::string, Slice>& slices);
  /*! \brief Get all individual inputs in the batch. */
  inline const std::vector<std::shared_ptr<Input> >& inputs() const {
    return inputs_;
  }
  /*! \brief Get all individual outputs in the batch. */
  inline const std::vector<std::shared_ptr<Output> >& outputs() const {
    return outputs_;
  }
  /*! \brief Set individual outputs. */
  void set_outputs(const std::vector<std::shared_ptr<Output> >& outputs);
  /*! \brief Get all tasks in the batch. */
  inline const std::vector<std::shared_ptr<Task> >& tasks() const {
    return tasks_;
  }

 private:
  /*! \brief Batch ID. */
  uint64_t batch_id_;
  /*! \brief Max batch size. */
  uint32_t max_batch_;
  /*! \brief Array that holds batch input data. */
  ArrayPtr input_array_;
  /*! \brief Write pointer to input_array_. */
  float* input_write_pt_;
  /*! \brief Number of floats added in the input_array_. */
  size_t input_nfloats_;
  /*! \brief Map from name to array. */
  std::unordered_map<std::string, ArrayPtr> output_arrays_;
  /*! \brief Tasks in the batch */
  std::vector<std::shared_ptr<Task> > tasks_;
  /*! \brief Individual inputs in the batch */
  std::vector<std::shared_ptr<Input> > inputs_;
  /*! \brief Individual outputs in the batch */
  std::vector<std::shared_ptr<Output> > outputs_;
};

} // namespace backend
} // namespace nexus

#endif // NEXUS_BACKEND_BATCH_TASK_H_
