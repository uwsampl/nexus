#ifndef NEXUS_BACKEND_BATCH_H_
#define NEXUS_BACKEND_BATCH_H_

#include <atomic>
#include <condition_variable>
#include <memory>

#include "nexus/backend/slice.h"
#include "nexus/common/block_queue.h"
#include "nexus/common/data_type.h"

namespace nexus {
namespace backend {

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
  Input(std::shared_ptr<Task> task, std::shared_ptr<Array> arr, int idx);
  /*! \brief Task that input belongs to. */
  std::shared_ptr<Task> task;
  /*! \brief Input array that contains the data. */
  std::shared_ptr<Array> array;
  /*! \brief Index within all inputs belonging to task. */
  int index;
};


/*!
 * \brief BatchInput contains a batch of inputs.
 */
class BatchInput {
 public:
  /*!
   * \brief Construct a BatchInput.
   * \param batch_id Batch index
   * \param batch_size Number of inputs in the batch
   * \param arr Array with a continuous GPU buffer for holding all input data
   */
  BatchInput(uint64_t batch_id, ArrayPtr arr);
  /*!
   * \brief Append a new input into the batch input.
   * \param input A single input
   */
  void Append(std::shared_ptr<Input> input);
  /*! \brief Get batch id */
  uint64_t batch_id() const { return batch_id_; }
  /*! \brief Get the batch size */
  size_t batch_size() const { return inputs_.size(); }
  /*! \brief Get the array */
  ArrayPtr array() const { return array_; }
  /*! \brief Get all inputs added to this batch */
  std::vector<std::shared_ptr<Input> > inputs() const { return inputs_; }

 private:
  /*! \brief Batch id */
  uint64_t batch_id_;
  /*! \brief Array that holds batch input data */
  ArrayPtr array_;
  /*! \brief Inputs in the batch */
  std::vector<std::shared_ptr<Input> > inputs_;
  /*! \brief Write pointer to array_ */
  float* write_pt_;
  /*! \brief Number of floats written in the array_ */
  size_t num_elements_;
};

class BatchOutput; // forward declare

/*!
 * \brief Output contains the data of a single output.
 */
class Output {
 public:
  /*! \brief Return batch id */
  uint64_t batch_id() const;
  /*!
   * \brief Get the output array given name.
   * \param name Name of array.
   * \return Array pointer corresponding to the name
   */
  ArrayPtr GetArray(const std::string& name) const;
  /*!
   * \brief Get the output array given name.
   * \param name Name of array.
   * \return Array pointer corresponding to the name
   */
  ArrayPtr operator[](const std::string& name) const;

 private:
  /*!
   * \brief Construct an Output.
   * \param output_batch Pointer to the BatchOutput class.
   * \param arrays Map from name to arrays.
   */
  Output(std::shared_ptr<BatchOutput> batch_output,
         const std::unordered_map<std::string, ArrayPtr>& arrays);

  friend class BatchOutput;

  /*! \brief Pointer to BatchOutput. */
  std::shared_ptr<BatchOutput> batch_output_;
  /*! \brief Map from array name to array. */
  std::unordered_map<std::string, ArrayPtr> arrays_;
};

/*!
 * \brief BatchOutput contains a batch of outputs.
 */
class BatchOutput : public std::enable_shared_from_this<BatchOutput> {
 public:
  /*!
   * \brief construct a BatchOutput
   * \param batch_id Batch index
   * \param batch_size Batch size
   */
  BatchOutput(uint64_t batch_id, size_t batch_size);
  /*! \brief get batch id */
  uint64_t batch_id() const { return batch_id_; }
  /*! \brief get batch size */
  size_t batch_size() const { return batch_size_; }
  /*!
   * \brief Set arrays for holding the batch output results.
   * \param arrays Map from name to arrays.
   */
  void SetArrays(const std::unordered_map<std::string, ArrayPtr>& arrays);
  /*!
   * \brief Create arrays to hold the batch output results.
   * \param sizes Map from name to output sizes in float for a single batch.
   * \param device Device for allocation of output arrays.
   */
  void CreateArrays(const std::unordered_map<std::string, size_t>& sizes,
                    Device* device);
  /*!
   * \brief Get the batch output array given name.
   * \param name Name of array.
   * \return Array pointer corresponding to the name
   */
  ArrayPtr GetArray(const std::string& name);
  /*!
   * \brief Slice the batch output into individual outputs.
   * \param slices Slices for all arrays.
   */
  void SliceBatch(const std::unordered_map<std::string, Slice>& slices);
  /*!
   * \brief Get all individual outputs in the batch.
   * \return A vector of Output.
   */
  std::vector<std::unique_ptr<Output> > GetOutputs();

 private:
  /*! \brief Batch index */
  uint64_t batch_id_;
  /*! \brief Batch size */
  size_t batch_size_;
  /*! \brief Map from name to array */
  std::unordered_map<std::string, ArrayPtr> arrays_;
  /*! \brief Map from name to slices */
  std::unordered_map<std::string, Slice> slices_;
};

} // namespace backend
} // namespace nexus

#endif // NEXUS_BACKEND_BATCH_H_
