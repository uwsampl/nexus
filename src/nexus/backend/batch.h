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
   * \brief construct a Input
   * \param task Task that input data comes from
   * \param arr Input array that contains the input data
   * \param idx Index in the inputs of task
   */
  Input(std::shared_ptr<Task> task, std::shared_ptr<Array> arr, int idx);
  /*! \brief task that input data comes from */
  std::shared_ptr<Task> task;
  /*! \brief input array that contains the data */
  std::shared_ptr<Array> array;
  /*! \brief index among all inputs from the task */
  int index;
};


/*!
 * \brief BatchInput contains a batch of inputs.
 */
class BatchInput {
 public:
  /*!
   * \brief construct a BatchInput.
   * \param batch_id Batch index
   * \param batch_size Number of inputs in the batch
   * \param arr Array with a continuous GPU buffer for holding all input data
   */
  BatchInput(uint64_t batch_id, ArrayPtr arr);
  /*!
   * \brief
   * \param input A single input
   */
  void Append(std::shared_ptr<Input> input);
  /*! \brief get batch id */
  uint64_t batch_id() const { return batch_id_; }
  /*! \brief get the batch size */
  size_t batch_size() const { return inputs_.size(); }
  /*! \brief get the array */
  ArrayPtr array() const { return array_; }
  /*! \brief get all inputs added to this batch */
  std::vector<std::shared_ptr<Input> > inputs() const { return inputs_; }

 private:
  /*! \brief batch id */
  uint64_t batch_id_;
  /*! \brief array that holds batch input data */
  ArrayPtr array_;
  /*! \brief inputs in the batch */
  std::vector<std::shared_ptr<Input> > inputs_;
  /*! \brief write pointer to array_ */
  float* write_pt_;
  /*! \brief number of floats written in the array_ */
  size_t num_elements_;
};

class BatchOutput; // forward declare

/*!
 * \brief Output contains the data of a single output.
 */
class Output {
 public:
  /*! \brief get batch id */
  uint64_t batch_id() const;
  /*!
   * \brief get all outputs
   * \return a vector of arrays
   */
  std::vector<ArrayPtr> GetOutputs() const { return outputs_; }
  /*!
   * \brief get i-th output data
   * \return an array
   */
  ArrayPtr GetOutput(int i) const;
  /*!
   * \brief get output data with name
   * \return an array
   */
  ArrayPtr GetOutput(const std::string& name) const;
  /*!
   * \brief finish the use of this output
   * \return whether all outputs in this batch are finished
   */
  bool Finish() const;

 private:
  /*!
   * \brief private constructor of Return
   * \param output_batch Pointer to the BatchOutput class
   * \param output A vector of pointers to the output buffer
   */
  Output(BatchOutput* batch_output, const std::vector<ArrayPtr>& outputs) :
      batch_output_(batch_output),
      outputs_(outputs) {}
  /*!
   * \brief private constructor of Return
   * \param output_batch Pointer to the BatchOutput class
   * \param outputs A vector of pointers to output buffers
   * \param names A vector of names of output buffers
   */
  Output(BatchOutput* batch_output, const std::vector<ArrayPtr>& outputs,
         const std::vector<std::string> names) :
      batch_output_(batch_output),
      outputs_(outputs),
      output_names_(names) {}
  /*! \brief friend class of BatchOutput to allow it create Output */
  friend class BatchOutput;

 private:
  /*! \brief Pointer to BatchOutput */
  BatchOutput* batch_output_;
  /*! \brief Pointers to output data */
  std::vector<ArrayPtr> outputs_;
  /*! \brief Names of output data */
  std::vector<std::string> output_names_;
};

/*!
 * \brief BatchOutput contains a batch of outputs.
 */
class BatchOutput {
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
   * \brief set the output buffer and batch size
   * \param outputs A vector of batch output arrays
   * \param slices Slices for each batch output to single outputs
   */
  void SetOutputBatch(const std::vector<ArrayPtr>& outputs,
                      const std::vector<Slice>& slices);
  /*!
   * \brief set the output buffer and batch size
   * \param outputs A vector of batch output batch arrays
   * \param slices Slices for each batch output to single outputs
   * \param names Names of output buffers
   */
  void SetOutputBatch(const std::vector<ArrayPtr>& outputs,
                      const std::vector<Slice>& slices,
                      const std::vector<std::string>& names);
  /*!
   * \brief set the output buffer and batch size
   * \param outputs A vector of raw pointers to batch output data
   * \param slices Slices for each batch output to single outputs
   * \param device Device where all buffers in the outputs is located
   */
  void SetOutputBatch(const std::vector<float*>& outputs,
                      const std::vector<Slice>& slices, Device* device);
  /*!
   * \brief get all single outputs in the batch
   * \return A vector of Outputs
   */
  std::vector<std::unique_ptr<Output> > GetOutputs();
  /*!
   * \brief finish the use of one single output
   * \return whether all outputs in this batch are finished
   */
  bool Finish();

 private:
  /*! \brief Batch index */
  uint64_t batch_id_;
  /*! \brief Batch size */
  size_t batch_size_;
  /*! \brief Arrays of all batch outputs */
  std::vector<ArrayPtr> outputs_;
  /*! \brief Slices for batch outputs */
  std::vector<Slice> slices_;
  /*! \brief Names of batch outputs */
  std::vector<std::string> output_names_;
  /*! \brief Number of outputs that are finished of processing */
  std::atomic_uint finished_;
};

} // namespace backend
} // namespace nexus

#endif // NEXUS_BACKEND_BATCH_H_
