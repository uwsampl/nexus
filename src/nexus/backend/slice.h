#ifndef NEXUS_BACKEND_SLICE_H_
#define NEXUS_BACKEND_SLICE_H_

#include <cstring>
#include <vector>

namespace nexus {
namespace backend {

class Slice {
 public:
  /*!
   * \brief construct a slice with equal size splits.
   * \param nsplits Number of splits
   * \param nfloats Number of floats in a slice
   */
  Slice(size_t nsplits, size_t nfloats);
  /*!
   * \brief construct a slice with varied sizes.
   * \param nfloats A vector of number of floats
   * \param multiplier Multiplier to number of floats
   */
  Slice(std::vector<size_t> nfloats, size_t multiplier = 1);
  Slice(std::vector<float> nfloats, size_t multiplier = 1);
  /*!
   * \brief construct a slice with varied sizes.
   * \param nsplits Number of splits
   * \param nfloats An array of number of floats
   * \param multiplier Multiplier to number of floats
   */
  Slice(size_t nslices, float* nfloats, size_t multiplier = 1);
  /*!
   * \brief get the offset for idx-th slice
   * \param idx Index of the slice
   * \return offset of idx-th slice
   */
  size_t offset(int idx) const;
  /*!
   * \brief get the number of floats in slice idx
   * \param idx Index of the split
   * \return number of floats
   */
  size_t num_elements(int idx) const;
  /*! \brief get number of splits */
  size_t num_splits() const { return offsets_.size(); }
  /*! \brief total number of floats in the buffer */
  size_t total_elements() const { return total_elements_; }

 private:
  bool equal_split_;
  size_t equal_slice_size_;
  std::vector<size_t> slice_sizes_;
  std::vector<size_t> offsets_;
  size_t total_elements_;
};

} // namespace backend
} // namespace nexus

#endif // NEXUS_BACKEND_SLICE_H_
