#ifndef PERCEPTRON_TENSORS_SHUFFLERS_BATCHSHUFFLER_H
#define PERCEPTRON_TENSORS_SHUFFLERS_BATCHSHUFFLER_H

#include "perceptron/common/Common.h"
#include "perceptron/tensors/shufflers/IShuffler.h"
#include "perceptron/tensors/ops/Ops.h"
#include "perceptron/common/utils/MemoryUtils.h"

#include <random>
#include <algorithm>
#include <vector>

namespace perceptron {
namespace tensors {
namespace shufflers {

class BatchShuffler : public IShuffler {
public:
  using IShuffler::get_shuffled;

  BatchShuffler(size_type nrows, size_type batch_size, size_type seed, cudaStream_t stream);
  BatchShuffler(size_type nrows, size_type batch_size, size_type seed);
  BatchShuffler(size_type nrows, size_type batch_size, cudaStream_t stream);
  BatchShuffler(size_type nrows, size_type batch_size);

  void
  shuffle() override;

  TensorOwner2D<float>
  get_shuffled(TensorReadOnly2D<float, false> tensor_to_shuffle,
               cudaStream_t stream) override;

  TensorOwner2D<float>
  get_shuffled(TensorReadOnly2D<float, true> tensor_to_shuffle,
               cudaStream_t stream) override;

  TensorOwner2D<double>
  get_shuffled(TensorReadOnly2D<double, false> tensor_to_shuffle,
               cudaStream_t stream) override;

  TensorOwner2D<double>
  get_shuffled(TensorReadOnly2D<double, true> tensor_to_shuffle,
               cudaStream_t stream) override;

  void
  get_shuffled(TensorReadOnly2D<float, false> tensor_to_shuffle,
               TensorWriteable2D<float> shuffled,
               cudaStream_t stream) override;

  void
  get_shuffled(TensorReadOnly2D<float, true> tensor_to_shuffle,
               TensorWriteable2D<float> shuffled,
               cudaStream_t stream) override;

  void
  get_shuffled(TensorReadOnly2D<double, false> tensor_to_shuffle,
               TensorWriteable2D<double> shuffled,
               cudaStream_t stream) override;

  void
  get_shuffled(TensorReadOnly2D<double, true> tensor_to_shuffle,
               TensorWriteable2D<double> shuffled,
               cudaStream_t stream) override;

private:
  TensorOwner1D<size_type> m_batch_indices_owner;
  std::mt19937 m_g;
  std::vector<size_type> m_indices;
  size_type m_batch_size;
  cudaStream_t m_stream;

  template<typename T, bool trans>
  void
  get_shuffled_impl(TensorReadOnly2D<T, trans> tensor_to_shuffle,
                    TensorWriteable2D<T> shuffled,
                    cudaStream_t stream);

  void
  update_batch_indices();
};

template<typename T, bool trans>
void
BatchShuffler::get_shuffled_impl(TensorReadOnly2D<T, trans> tensor_to_shuffle,
                                 TensorWriteable2D<T> shuffled,
                                 cudaStream_t stream) {
  auto batch_indices_view = m_batch_indices_owner.tensor_view();

  ops::copy_rows_by_indices(tensor_to_shuffle,
                            batch_indices_view.to_read_only(),
                            shuffled,
                            m_stream);
}

} // perceptron
} // tensors
} // shufflers

#endif //PERCEPTRON_TENSORS_SHUFFLERS_BATCHSHUFFLER_H
