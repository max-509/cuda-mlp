#ifndef PERCEPTRON_TENSORS_SHUFFLERS_BATCHSHUFFLER_H
#define PERCEPTRON_TENSORS_SHUFFLERS_BATCHSHUFFLER_H

#include "perceptron/common/Common.h"
#include "perceptron/tensors/shufflers/IShuffler.h"
#include "perceptron/tensors/ops/Ops.h"
#include "perceptron/common/utils/MemoryUtils.h"

#include <random>
#include <algorithm>
#include <vector>
#include <tuple>

namespace perceptron {
namespace tensors {
namespace shufflers {

class BatchShuffler : public IShuffler {
public:
  using IShuffler::get_shuffled;
  using IShuffler::shuffle;

  BatchShuffler(size_type dataset_size, size_type batch_size, size_type seed, cudaStream_t stream);
  BatchShuffler(size_type dataset_size, size_type batch_size, size_type seed);
  BatchShuffler(size_type dataset_size, size_type batch_size, cudaStream_t stream);
  BatchShuffler(size_type dataset_size, size_type batch_size);

  void
  shuffle(cudaStream_t stream) override;

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

  void
  train_test_split(TensorReadOnly2D<float, false> tensor_to_split,
                   TensorWriteable2D<float> train,
                   TensorWriteable2D<float> test,
                   cudaStream_t stream = nullptr);

  void
  train_test_split(TensorReadOnly2D<float, true> tensor_to_split,
                   TensorWriteable2D<float> train,
                   TensorWriteable2D<float> test,
                   cudaStream_t stream = nullptr);

  void
  train_test_split(TensorReadOnly2D<double, false> tensor_to_split,
                   TensorWriteable2D<double> train,
                   TensorWriteable2D<double> test,
                   cudaStream_t stream = nullptr);

  void
  train_test_split(TensorReadOnly2D<double, true> tensor_to_split,
                   TensorWriteable2D<double> train,
                   TensorWriteable2D<double> test,
                   cudaStream_t stream = nullptr);

  std::tuple<TensorOwner2D<float>, TensorOwner2D<float>>
  train_test_split(TensorReadOnly2D<float, false> tensor_to_split,
                   cudaStream_t stream = nullptr);

  std::tuple<TensorOwner2D<float>, TensorOwner2D<float>>
  train_test_split(TensorReadOnly2D<float, true> tensor_to_split,
                   cudaStream_t stream = nullptr);

  std::tuple<TensorOwner2D<double>, TensorOwner2D<double>>
  train_test_split(TensorReadOnly2D<double, false> tensor_to_split,
                   cudaStream_t stream = nullptr);

  std::tuple<TensorOwner2D<double>, TensorOwner2D<double>>
  train_test_split(TensorReadOnly2D<double, true> tensor_to_split,
                   cudaStream_t stream = nullptr);

private:
  TensorOwner1D<size_type> m_batch_indices_owner;
  std::mt19937 m_g;
  std::vector<size_type> m_indices;
  size_type m_batch_size;

  template<typename T, bool trans>
  void
  get_shuffled_impl(TensorReadOnly2D<T, trans> tensor_to_shuffle,
                    TensorWriteable2D<T> shuffled,
                    cudaStream_t stream);

  template<typename T, bool trans>
  void
  train_test_split_impl(TensorReadOnly2D<T, trans> tensor_to_split,
                        TensorWriteable2D<T> train,
                        TensorWriteable2D<T> test,
                        cudaStream_t stream);

  void
  update_batch_indices(cudaStream_t stream);

  void
  update_batch_indices(TensorWriteable1D<size_type> batch_indices_tensor,
                       size_type *indices_data,
                       size_type indices_size,
                       cudaStream_t stream);
};

template<typename T, bool trans>
void
BatchShuffler::get_shuffled_impl(TensorReadOnly2D<T, trans> tensor_to_shuffle,
                                 TensorWriteable2D<T> shuffled,
                                 cudaStream_t stream) {
  ops::copy_rows_by_indices(tensor_to_shuffle,
                            m_batch_indices_owner.tensor_view().to_read_only(),
                            shuffled,
                            stream);
}

template<typename T, bool trans>
void
BatchShuffler::train_test_split_impl(TensorReadOnly2D<T, trans> tensor_to_split,
                                     TensorWriteable2D<T> train,
                                     TensorWriteable2D<T> test,
                                     cudaStream_t stream) {
  auto train_size = m_batch_size;
  auto test_size = static_cast<size_type>(m_indices.size()) - m_batch_size;

  ops::copy_rows_by_indices(tensor_to_split,
                            m_batch_indices_owner.tensor_view().to_read_only(),
                            train,
                            stream);

  auto test_indices_owner = constructTensorOwnerDevice1D<size_type>(test_size, stream);
  auto test_indices_view = test_indices_owner.tensor_view();
  update_batch_indices(test_indices_view, m_indices.data() + train_size, test_size, stream);

  ops::copy_rows_by_indices(tensor_to_split,
                            test_indices_view.to_read_only(),
                            test,
                            stream);
}

} // perceptron
} // tensors
} // shufflers

#endif //PERCEPTRON_TENSORS_SHUFFLERS_BATCHSHUFFLER_H
