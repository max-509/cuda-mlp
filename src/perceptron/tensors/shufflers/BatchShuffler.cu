#include "perceptron/tensors/shufflers/BatchShuffler.h"

namespace perceptron {
namespace tensors {
namespace shufflers {

BatchShuffler::BatchShuffler(size_type nrows,
                             size_type batch_size,
                             size_type seed,
                             cudaStream_t stream)
    : m_batch_indices_owner(constructTensorOwnerDevice1D<size_type>(batch_size, 1, stream)),
      m_g(seed),
      m_indices(std::vector<size_type>(nrows)),
      m_batch_size(batch_size) {
  std::iota(m_indices.begin(), m_indices.end(), 0);
  update_batch_indices(stream);
}

BatchShuffler::BatchShuffler(size_type nrows, size_type batch_size, size_type seed)
    : BatchShuffler(nrows, batch_size, seed, nullptr) {}

BatchShuffler::BatchShuffler(size_type nrows, size_type batch_size, cudaStream_t stream)
    : BatchShuffler(nrows, batch_size, 0, stream) {}

BatchShuffler::BatchShuffler(size_type nrows, size_type batch_size)
    : BatchShuffler(nrows, batch_size, 0, nullptr) {}

void
BatchShuffler::shuffle(cudaStream_t stream) {
  std::shuffle(m_indices.begin(), m_indices.end(), m_g);
  update_batch_indices(stream);
}

TensorOwner2D<float>
BatchShuffler::get_shuffled(TensorReadOnly2D<float, false> tensor_to_shuffle,
                            cudaStream_t stream) {
  auto shuffled = constructTensorOwnerDevice2D<float>(m_batch_size, tensor_to_shuffle.get_ncols(),
                                                      DEFAULT_2D_STRIDE, stream);
  get_shuffled(tensor_to_shuffle, shuffled.tensor_view(), stream);
  return shuffled;
}

TensorOwner2D<float>
BatchShuffler::get_shuffled(TensorReadOnly2D<float, true> tensor_to_shuffle,
                            cudaStream_t stream) {
  auto shuffled = constructTensorOwnerDevice2D<float>(m_batch_size, tensor_to_shuffle.get_ncols(),
                                                      DEFAULT_2D_STRIDE, stream);
  get_shuffled(tensor_to_shuffle, shuffled.tensor_view(), stream);
  return shuffled;
}

TensorOwner2D<double>
BatchShuffler::get_shuffled(TensorReadOnly2D<double, false> tensor_to_shuffle,
                            cudaStream_t stream) {
  auto shuffled = constructTensorOwnerDevice2D<double>(m_batch_size, tensor_to_shuffle.get_ncols(),
                                                       DEFAULT_2D_STRIDE, stream);
  get_shuffled(tensor_to_shuffle, shuffled.tensor_view(), stream);
  return shuffled;
}

TensorOwner2D<double>
BatchShuffler::get_shuffled(TensorReadOnly2D<double, true> tensor_to_shuffle,
                            cudaStream_t stream) {
  auto shuffled = constructTensorOwnerDevice2D<double>(m_batch_size, tensor_to_shuffle.get_ncols(),
                                                       DEFAULT_2D_STRIDE, stream);
  get_shuffled(tensor_to_shuffle, shuffled.tensor_view(), stream);
  return shuffled;
}

void
BatchShuffler::get_shuffled(TensorReadOnly2D<float, false> tensor_to_shuffle,
                            TensorWriteable2D<float> shuffled,
                            cudaStream_t stream) {
  get_shuffled_impl(tensor_to_shuffle, shuffled, stream);
}

void
BatchShuffler::get_shuffled(TensorReadOnly2D<float, true> tensor_to_shuffle,
                            TensorWriteable2D<float> shuffled,
                            cudaStream_t stream) {
  get_shuffled_impl(tensor_to_shuffle, shuffled, stream);
}

void
BatchShuffler::get_shuffled(TensorReadOnly2D<double, false> tensor_to_shuffle,
                            TensorWriteable2D<double> shuffled,
                            cudaStream_t stream) {
  get_shuffled_impl(tensor_to_shuffle, shuffled, stream);
}

void
BatchShuffler::get_shuffled(TensorReadOnly2D<double, true> tensor_to_shuffle,
                            TensorWriteable2D<double> shuffled,
                            cudaStream_t stream) {
  get_shuffled_impl(tensor_to_shuffle, shuffled, stream);
}

void
BatchShuffler::train_test_split(TensorReadOnly2D<float, false> tensor_to_split,
                                TensorWriteable2D<float> train,
                                TensorWriteable2D<float> test,
                                cudaStream_t stream) {
  train_test_split_impl(tensor_to_split, train, test, stream);
}

void
BatchShuffler::train_test_split(TensorReadOnly2D<float, true> tensor_to_split,
                                TensorWriteable2D<float> train,
                                TensorWriteable2D<float> test,
                                cudaStream_t stream) {
  train_test_split_impl(tensor_to_split, train, test, stream);
}

void
BatchShuffler::train_test_split(TensorReadOnly2D<double, false> tensor_to_split,
                                TensorWriteable2D<double> train,
                                TensorWriteable2D<double> test,
                                cudaStream_t stream) {
  train_test_split_impl(tensor_to_split, train, test, stream);
}

void
BatchShuffler::train_test_split(TensorReadOnly2D<double, true> tensor_to_split,
                                TensorWriteable2D<double> train,
                                TensorWriteable2D<double> test,
                                cudaStream_t stream) {
  train_test_split_impl(tensor_to_split, train, test, stream);
}

std::tuple<TensorOwner2D<float>, TensorOwner2D<float>>
BatchShuffler::train_test_split(TensorReadOnly2D<float, false> tensor_to_split,
                                cudaStream_t stream) {
  auto train_owner = constructTensorOwnerDevice2D<float>(m_batch_size, tensor_to_split.get_ncols(), stream);
  auto test_owner =
      constructTensorOwnerDevice2D<float>(m_indices.size() - m_batch_size, tensor_to_split.get_ncols(), stream);
  train_test_split(tensor_to_split, train_owner.tensor_view(), test_owner.tensor_view(), stream);

  return std::make_tuple(std::move(train_owner), std::move(test_owner));
}

std::tuple<TensorOwner2D<float>, TensorOwner2D<float>>
BatchShuffler::train_test_split(TensorReadOnly2D<float, true> tensor_to_split,
                                cudaStream_t stream) {
  auto train_owner = constructTensorOwnerDevice2D<float>(m_batch_size, tensor_to_split.get_ncols(), stream);
  auto test_owner =
      constructTensorOwnerDevice2D<float>(m_indices.size() - m_batch_size, tensor_to_split.get_ncols(), stream);
  train_test_split(tensor_to_split, train_owner.tensor_view(), test_owner.tensor_view(), stream);

  return std::make_tuple(std::move(train_owner), std::move(test_owner));
}

std::tuple<TensorOwner2D<double>, TensorOwner2D<double>>
BatchShuffler::train_test_split(TensorReadOnly2D<double, false> tensor_to_split,
                                cudaStream_t stream) {
  auto train_owner = constructTensorOwnerDevice2D<double>(m_batch_size, tensor_to_split.get_ncols(), stream);
  auto test_owner =
      constructTensorOwnerDevice2D<double>(m_indices.size() - m_batch_size, tensor_to_split.get_ncols(), stream);
  train_test_split(tensor_to_split, train_owner.tensor_view(), test_owner.tensor_view(), stream);

  return std::make_tuple(std::move(train_owner), std::move(test_owner));
}

std::tuple<TensorOwner2D<double>, TensorOwner2D<double>>
BatchShuffler::train_test_split(TensorReadOnly2D<double, true> tensor_to_split,
                                cudaStream_t stream) {
  auto train_owner = constructTensorOwnerDevice2D<double>(m_batch_size, tensor_to_split.get_ncols(), stream);
  auto test_owner =
      constructTensorOwnerDevice2D<double>(m_indices.size() - m_batch_size, tensor_to_split.get_ncols(), stream);
  train_test_split(tensor_to_split, train_owner.tensor_view(), test_owner.tensor_view(), stream);

  return std::make_tuple(std::move(train_owner), std::move(test_owner));
}

void
BatchShuffler::update_batch_indices(cudaStream_t stream) {
  auto batch_indices_view = m_batch_indices_owner.tensor_view();

  update_batch_indices(batch_indices_view, m_indices.data(), m_batch_size, stream);
}

void
BatchShuffler::update_batch_indices(TensorWriteable1D<size_type> batch_indices_tensor,
                                    size_type *indices_data,
                                    size_type indices_size,
                                    cudaStream_t stream) {
  utils::cu_memcpy_async(batch_indices_tensor.get(), indices_data, indices_size, cudaMemcpyDefault, stream);
}

} // perceptron
} // tensors
} // shufflers
