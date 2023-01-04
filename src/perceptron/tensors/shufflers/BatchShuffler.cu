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
      m_batch_size(batch_size),
      m_stream(stream) {
  std::iota(m_indices.begin(), m_indices.end(), 0);
  update_batch_indices();
}

BatchShuffler::BatchShuffler(size_type nrows, size_type batch_size, size_type seed)
    : BatchShuffler(nrows, batch_size, seed, nullptr) {}

BatchShuffler::BatchShuffler(size_type nrows, size_type batch_size, cudaStream_t stream)
    : BatchShuffler(nrows, batch_size, 0, stream) {}

BatchShuffler::BatchShuffler(size_type nrows, size_type batch_size)
    : BatchShuffler(nrows, batch_size, 0, nullptr) {}

void
BatchShuffler::shuffle() {
  std::shuffle(m_indices.begin(), m_indices.end(), m_g);
  update_batch_indices();
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
BatchShuffler::update_batch_indices() {
  auto batch_indices_view = m_batch_indices_owner.tensor_view();

  utils::cu_memcpy_async(batch_indices_view.get(), m_indices.data(), m_batch_size, cudaMemcpyDefault, m_stream);
}

} // perceptron
} // tensors
} // shufflers
