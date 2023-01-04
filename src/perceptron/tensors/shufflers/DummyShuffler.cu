#include "perceptron/tensors/shufflers/DummyShuffler.h"

namespace perceptron {
namespace tensors {
namespace shufflers {

void DummyShuffler::shuffle() {}

TensorOwner2D<float>
DummyShuffler::get_shuffled(TensorReadOnly2D<float, false> tensor_to_shuffle,
                            cudaStream_t stream) {
  auto shuffled = constructTensorOwnerDevice2D<float>(tensor_to_shuffle.get_nrows(), tensor_to_shuffle.get_ncols(),
                                                      DEFAULT_2D_STRIDE, stream);
  get_shuffled(tensor_to_shuffle, shuffled.tensor_view(), stream);
  return shuffled;
}

TensorOwner2D<float>
DummyShuffler::get_shuffled(TensorReadOnly2D<float, true> tensor_to_shuffle,
                            cudaStream_t stream) {
  auto shuffled = constructTensorOwnerDevice2D<float>(tensor_to_shuffle.get_nrows(), tensor_to_shuffle.get_ncols(),
                                                      DEFAULT_2D_STRIDE, stream);
  get_shuffled(tensor_to_shuffle, shuffled.tensor_view(), stream);
  return shuffled;
}

TensorOwner2D<double>
DummyShuffler::get_shuffled(TensorReadOnly2D<double, false> tensor_to_shuffle,
                            cudaStream_t stream) {
  auto shuffled = constructTensorOwnerDevice2D<double>(tensor_to_shuffle.get_nrows(), tensor_to_shuffle.get_ncols(),
                                                       DEFAULT_2D_STRIDE, stream);
  get_shuffled(tensor_to_shuffle, shuffled.tensor_view(), stream);
  return shuffled;
}

TensorOwner2D<double>
DummyShuffler::get_shuffled(TensorReadOnly2D<double, true> tensor_to_shuffle,
                            cudaStream_t stream) {
  auto shuffled = constructTensorOwnerDevice2D<double>(tensor_to_shuffle.get_nrows(), tensor_to_shuffle.get_ncols(),
                                                       DEFAULT_2D_STRIDE, stream);
  get_shuffled(tensor_to_shuffle, shuffled.tensor_view(), stream);
  return shuffled;
}

void
DummyShuffler::get_shuffled(TensorReadOnly2D<float, false> tensor_to_shuffle,
                            TensorWriteable2D<float> shuffled,
                            cudaStream_t stream) {
  get_shuffled_impl(tensor_to_shuffle, shuffled, stream);
}

void
DummyShuffler::get_shuffled(TensorReadOnly2D<float, true> tensor_to_shuffle,
                            TensorWriteable2D<float> shuffled,
                            cudaStream_t stream) {
  get_shuffled_impl(tensor_to_shuffle, shuffled, stream);
}

void
DummyShuffler::get_shuffled(TensorReadOnly2D<double, false> tensor_to_shuffle,
                            TensorWriteable2D<double> shuffled,
                            cudaStream_t stream) {
  get_shuffled_impl(tensor_to_shuffle, shuffled, stream);
}

void
DummyShuffler::get_shuffled(TensorReadOnly2D<double, true> tensor_to_shuffle,
                            TensorWriteable2D<double> shuffled,
                            cudaStream_t stream) {
  get_shuffled_impl(tensor_to_shuffle, shuffled, stream);
}

} // perceptron
} // tensors
} // shufflers