#include "perceptron/tensors/shufflers/IShuffler.h"

namespace perceptron {
namespace tensors {
namespace shufflers {

void
IShuffler::shuffle() {
  shuffle(nullptr);
}

TensorOwner2D<float>
IShuffler::get_shuffled(TensorReadOnly2D<float, false> tensor_to_shuffle) {
  return get_shuffled(tensor_to_shuffle, nullptr);
}

TensorOwner2D<float>
IShuffler::get_shuffled(TensorReadOnly2D<float, true> tensor_to_shuffle) {
  return get_shuffled(tensor_to_shuffle, nullptr);
}

TensorOwner2D<double>
IShuffler::get_shuffled(TensorReadOnly2D<double, false> tensor_to_shuffle) {
  return get_shuffled(tensor_to_shuffle, nullptr);
}

TensorOwner2D<double>
IShuffler::get_shuffled(TensorReadOnly2D<double, true> tensor_to_shuffle) {
  return get_shuffled(tensor_to_shuffle, nullptr);
}

void
IShuffler::get_shuffled(TensorReadOnly2D<float, false> tensor_to_shuffle,
                        TensorWriteable2D<float> shuffled) {
  get_shuffled(tensor_to_shuffle, shuffled, nullptr);
}

void
IShuffler::get_shuffled(TensorReadOnly2D<float, true> tensor_to_shuffle,
                        TensorWriteable2D<float> shuffled) {
  get_shuffled(tensor_to_shuffle, shuffled, nullptr);
}

void
IShuffler::get_shuffled(TensorReadOnly2D<double, false> tensor_to_shuffle,
                        TensorWriteable2D<double> shuffled) {
  get_shuffled(tensor_to_shuffle, shuffled, nullptr);
}

void
IShuffler::get_shuffled(TensorReadOnly2D<double, true> tensor_to_shuffle,
                        TensorWriteable2D<double> shuffled) {
  get_shuffled(tensor_to_shuffle, shuffled, nullptr);
}

} // perceptron
} // tensors
} // shufflers