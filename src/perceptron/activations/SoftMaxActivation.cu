#include "perceptron/activations/SoftMaxActivation.h"

namespace perceptron {
namespace activations {

void
SoftMaxActivation::compute(tensors::TensorReadOnly2D<float, false> inputs,
                           tensors::TensorWriteable2D<float> outputs) {
  compute_impl(inputs, outputs);
}

void
SoftMaxActivation::compute(tensors::TensorReadOnly2D<float, true> inputs,
                           tensors::TensorWriteable2D<float> outputs) {
  compute_impl(inputs, outputs);
}

void
SoftMaxActivation::derivative(tensors::TensorReadOnly2D<float, false> inputs,
                              tensors::TensorWriteable2D<float> outputs) {
  derivative_impl(inputs, outputs);
}

void
SoftMaxActivation::derivative(tensors::TensorReadOnly2D<float, true> inputs,
                              tensors::TensorWriteable2D<float> outputs) {
  derivative_impl(inputs, outputs);
}

} // perceptron
} // activations