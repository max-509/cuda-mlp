#include "perceptron/activations/SigmoidActivation.h"

namespace perceptron {
namespace activations {

void
SigmoidActivation::compute(tensors::TensorReadOnly2D<float, false> inputs,
                           tensors::TensorWriteable2D<float> outputs) {
  compute_impl(inputs, outputs);
}

void
SigmoidActivation::compute(tensors::TensorReadOnly2D<float, true> inputs,
                           tensors::TensorWriteable2D<float> outputs) {
  compute_impl(inputs, outputs);
}

void
SigmoidActivation::derivative(tensors::TensorReadOnly2D<float, false> inputs,
                              tensors::TensorWriteable2D<float> outputs) {
  derivative_impl(inputs, outputs);
}

void
SigmoidActivation::derivative(tensors::TensorReadOnly2D<float, true> inputs,
                              tensors::TensorWriteable2D<float> outputs) {
  derivative_impl(inputs, outputs);
}

} // perceptron
} // activations