#include "perceptron/activations/IdentityActivation.h"

namespace perceptron {
namespace activations {

void
IdentityActivation::compute(tensors::TensorReadOnly2D<float, false> inputs,
                            tensors::TensorWriteable2D<float> outputs) {
  compute_impl(inputs, outputs);
}

void
IdentityActivation::compute(tensors::TensorReadOnly2D<float, true> inputs,
                            tensors::TensorWriteable2D<float> outputs) {
  compute_impl(inputs, outputs);
}

void
IdentityActivation::derivative(tensors::TensorReadOnly2D<float, false> inputs,
                               tensors::TensorWriteable2D<float> outputs) {
  derivative_impl(inputs, outputs);
}

void
IdentityActivation::derivative(tensors::TensorReadOnly2D<float, true> inputs,
                               tensors::TensorWriteable2D<float> outputs) {
  derivative_impl(inputs, outputs);
}

} // perceptron
} // activations
