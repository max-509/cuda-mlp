#include "perceptron/activations/ReLUActivation.h"
#include "perceptron/tensors/Tensor2D.h"

namespace perceptron {
namespace activations {
namespace details {

// Reason for non-private methods: nvcc error
// The enclosing parent function ("compute_impl") for an extended __device__ lambda cannot have private or protected access within its class
template<typename T, bool trans>
static void
compute_impl(tensors::TensorReadOnly2D<T, trans> inputs,
             tensors::TensorWriteable2D<T> outputs) {
  tensors::ops::copy(inputs, outputs);
  tensors::ops::copy_if(inputs, [] DEVICE_CALLABLE(T value) { return value > 0; }, outputs);
}

template<typename T, bool trans>
static void
derivative_impl(tensors::TensorReadOnly2D<T, trans> inputs,
                tensors::TensorWriteable2D<T> outputs) {
  tensors::ops::copy(inputs, outputs);
  tensors::ops::set_if(static_cast<T>(1.0), [] DEVICE_CALLABLE(T value) { return value > 0; }, outputs);
}

} // details

void
ReLUActivation::compute(tensors::TensorReadOnly2D<float, false> inputs,
                        tensors::TensorWriteable2D<float> outputs) {
  details::compute_impl(inputs, outputs);
}

void
ReLUActivation::compute(tensors::TensorReadOnly2D<float, true> inputs,
                        tensors::TensorWriteable2D<float> outputs) {
  details::compute_impl(inputs, outputs);
}

void
ReLUActivation::derivative(tensors::TensorReadOnly2D<float, false> inputs,
                           tensors::TensorWriteable2D<float> outputs) {
  details::derivative_impl(inputs, outputs);
}

void
ReLUActivation::derivative(tensors::TensorReadOnly2D<float, true> inputs,
                           tensors::TensorWriteable2D<float> outputs) {
  details::derivative_impl(inputs, outputs);
}

} // perceptron
} // activations