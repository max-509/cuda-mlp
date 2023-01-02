#include "perceptron/activations/LeakyReLUActivation.h"

namespace perceptron {
namespace activations {

namespace details {

// Reason for non-private methods: nvcc error
// The enclosing parent function ("compute_impl") for an extended __device__ lambda cannot have private or protected access within its class
template<typename T, bool trans>
static void
compute_impl(tensors::TensorReadOnly2D<T, trans> inputs,
             tensors::TensorWriteable2D<T> outputs,
             float negative_slope) {
  tensors::ops::copy_or_transform(inputs,
                                  [] DEVICE_CALLABLE(T value) { return value >= static_cast<T>(0.0); },
  [negative_slope] DEVICE_CALLABLE(T
  value) { return value * negative_slope; },
  outputs);
}

template<typename T, bool trans>
static void
derivative_impl(tensors::TensorReadOnly2D<T, trans> inputs,
                tensors::TensorWriteable2D<T> outputs,
                float negative_slope) {
  tensors::ops::set1_or_set2(static_cast<T>(1.0), static_cast<T>(negative_slope),
                             inputs,
                             [] DEVICE_CALLABLE(T value) { return value >= static_cast<T>(0.0); },
                             outputs);
}

} // details

LeakyReLUActivation::LeakyReLUActivation(float negative_slope) : m_negative_slope(negative_slope) {}

void
LeakyReLUActivation::compute(tensors::TensorReadOnly2D<float, false> inputs,
                             tensors::TensorWriteable2D<float> outputs) {
  details::compute_impl(inputs, outputs, m_negative_slope);
}

void
LeakyReLUActivation::compute(tensors::TensorReadOnly2D<float, true> inputs,
                             tensors::TensorWriteable2D<float> outputs) {
  details::compute_impl(inputs, outputs, m_negative_slope);
}

void
LeakyReLUActivation::derivative(tensors::TensorReadOnly2D<float, false> inputs,
                                tensors::TensorWriteable2D<float> outputs) {
  details::derivative_impl(inputs, outputs, m_negative_slope);
}

void
LeakyReLUActivation::derivative(tensors::TensorReadOnly2D<float, true> inputs,
                                tensors::TensorWriteable2D<float> outputs) {
  details::derivative_impl(inputs, outputs, m_negative_slope);
}

} // perceptron
} // activations
