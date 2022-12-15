#include "perceptron/optimizers/SGD.h"

namespace perceptron {
namespace optimizers {
void SGD::descent(tensors::TensorReadOnly2D<float, false> grads, tensors::TensorWriteable2D<float> weigths) {
  descent_impl(grads, weigths);
}
void SGD::descent(tensors::TensorReadOnly2D<float, true> grads, tensors::TensorWriteable2D<float> weigths) {
  descent_impl(grads, weigths);
}
void SGD::reset() {
  m_velocity = std::nullopt;
}
} // perceptron
} // optimizers