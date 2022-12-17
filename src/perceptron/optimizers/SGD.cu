#include "perceptron/optimizers/SGD.h"

namespace perceptron {
namespace optimizers {

SGD::SGD(double lr, double weights_decay, double momentum, double dampening, bool netsterov)
    : m_lr(lr), m_weights_decay(weights_decay), m_momentum(momentum), m_dampening(dampening), m_nesterov(netsterov) {}

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