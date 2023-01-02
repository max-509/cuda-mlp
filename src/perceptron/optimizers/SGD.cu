#include "perceptron/optimizers/SGD.h"

namespace perceptron {
namespace optimizers {

SGD::describer_t
SGD::describer_t::set_lr(double value) {
  auto copy = *this;
  copy.lr = value;
  return copy;
}

SGD::describer_t
SGD::describer_t::set_weights_decay(double value) {
  auto copy = *this;
  copy.weights_decay = value;
  return copy;
}

SGD::describer_t
SGD::describer_t::set_momentum(double value) {
  auto copy = *this;
  copy.momentum = value;
  return copy;
}

SGD::describer_t
SGD::describer_t::set_dampening(double value) {
  auto copy = *this;
  copy.dampening = value;
  return copy;
}

SGD::describer_t
SGD::describer_t::set_nesterov(bool value) {
  auto copy = *this;
  copy.nesterov = value;
  return copy;
}


SGD::SGD(double lr, double weights_decay, double momentum, double dampening, bool nesterov)
    : m_lr(lr), m_weights_decay(weights_decay), m_momentum(momentum), m_dampening(dampening), m_nesterov(nesterov) {}

SGD::SGD(SGD::describer_t describer) : SGD(describer.lr,
                                           describer.weights_decay,
                                           describer.momentum,
                                           describer.dampening,
                                           describer.nesterov) {}

void
SGD::descent(tensors::TensorReadOnly2D<float, false> grads, tensors::TensorWriteable2D<float> weigths) {
  descent_impl(grads, weigths);
}

void
SGD::descent(tensors::TensorReadOnly2D<float, true> grads, tensors::TensorWriteable2D<float> weigths) {
  descent_impl(grads, weigths);
}

void
SGD::reset() {
  m_velocity = std::nullopt;
}

} // perceptron
} // optimizers