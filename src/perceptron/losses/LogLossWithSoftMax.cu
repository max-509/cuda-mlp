#include "perceptron/losses/LogLossWithSoftMax.h"

namespace perceptron {
namespace losses {

bool
LogLossWithSoftMax::with_activation() const {
  return true;
}

double
LogLossWithSoftMax::compute(tensors::TensorReadOnly2D<float, true> preds,
                            tensors::TensorReadOnly2D<float, true> trues) {
  return compute_impl(preds, trues);
}

double
LogLossWithSoftMax::compute(tensors::TensorReadOnly2D<float, true> preds,
                            tensors::TensorReadOnly2D<float, false> trues) {
  return compute_impl(preds, trues);
}

double
LogLossWithSoftMax::compute(tensors::TensorReadOnly2D<float, false> preds,
                            tensors::TensorReadOnly2D<float, true> trues) {
  return compute_impl(preds, trues);
}

double
LogLossWithSoftMax::compute(tensors::TensorReadOnly2D<float, false> preds,
                            tensors::TensorReadOnly2D<float, false> trues) {
  return compute_impl(preds, trues);
}

void
LogLossWithSoftMax::derivative(tensors::TensorReadOnly2D<float, true> preds,
                               tensors::TensorReadOnly2D<float, true> trues,
                               tensors::TensorWriteable2D<float> outputs) {
  derivative_impl(preds, trues, outputs);
}

void
LogLossWithSoftMax::derivative(tensors::TensorReadOnly2D<float, true> preds,
                               tensors::TensorReadOnly2D<float, false> trues,
                               tensors::TensorWriteable2D<float> outputs) {
  derivative_impl(preds, trues, outputs);
}

void
LogLossWithSoftMax::derivative(tensors::TensorReadOnly2D<float, false> preds,
                               tensors::TensorReadOnly2D<float, true> trues,
                               tensors::TensorWriteable2D<float> outputs) {
  derivative_impl(preds, trues, outputs);
}

void
LogLossWithSoftMax::derivative(tensors::TensorReadOnly2D<float, false> preds,
                               tensors::TensorReadOnly2D<float, false> trues,
                               tensors::TensorWriteable2D<float> outputs) {
  derivative_impl(preds, trues, outputs);
}

} // perceptron
} // losses