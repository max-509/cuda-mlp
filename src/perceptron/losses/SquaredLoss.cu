#include "perceptron/losses/SquaredLoss.h"

namespace perceptron {
namespace losses {

double
SquaredLoss::compute(tensors::TensorReadOnly2D<float, true> preds,
                     tensors::TensorReadOnly2D<float, true> trues) {
  return compute_impl(preds, trues);
}

double
SquaredLoss::compute(tensors::TensorReadOnly2D<float, true> preds,
                     tensors::TensorReadOnly2D<float, false> trues) {
  return compute_impl(preds, trues);
}

double
SquaredLoss::compute(tensors::TensorReadOnly2D<float, false> preds,
                     tensors::TensorReadOnly2D<float, true> trues) {
  return compute_impl(preds, trues);
}

double
SquaredLoss::compute(tensors::TensorReadOnly2D<float, false> preds,
                     tensors::TensorReadOnly2D<float, false> trues) {
  return compute_impl(preds, trues);
}

void
SquaredLoss::derivative(tensors::TensorReadOnly2D<float, true> preds,
                        tensors::TensorReadOnly2D<float, true> trues,
                        tensors::TensorWriteable2D<float> outputs) {
  derivative_impl(preds, trues, outputs);
}

void
SquaredLoss::derivative(tensors::TensorReadOnly2D<float, true> preds,
                        tensors::TensorReadOnly2D<float, false> trues,
                        tensors::TensorWriteable2D<float> outputs) {
  derivative_impl(preds, trues, outputs);
}

void
SquaredLoss::derivative(tensors::TensorReadOnly2D<float, false> preds,
                        tensors::TensorReadOnly2D<float, true> trues,
                        tensors::TensorWriteable2D<float> outputs) {
  derivative_impl(preds, trues, outputs);
}

void
SquaredLoss::derivative(tensors::TensorReadOnly2D<float, false> preds,
                        tensors::TensorReadOnly2D<float, false> trues,
                        tensors::TensorWriteable2D<float> outputs) {
  derivative_impl(preds, trues, outputs);
}

} // perceptron
} // losses