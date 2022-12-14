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

tensors::TensorOwnerDevice2D<float>
SquaredLoss::derivative(tensors::TensorReadOnly2D<float, true> preds,
                        tensors::TensorReadOnly2D<float, true> trues) {
  return derivative_impl(preds, trues);
}

tensors::TensorOwnerDevice2D<float>
SquaredLoss::derivative(tensors::TensorReadOnly2D<float, true> preds,
                        tensors::TensorReadOnly2D<float, false> trues) {
  return derivative_impl(preds, trues);
}

tensors::TensorOwnerDevice2D<float>
SquaredLoss::derivative(tensors::TensorReadOnly2D<float, false> preds,
                        tensors::TensorReadOnly2D<float, true> trues) {
  return derivative_impl(preds, trues);
}

tensors::TensorOwnerDevice2D<float>
SquaredLoss::derivative(tensors::TensorReadOnly2D<float, false> preds,
                        tensors::TensorReadOnly2D<float, false> trues) {
  return derivative_impl(preds, trues);
}

} // perceptron
} // losses