#include "perceptron/losses/ILoss.h"

namespace perceptron {
namespace losses {

bool
ILoss::with_activation() const {
  return false;
}

tensors::TensorOwner2D<float>
ILoss::derivative(tensors::TensorReadOnly2D<float, true> preds,
                  tensors::TensorReadOnly2D<float, true> trues) {
  return derivative_impl(preds, trues);
}

tensors::TensorOwner2D<float>
ILoss::derivative(tensors::TensorReadOnly2D<float, true> preds,
                  tensors::TensorReadOnly2D<float, false> trues) {
  return derivative_impl(preds, trues);
}

tensors::TensorOwner2D<float>
ILoss::derivative(tensors::TensorReadOnly2D<float, false> preds,
                  tensors::TensorReadOnly2D<float, true> trues) {
  return derivative_impl(preds, trues);
}

tensors::TensorOwner2D<float>
ILoss::derivative(tensors::TensorReadOnly2D<float, false> preds,
                  tensors::TensorReadOnly2D<float, false> trues) {
  return derivative_impl(preds, trues);
}

} // perceptron
} // losses