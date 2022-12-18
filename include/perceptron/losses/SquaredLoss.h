#ifndef PERCEPTRON_LOSSES_SQUAREDLOSS_H
#define PERCEPTRON_LOSSES_SQUAREDLOSS_H

#include "perceptron/common/Common.h"
#include "perceptron/losses/ILoss.h"
#include "perceptron/tensors/ops/MathOps.h"

namespace perceptron {
namespace losses {

class SquaredLoss : public ILoss {
public:
  double
  compute(tensors::TensorReadOnly2D<float, true> preds, tensors::TensorReadOnly2D<float, true> trues) override;

  double
  compute(tensors::TensorReadOnly2D<float, true> preds, tensors::TensorReadOnly2D<float, false> trues) override;

  double
  compute(tensors::TensorReadOnly2D<float, false> preds, tensors::TensorReadOnly2D<float, true> trues) override;

  double
  compute(tensors::TensorReadOnly2D<float, false> preds, tensors::TensorReadOnly2D<float, false> trues) override;

  tensors::TensorOwnerDevice2D<float>
  derivative(tensors::TensorReadOnly2D<float, true> preds,
             tensors::TensorReadOnly2D<float, true> trues) override;

  tensors::TensorOwnerDevice2D<float>
  derivative(tensors::TensorReadOnly2D<float, true> preds,
             tensors::TensorReadOnly2D<float, false> trues) override;

  tensors::TensorOwnerDevice2D<float>
  derivative(tensors::TensorReadOnly2D<float, false> preds,
             tensors::TensorReadOnly2D<float, true> trues) override;

  tensors::TensorOwnerDevice2D<float>
  derivative(tensors::TensorReadOnly2D<float, false> preds,
             tensors::TensorReadOnly2D<float, false> trues) override;

private:
  template<typename T, bool trans_preds, bool trans_trues>
  double
  compute_impl(tensors::TensorReadOnly2D<T, trans_preds> preds, tensors::TensorReadOnly2D<T, trans_trues> trues);

  template<typename T, bool trans_preds, bool trans_trues>
  tensors::TensorOwnerDevice2D<T>
  derivative_impl(tensors::TensorReadOnly2D<T, trans_preds> preds, tensors::TensorReadOnly2D<T, trans_trues> trues);
};

template<typename T, bool trans_preds, bool trans_trues>
double
SquaredLoss::compute_impl(tensors::TensorReadOnly2D<T, trans_preds> preds,
                          tensors::TensorReadOnly2D<T, trans_trues> trues) {
  is_valid_type<T>();
  auto diff_tensor_owner =
      tensors::constructTensorOwnerDevice2D<T>(preds.get_nrows(), preds.get_ncols());
  auto diff_tensor_view = diff_tensor_owner.tensor_view();

  tensors::ops::geam(preds, static_cast<T>(1.0),
                     trues, static_cast<T>(-1.0), diff_tensor_view);
  auto loss = tensors::ops::nrm2(tensors::constructTensorReadOnly1D(diff_tensor_view.get(),
                                                                    diff_tensor_view.get_nrows()
                                                                        * diff_tensor_view.get_ncols()));

  return loss / static_cast<float>(preds.get_nrows());
}

template<typename T, bool trans_preds, bool trans_trues>
tensors::TensorOwnerDevice2D<T>
SquaredLoss::derivative_impl(tensors::TensorReadOnly2D<T, trans_preds> preds,
                             tensors::TensorReadOnly2D<T, trans_trues> trues) {
  is_valid_type<T>();
  auto diff_tensor_owner =
      tensors::constructTensorOwnerDevice2D<T>(preds.get_nrows(), preds.get_ncols());
  auto diff_tensor_view = diff_tensor_owner.tensor_view();

  tensors::ops::geam(preds, static_cast<T>(1.0),
                     trues, static_cast<T>(-1.0), diff_tensor_view);

  const auto derivative_coeff = static_cast<T>(2.0 / preds.get_nrows());
  tensors::ops::scal(derivative_coeff, tensors::constructTensorWriteable1D(diff_tensor_view.get(),
                                                                           diff_tensor_view.get_nrows()
                                                                               * diff_tensor_view.get_ncols()));

  return diff_tensor_owner;
}

} // perceptron
} // losses

#endif //PERCEPTRON_LOSSES_SQUAREDLOSS_H
