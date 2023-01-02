#ifndef PERCEPTRON_LOSSES_SQUAREDLOSS_H
#define PERCEPTRON_LOSSES_SQUAREDLOSS_H

#include "perceptron/common/Common.h"
#include "perceptron/losses/ILoss.h"
#include "perceptron/tensors/ops/MathOps.h"

namespace perceptron {
namespace losses {

class SquaredLoss : public ILoss {
public:
  using ILoss::derivative;

  double
  compute(tensors::TensorReadOnly2D<float, true> preds, tensors::TensorReadOnly2D<float, true> trues) override;

  double
  compute(tensors::TensorReadOnly2D<float, true> preds, tensors::TensorReadOnly2D<float, false> trues) override;

  double
  compute(tensors::TensorReadOnly2D<float, false> preds, tensors::TensorReadOnly2D<float, true> trues) override;

  double
  compute(tensors::TensorReadOnly2D<float, false> preds, tensors::TensorReadOnly2D<float, false> trues) override;

  void
  derivative(tensors::TensorReadOnly2D<float, true> preds,
             tensors::TensorReadOnly2D<float, true> trues,
             tensors::TensorWriteable2D<float> outputs) override;

  void
  derivative(tensors::TensorReadOnly2D<float, true> preds,
             tensors::TensorReadOnly2D<float, false> trues,
             tensors::TensorWriteable2D<float> outputs) override;

  void
  derivative(tensors::TensorReadOnly2D<float, false> preds,
             tensors::TensorReadOnly2D<float, true> trues,
             tensors::TensorWriteable2D<float> outputs) override;

  void
  derivative(tensors::TensorReadOnly2D<float, false> preds,
             tensors::TensorReadOnly2D<float, false> trues,
             tensors::TensorWriteable2D<float> outputs) override;

private:
  template<typename T, bool trans_preds, bool trans_trues>
  double
  compute_impl(tensors::TensorReadOnly2D<T, trans_preds> preds,
               tensors::TensorReadOnly2D<T, trans_trues> trues);

  template<typename T, bool trans_preds, bool trans_trues>
  void
  derivative_impl(tensors::TensorReadOnly2D<T, trans_preds> preds,
                  tensors::TensorReadOnly2D<T, trans_trues> trues,
                  tensors::TensorWriteable2D<T> outputs);
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
  auto loss = tensors::ops::nrm2(diff_tensor_view.to_read_only());

  return loss / static_cast<float>(preds.get_nrows());
}

template<typename T, bool trans_preds, bool trans_trues>
void
SquaredLoss::derivative_impl(tensors::TensorReadOnly2D<T, trans_preds> preds,
                             tensors::TensorReadOnly2D<T, trans_trues> trues,
                             tensors::TensorWriteable2D<T> outputs) {
  is_valid_type<T>();

  tensors::ops::geam(preds, static_cast<T>(1.0),
                     trues, static_cast<T>(-1.0), outputs);

  const auto derivative_coeff = static_cast<T>(2.0 / preds.get_nrows());
  tensors::ops::scal(derivative_coeff, outputs);
}

} // perceptron
} // losses

#endif //PERCEPTRON_LOSSES_SQUAREDLOSS_H
