#ifndef PERCEPTRON_LOSSES_LOGLOSSWITHSOFTMAX_H
#define PERCEPTRON_LOSSES_LOGLOSSWITHSOFTMAX_H

#include "perceptron/common/Common.h"
#include "perceptron/losses/ILoss.h"
#include "perceptron/tensors/Tensor2D.h"
#include "perceptron/tensors/ops/Ops.h"

namespace perceptron {
namespace losses {

class LogLossWithSoftMax : public ILoss {
  using ILoss::derivative;

  bool
  with_activation() const override;

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
LogLossWithSoftMax::compute_impl(tensors::TensorReadOnly2D<T, trans_preds> preds,
                                 tensors::TensorReadOnly2D<T, trans_trues> trues) {
  // TODO: Finish
  return 0.0;
}

template<typename T, bool trans_preds, bool trans_trues>
void
LogLossWithSoftMax::derivative_impl(tensors::TensorReadOnly2D<T, trans_preds> preds,
                                    tensors::TensorReadOnly2D<T, trans_trues> trues,
                                    tensors::TensorWriteable2D<T> outputs) {
  // TODO: Finish
}

} // perceptron
} // losses

#endif //PERCEPTRON_LOSSES_LOGLOSSWITHSOFTMAX_H
