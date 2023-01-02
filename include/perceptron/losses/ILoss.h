#ifndef PERCEPTRON_LOSSES_ILOSS_H
#define PERCEPTRON_LOSSES_ILOSS_H

#include "perceptron/common/Common.h"
#include "perceptron/tensors/Tensor2D.h"

#include <string>
#include <sstream>
#include <stdexcept>

namespace perceptron {
namespace losses {

class ILoss {
public:
  ILoss() = default;
  ILoss(const ILoss &) = default;
  ILoss(ILoss &&) = default;

  ILoss &operator=(const ILoss &) = default;
  ILoss &operator=(ILoss &&) = default;

  virtual ~ILoss() = default;

  virtual double
  compute(tensors::TensorReadOnly2D<float, true> preds,
          tensors::TensorReadOnly2D<float, true> trues) = 0;

  virtual double
  compute(tensors::TensorReadOnly2D<float, true> preds,
          tensors::TensorReadOnly2D<float, false> trues) = 0;

  virtual double
  compute(tensors::TensorReadOnly2D<float, false> preds,
          tensors::TensorReadOnly2D<float, true> trues) = 0;

  virtual double
  compute(tensors::TensorReadOnly2D<float, false> preds,
          tensors::TensorReadOnly2D<float, false> trues) = 0;

  tensors::TensorOwner2D<float>
  derivative(tensors::TensorReadOnly2D<float, true> preds,
             tensors::TensorReadOnly2D<float, true> trues);

  tensors::TensorOwner2D<float>
  derivative(tensors::TensorReadOnly2D<float, true> preds,
             tensors::TensorReadOnly2D<float, false> trues);

  tensors::TensorOwner2D<float>
  derivative(tensors::TensorReadOnly2D<float, false> preds,
             tensors::TensorReadOnly2D<float, true> trues);

  tensors::TensorOwner2D<float>
  derivative(tensors::TensorReadOnly2D<float, false> preds,
             tensors::TensorReadOnly2D<float, false> trues);

  virtual void
  derivative(tensors::TensorReadOnly2D<float, true> preds,
             tensors::TensorReadOnly2D<float, true> trues,
             tensors::TensorWriteable2D<float> outputs) = 0;

  virtual void
  derivative(tensors::TensorReadOnly2D<float, true> preds,
             tensors::TensorReadOnly2D<float, false> trues,
             tensors::TensorWriteable2D<float> outputs) = 0;

  virtual void
  derivative(tensors::TensorReadOnly2D<float, false> preds,
             tensors::TensorReadOnly2D<float, true> trues,
             tensors::TensorWriteable2D<float> outputs) = 0;

  virtual void
  derivative(tensors::TensorReadOnly2D<float, false> preds,
             tensors::TensorReadOnly2D<float, false> trues,
             tensors::TensorWriteable2D<float> outputs) = 0;

private:
  template<typename T, bool trans_preds, bool trans_trues>
  tensors::TensorOwner2D<T>
  derivative_impl(tensors::TensorReadOnly2D<T, trans_preds> preds,
                  tensors::TensorReadOnly2D<T, trans_trues> trues);

};

template<typename T, bool trans_preds, bool trans_trues>
tensors::TensorOwner2D<T>
ILoss::derivative_impl(tensors::TensorReadOnly2D<T, trans_preds> preds,
                       tensors::TensorReadOnly2D<T, trans_trues> trues) {
  is_valid_type<T>();
  if (preds.get_nrows() != trues.get_nrows() || preds.get_ncols() != trues.get_ncols()) {
    std::stringstream stream;
    stream << "Preds shape and trues shape must be equal, but preds shape is " << preds.shape_repr()
           << ", trues shape is " << trues.shape_repr();
    throw std::invalid_argument{stream.str()};
  }
  auto outputs_owner = tensors::constructTensorOwnerDevice2D<T>(preds.get_nrows(), preds.get_ncols());
  auto outputs_view = outputs_owner.tensor_view();
  derivative(preds, trues, outputs_view);
  return outputs_owner;
}

} // perceptron
} // losses

#endif // PERCEPTRON_LOSSES_ILOSS_H
