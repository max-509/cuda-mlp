#ifndef PERCEPTRON_LOSSES_ILOSS_H
#define PERCEPTRON_LOSSES_ILOSS_H

#include "perceptron/common/Common.h"
#include "perceptron/tensors/Tensor2D.h"

namespace perceptron {
namespace losses {

class ILoss {
public:
  ILoss() = default;
  ILoss(const ILoss &) = default;
  ILoss(ILoss &&) = default;

  ILoss &operator=(const ILoss &) = default;
  ILoss &operator=(ILoss &&) = default;

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

  virtual tensors::TensorOwnerDevice2D<float>
  derivative(tensors::TensorReadOnly2D<float, true> preds,
             tensors::TensorReadOnly2D<float, true> trues) = 0;

  virtual tensors::TensorOwnerDevice2D<float>
  derivative(tensors::TensorReadOnly2D<float, true> preds,
             tensors::TensorReadOnly2D<float, false> trues) = 0;

  virtual tensors::TensorOwnerDevice2D<float>
  derivative(tensors::TensorReadOnly2D<float, false> preds,
             tensors::TensorReadOnly2D<float, true> trues) = 0;

  virtual tensors::TensorOwnerDevice2D<float>
  derivative(tensors::TensorReadOnly2D<float, false> preds,
             tensors::TensorReadOnly2D<float, false> trues) = 0;

  virtual ~ILoss() = default;
};

} // perceptron
} // losses

#endif // PERCEPTRON_LOSSES_ILOSS_H
