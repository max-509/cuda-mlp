#ifndef PERCEPTRON_ACTIVATIONS_IACTIVATION_H
#define PERCEPTRON_ACTIVATIONS_IACTIVATION_H

#include "perceptron/common/Common.h"
#include "perceptron/tensors/Tensor2D.h"

namespace perceptron {
namespace activations {

class IActivation {
public:
  IActivation() = default;
  IActivation(const IActivation &) = default;
  IActivation(IActivation &&) = default;
  IActivation &operator=(const IActivation &) = default;
  IActivation &operator=(IActivation &&) = default;

  tensors::TensorOwner2D<float>
  compute(tensors::TensorReadOnly2D<float, false> inputs);

  tensors::TensorOwner2D<float>
  compute(tensors::TensorReadOnly2D<float, true> inputs);

  virtual void
  compute(tensors::TensorReadOnly2D<float, false> inputs,
          tensors::TensorWriteable2D<float> outputs) = 0;

  virtual void
  compute(tensors::TensorReadOnly2D<float, true> inputs,
          tensors::TensorWriteable2D<float> outputs) = 0;

  tensors::TensorOwner2D<float>
  derivative(tensors::TensorReadOnly2D<float, false> inputs);

  tensors::TensorOwner2D<float>
  derivative(tensors::TensorReadOnly2D<float, true> inputs);

  virtual void
  derivative(tensors::TensorReadOnly2D<float, false> inputs,
             tensors::TensorWriteable2D<float> outputs) = 0;

  virtual void
  derivative(tensors::TensorReadOnly2D<float, true> inputs,
             tensors::TensorWriteable2D<float> outputs) = 0;

  virtual ~IActivation() = default;
};

} // perceptron
} // activations

#endif //PERCEPTRON_ACTIVATIONS_IACTIVATION_H
