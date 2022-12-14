#ifndef PERCEPTRON_ACTIVATIONS_SIGMOIDACTIVATION_H
#define PERCEPTRON_ACTIVATIONS_SIGMOIDACTIVATION_H

#include "perceptron/common/Common.h"
#include "perceptron/activations/IActivation.h"
#include "perceptron/tensors/ops/MathOps.h"
#include "perceptron/tensors/ops/MemoryOps.h"

namespace perceptron {
namespace activations {

class SigmoidActivation : public IActivation {
public:
  tensors::TensorOwnerDevice2D<float>
  compute(tensors::TensorReadOnly2D<float, false> inputs) override;
  tensors::TensorOwnerDevice2D<float>
  compute(tensors::TensorReadOnly2D<float, true> inputs) override;

  tensors::TensorOwnerDevice2D<float>
  derivative(tensors::TensorReadOnly2D<float, false> inputs) override;
  tensors::TensorOwnerDevice2D<float>
  derivative(tensors::TensorReadOnly2D<float, true> inputs) override;

private:
  template<typename T, bool trans>
  tensors::TensorOwnerDevice2D<T>
  compute_impl(tensors::TensorReadOnly2D<T, trans> inputs);

  template<typename T, bool trans>
  tensors::TensorOwnerDevice2D<T>
  derivative_impl(tensors::TensorReadOnly2D<T, trans> inputs);
};

template<typename T, bool trans>
tensors::TensorOwnerDevice2D<T>
SigmoidActivation::compute_impl(tensors::TensorReadOnly2D<T, trans> inputs) {
  auto output_owner = tensors::constructTensorOwnerDevice2D<T>(inputs.get_y_dim(),
                                                               inputs.get_x_dim());

  tensors::ops::negative_exp(inputs, output_owner.tensor_view());
  tensors::ops::add(static_cast<T>(1.0), output_owner.tensor_view());
  tensors::ops::reverse_scal(static_cast<T>(1.0), output_owner.tensor_view());

  return output_owner;
}

template<typename T, bool trans>
tensors::TensorOwnerDevice2D<T>
SigmoidActivation::derivative_impl(tensors::TensorReadOnly2D<T, trans> inputs) {
  auto output_owner = tensors::constructTensorOwnerDevice2D<T>(inputs.get_y_dim(),
                                                               inputs.get_x_dim());
  auto sigm = compute_impl(inputs);
  tensors::ops::copy(sigm.tensor_view().to_read_only(), output_owner.tensor_view());
  tensors::ops::add_negative(static_cast<T>(1.0), sigm.tensor_view());
  tensors::ops::element_wise_mul(sigm.tensor_view().to_read_only(), output_owner.tensor_view());

  return output_owner;
}

} // perceptron
} // activations

#endif //PERCEPTRON_ACTIVATIONS_SIGMOIDACTIVATION_H
