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
  using IActivation::compute;
  void
  compute(tensors::TensorReadOnly2D<float, false> inputs,
          tensors::TensorWriteable2D<float> outputs) override;
  void
  compute(tensors::TensorReadOnly2D<float, true> inputs,
          tensors::TensorWriteable2D<float> outputs) override;

  void
  derivative(tensors::TensorReadOnly2D<float, false> inputs,
             tensors::TensorWriteable2D<float> outputs) override;
  void
  derivative(tensors::TensorReadOnly2D<float, true> inputs,
             tensors::TensorWriteable2D<float> outputs) override;

private:
  template<typename T, bool trans>
  void
  compute_impl(tensors::TensorReadOnly2D<T, trans> inputs,
               tensors::TensorWriteable2D<T> outputs);

  template<typename T, bool trans>
  void
  derivative_impl(tensors::TensorReadOnly2D<T, trans> inputs,
                  tensors::TensorWriteable2D<T> outputs);
};

template<typename T, bool trans>
void
SigmoidActivation::compute_impl(tensors::TensorReadOnly2D<T, trans> inputs,
                                tensors::TensorWriteable2D<T> outputs) {
  tensors::ops::negative_exp(inputs, outputs);
  tensors::ops::add(static_cast<T>(1.0), outputs);
  tensors::ops::reverse_scal(static_cast<T>(1.0), outputs);
}

template<typename T, bool trans>
void
SigmoidActivation::derivative_impl(tensors::TensorReadOnly2D<T, trans> inputs,
                                   tensors::TensorWriteable2D<T> outputs) {
  auto sigm = compute(inputs);
  tensors::ops::copy(sigm.tensor_view().to_read_only(), outputs);
  tensors::ops::add_negative(static_cast<T>(1.0), sigm.tensor_view());
  tensors::ops::element_wise_mul(sigm.tensor_view().to_read_only(), outputs);
}

} // perceptron
} // activations

#endif //PERCEPTRON_ACTIVATIONS_SIGMOIDACTIVATION_H
