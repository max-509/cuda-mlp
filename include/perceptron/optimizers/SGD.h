#ifndef PERCEPTRON_OPTIMIZERS_SGD_H
#define PERCEPTRON_OPTIMIZERS_SGD_H

#include "perceptron/common/Common.h"
#include "perceptron/optimizers/IOptimizer.h"
#include "perceptron/tensors/ops/MathOps.h"
#include "perceptron/tensors/ops/MemoryOps.h"
#include "perceptron/common/utils/StreamUtils.h"
#include "perceptron/common/utils/CuBLASUtils.h"

#include <optional>
#include <memory>

namespace perceptron {
namespace optimizers {

class SGD : public IOptimizer {
public:
  static constexpr auto DEFAULT_LR = 1e-5;
  static constexpr auto DEFAULT_WEIGHTS_DECAY = 0.0;
  static constexpr auto DEFAULT_MOMENTUM = 0.0;
  static constexpr auto DEFAULT_DAMPENING = 0.0;
  static constexpr auto DEFAULT_NESTEROV = false;

  struct describer_t {
    using type = SGD;

    double lr{DEFAULT_LR};
    double weights_decay{DEFAULT_WEIGHTS_DECAY};
    double momentum{DEFAULT_MOMENTUM};
    double dampening{DEFAULT_DAMPENING};
    bool nesterov{DEFAULT_NESTEROV};

    describer_t
    set_lr(double value);
    describer_t
    set_weights_decay(double value);
    describer_t
    set_momentum(double value);
    describer_t
    set_dampening(double value);
    describer_t
    set_nesterov(bool value);
  };

  SGD() = default;
  SGD(double lr, double weights_decay, double momentum, double dampening, bool nesterov);
  explicit SGD(describer_t describer);

  void descent(tensors::TensorReadOnly2D<float, false> grads, tensors::TensorWriteable2D<float> weights) override;
  void descent(tensors::TensorReadOnly2D<float, true> grads, tensors::TensorWriteable2D<float> weights) override;
  void reset() override;

private:
  double m_lr{DEFAULT_LR};
  double m_weights_decay{DEFAULT_WEIGHTS_DECAY};
  double m_momentum{DEFAULT_MOMENTUM};
  double m_dampening{DEFAULT_DAMPENING};
  bool m_nesterov{DEFAULT_NESTEROV};

  std::optional<tensors::TensorOwner2D<float>> m_velocity{};

  template<bool trans_grads>
  void
  descent_impl(tensors::TensorReadOnly2D<float, trans_grads> grads, tensors::TensorWriteable2D<float> weights);
};

template<bool trans_grads>
void
SGD::descent_impl(tensors::TensorReadOnly2D<float, trans_grads> grads,
                  tensors::TensorWriteable2D<float> weights) {
  auto stream = utils::cu_create_stream();

  auto grads_buffer_owner = tensors::constructTensorOwnerDevice2D<float>(grads.get_nrows(), grads.get_ncols(), *stream);
  auto grads_buffer_view = grads_buffer_owner.tensor_view();

  tensors::ops::copy(grads, grads_buffer_view, *stream);

  if (m_weights_decay >= std::numeric_limits<double>::min()) {
    tensors::ops::geam(weights.to_read_only(), static_cast<float>(m_weights_decay),
                       grads_buffer_view, 1.0f, *stream);
  }
  if (m_momentum >= std::numeric_limits<double>::min()) {
    if (m_velocity.has_value()) {
      tensors::ops::geam(grads_buffer_view.to_read_only(), static_cast<float>(1.0 - m_dampening),
                         m_velocity->tensor_view(), static_cast<float>(m_momentum), *stream);
    } else {
      m_velocity = std::make_optional(tensors::constructTensorOwnerDevice2D<float>(grads_buffer_view.get_nrows(),
                                                                                   grads_buffer_view.get_ncols(),
                                                                                   *stream));
      tensors::ops::copy(grads_buffer_view.to_read_only(), m_velocity->tensor_view(), *stream);
    }

    if (m_nesterov) {
      tensors::ops::geam(m_velocity->tensor_view().to_read_only(), static_cast<float>(m_momentum),
                         grads_buffer_view, 1.0f, *stream);
    } else {
      tensors::ops::copy(m_velocity->tensor_view().to_read_only(), grads_buffer_view, *stream);
    }
  }

  tensors::ops::geam(grads_buffer_view.to_read_only(), -static_cast<float>(m_lr),
                     weights, 1.0f, *stream);

  utils::cu_wait_stream(stream);
}

} // perceptron
} // optimizers

#endif //PERCEPTRON_OPTIMIZERS_SGD_H
