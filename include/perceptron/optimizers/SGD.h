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
private:
  static constexpr auto DEFAULT_LR = 1e-5;
  static constexpr auto DEFAULT_WEIGHTS_DECAY = 0.0;
  static constexpr auto DEFAULT_MOMENTUM = 0.0;
  static constexpr auto DEFAULT_DAMPENING = 0.0;
  static constexpr auto DEFAULT_NESTEROV = false;
public:
  struct describer_t {
    using type = SGD;

    double lr{DEFAULT_LR};
    double weights_decay{DEFAULT_WEIGHTS_DECAY};
    double momentum{DEFAULT_MOMENTUM};
    double dampening{DEFAULT_DAMPENING};
    bool nesterov{DEFAULT_NESTEROV};
  };

  SGD() = default;
  SGD(double lr, double weights_decay, double momentum, double dampening, bool netsterov);
  explicit SGD(describer_t describer);

  void descent(tensors::TensorReadOnly2D<float, false> grads, tensors::TensorWriteable2D<float> weigths) override;
  void descent(tensors::TensorReadOnly2D<float, true> grads, tensors::TensorWriteable2D<float> weigths) override;
  void reset() override;

private:
  double m_lr{DEFAULT_LR};
  double m_weights_decay{DEFAULT_WEIGHTS_DECAY};
  double m_momentum{DEFAULT_MOMENTUM};
  double m_dampening{DEFAULT_DAMPENING};
  bool m_nesterov{DEFAULT_NESTEROV};

  std::optional<tensors::TensorOwnerDevice2D<float>> m_velocity{};

  template<bool trans_grads>
  void
  descent_impl(tensors::TensorReadOnly2D<float, trans_grads> grads, tensors::TensorWriteable2D<float> weigths);
};

template<bool trans_grads>
void
SGD::descent_impl(tensors::TensorReadOnly2D<float, trans_grads> grads,
                  tensors::TensorWriteable2D<float> weigths) {
  auto grads_buffer_owner = tensors::constructTensorOwnerDevice2D<float>(grads.get_nrows(), grads.get_ncols());
  auto grads_buffer_view = grads_buffer_owner.tensor_view();

  auto streams = utils::cu_create_streams(2);

  utils::CuBLASHandle::set_stream(*streams[0]);

  tensors::ops::copy(grads, grads_buffer_view, *streams[1]);

  if (m_weights_decay >= std::numeric_limits<double>::min()) {
    tensors::ops::geam(weigths.to_read_only(), static_cast<float>(m_weights_decay),
                       grads_buffer_view, 1.0f);
  }
  if (m_momentum >= std::numeric_limits<double>::min()) {
    if (m_velocity.has_value()) {
      tensors::ops::geam(grads_buffer_view.to_read_only(), static_cast<float>(1.0 - m_dampening),
                         m_velocity->tensor_view(), static_cast<float>(m_momentum));
    } else {
      m_velocity = std::make_optional(tensors::constructTensorOwnerDevice2D<float>(grads_buffer_view.get_nrows(),
                                                                                   grads_buffer_view.get_ncols()));
      tensors::ops::copy(grads_buffer_view.to_read_only(), m_velocity->tensor_view(), *streams[1]);
    }

    if (m_nesterov) {
      tensors::ops::geam(m_velocity->tensor_view().to_read_only(), static_cast<float>(m_momentum),
                         grads_buffer_view, 1.0f);
    } else {
      tensors::ops::copy(m_velocity->tensor_view().to_read_only(), grads_buffer_view, *streams[1]);
    }
  }

  tensors::ops::geam(grads_buffer_view.to_read_only(), static_cast<float>(m_lr),
                     weigths, -1.0f);

  utils::cu_wait_streams(streams);
}

} // perceptron
} // optimizers

#endif //PERCEPTRON_OPTIMIZERS_SGD_H
