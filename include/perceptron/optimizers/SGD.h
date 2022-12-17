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
  struct describer_t {
    using type = SGD;

    double m_lr{1e-5};
    double m_weights_decay{0.0};
    double m_momentum{0.0};
    double m_dampening{0.0};
    bool m_nesterov{false};

    [[nodiscard]] std::unique_ptr<IOptimizer>
    build() const {
      return std::make_unique<SGD>(m_lr, m_weights_decay, m_momentum, m_dampening, m_nesterov);
    }
  };

  SGD() = default;
  SGD(double lr, double weights_decay, double momentum, double dampening, bool netsterov);

  void descent(tensors::TensorReadOnly2D<float, false> grads, tensors::TensorWriteable2D<float> weigths) override;
  void descent(tensors::TensorReadOnly2D<float, true> grads, tensors::TensorWriteable2D<float> weigths) override;
  void reset() override;

private:
  double m_lr{1e-5};
  double m_weights_decay{0.0};
  double m_momentum{0.0};
  double m_dampening{0.0};
  bool m_nesterov{false};

  std::optional<tensors::TensorOwnerDevice2D<float>> m_velocity{};

  template<bool trans_grads>
  void
  descent_impl(tensors::TensorReadOnly2D<float, trans_grads> grads, tensors::TensorWriteable2D<float> weigths);
};

template<bool trans_grads>
void
SGD::descent_impl(tensors::TensorReadOnly2D<float, trans_grads> grads,
                  tensors::TensorWriteable2D<float> weigths) {
  auto grads_buffer_owner = tensors::constructTensorOwnerDevice2D<float>(grads.get_y_dim(), grads.get_x_dim());
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
      m_velocity = std::make_optional(tensors::constructTensorOwnerDevice2D<float>(grads_buffer_view.get_y_dim(),
                                                                                   grads_buffer_view.get_x_dim()));
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
