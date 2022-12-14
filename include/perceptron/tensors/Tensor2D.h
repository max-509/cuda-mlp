#ifndef PERCEPTRON_TENSORS_TENSOR2D_H
#define PERCEPTRON_TENSORS_TENSOR2D_H

#include "perceptron/common/Common.h"
#include "perceptron/common/utils/MemoryUtils.h"
#include "perceptron/tensors/TensorHelper.h"

#include <cstddef>
#include <type_traits>
#include <memory>

#include <cuda_runtime.h>

namespace perceptron {
namespace tensors {

template<typename T, bool is_transposed = false>
class TensorReadOnly2D final {
private:
  using TensorHelper = details::TensorHelper<T>;
public:
  using value_type = typename TensorHelper::value_type;
  using pointer_type = typename TensorHelper::pointer_type;
  using creference_type = typename TensorHelper::creference_type;

  DEVICE_CALLABLE
  TensorReadOnly2D(const value_type *p_array,
                   size_type y_dim,
                   size_type x_dim,
                   size_type stride)
      : m_p_array(p_array), m_y_dim(y_dim), m_x_dim(x_dim), m_stride(stride) {}
  DEVICE_CALLABLE
  TensorReadOnly2D(const value_type *p_array, size_type y_dim, size_type x_dim)
      : TensorReadOnly2D(p_array, y_dim, x_dim, x_dim) {}
  DEVICE_CALLABLE
  TensorReadOnly2D(const TensorReadOnly2D &) = default;
  DEVICE_CALLABLE
  TensorReadOnly2D &operator=(const TensorReadOnly2D &) = default;
  DEVICE_CALLABLE
  TensorReadOnly2D &operator=(TensorReadOnly2D &&) noexcept = default;
  DEVICE_CALLABLE
  TensorReadOnly2D(TensorReadOnly2D &&) noexcept = default;
  DEVICE_CALLABLE
  ~TensorReadOnly2D() noexcept = default;

  DEVICE_CALLABLE
  inline creference_type
  operator()(size_type y, size_type x) const noexcept {
    return *get(y, x);
  }

  DEVICE_CALLABLE
  inline creference_type
  operator()(size_type y) const noexcept {
    return *get(y, 0);
  }

  DEVICE_CALLABLE
  inline creference_type
  operator()() const noexcept {
    return *m_p_array;
  }

  DEVICE_CALLABLE
  inline const value_type *
  get(size_type y, size_type x) const noexcept {
    if constexpr (is_transposed) {
      return m_p_array + y + m_stride * x;
    } else {
      return m_p_array + m_stride * y + x;
    }
  }

  DEVICE_CALLABLE
  inline const value_type *
  get(size_type y) const noexcept {
    return get(y, 0);
  }

  DEVICE_CALLABLE
  inline const value_type *
  get() const noexcept {
    return m_p_array;
  }

  DEVICE_CALLABLE
  [[nodiscard]] inline size_type
  get_y_dim() const noexcept { return m_y_dim; }

  DEVICE_CALLABLE
  [[nodiscard]] inline size_type
  get_x_dim() const noexcept { return m_x_dim; }

  DEVICE_CALLABLE
  [[nodiscard]] inline size_type
  get_stride() const noexcept { return m_stride; }

  DEVICE_CALLABLE
  [[nodiscard]] inline size_type
  transposed() const noexcept { return is_transposed; }

  DEVICE_CALLABLE
  inline TensorReadOnly2D<T, !is_transposed>
  t() const noexcept {
    return TensorReadOnly2D<T, !is_transposed>(m_p_array, m_x_dim, m_y_dim, m_stride);
  }

private:
  const value_type *m_p_array{nullptr};
  size_type m_y_dim{};
  size_type m_x_dim{};
  size_type m_stride{};
};

template<typename T>
class TensorWriteable2D final {
private:
  using TensorHelper = details::TensorHelper<T>;
public:
  using value_type = typename TensorHelper::value_type;
  using pointer_type = typename TensorHelper::pointer_type;
  using creference_type = typename TensorHelper::creference_type;
  using reference_type = typename TensorHelper::reference_type;
  using vref_type = typename TensorHelper::vref_type;

  DEVICE_CALLABLE
  TensorWriteable2D(pointer_type p_array,
                    size_type y_dim,
                    size_type x_dim,
                    size_type stride)
      : m_p_array(p_array), m_y_dim(y_dim), m_x_dim(x_dim), m_stride(stride) {}
  DEVICE_CALLABLE
  TensorWriteable2D(pointer_type p_array, size_type y_dim, size_type x_dim)
      : TensorWriteable2D(p_array, y_dim, x_dim, x_dim) {}
  DEVICE_CALLABLE
  TensorWriteable2D(const TensorWriteable2D &) = default;
  DEVICE_CALLABLE
  TensorWriteable2D &operator=(const TensorWriteable2D &) = default;
  DEVICE_CALLABLE
  TensorWriteable2D &operator=(TensorWriteable2D &&) noexcept = default;
  DEVICE_CALLABLE
  TensorWriteable2D(TensorWriteable2D &&) noexcept = default;
  DEVICE_CALLABLE
  ~TensorWriteable2D() noexcept = default;

  DEVICE_CALLABLE
  inline vref_type
  operator()(size_type y, size_type x) noexcept {
    return *get(y, x);
  }

  DEVICE_CALLABLE
  inline vref_type
  operator()(size_type y) noexcept {
    return *get(y, 0);
  }

  DEVICE_CALLABLE
  inline vref_type
  operator()() noexcept {
    return *m_p_array;
  }

  DEVICE_CALLABLE
  inline creference_type
  operator()(size_type y, size_type x) const noexcept {
    return *get(y, x);
  }

  DEVICE_CALLABLE
  inline creference_type
  operator()(size_type y) const noexcept {
    return *get(y, 0);
  }

  DEVICE_CALLABLE
  inline creference_type
  operator()() const noexcept {
    return *m_p_array;
  }

  DEVICE_CALLABLE
  inline pointer_type
  get(size_type y, size_type x) noexcept {
    return m_p_array + m_stride * y + x;
  }

  DEVICE_CALLABLE
  inline pointer_type
  get(size_type y) noexcept {
    return get(y, 0);
  }

  DEVICE_CALLABLE
  inline pointer_type
  get() noexcept {
    return m_p_array;
  }

  DEVICE_CALLABLE
  inline const value_type *
  get(size_type y, size_type x) const noexcept {
    return m_p_array + m_stride * y + x;
  }

  DEVICE_CALLABLE
  inline const value_type *
  get(size_type y) const noexcept {
    return *get(y, 0);
  }

  DEVICE_CALLABLE
  inline const value_type *
  get() const noexcept {
    return m_p_array;
  }

  DEVICE_CALLABLE
  [[nodiscard]] inline size_type
  get_y_dim() const noexcept {
    return m_y_dim;
  }

  DEVICE_CALLABLE
  [[nodiscard]] inline size_type
  get_x_dim() const noexcept { return m_x_dim; }

  DEVICE_CALLABLE
  [[nodiscard]] inline size_type
  get_stride() const noexcept { return m_stride; }

  DEVICE_CALLABLE
  inline auto
  to_read_only() const noexcept {
    return TensorReadOnly2D<T, false>(m_p_array, m_y_dim, m_x_dim, m_stride);
  }

  DEVICE_CALLABLE
  inline
  operator TensorReadOnly2D<T, false>() const {
    return to_read_only();
  }

private:
  pointer_type m_p_array{nullptr};
  size_type m_y_dim{};
  size_type m_x_dim{};
  size_type m_stride{};
};

template<typename T>
auto
constructTensorReadOnly2D(const T *p_tensor,
                          size_type y_dim,
                          size_type x_dim,
                          size_type stride = -1L) {
  if (stride == -1) {
    stride = x_dim;
  }
  return TensorReadOnly2D<T>{p_tensor, y_dim, x_dim, stride};
}

template<typename T>
auto
constructTensorWriteable2D(T *p_tensor,
                           size_type y_dim,
                           size_type x_dim,
                           size_type stride = -1L) {
  if (stride == -1) {
    stride = x_dim;
  }
  return TensorWriteable2D<T>{p_tensor, y_dim, x_dim, stride};
}

template<typename T, typename Deleter>
class TensorOwner2D final {
private:
  using owned_ptr_type = std::unique_ptr<T, Deleter>;
  using view_type = TensorWriteable2D<std::remove_extent_t<T>>;
public:
  TensorOwner2D(owned_ptr_type &&owned_ptr,
                size_type y_dim, size_type x_dim,
                size_type stride)
      : m_owned_ptr(std::move(owned_ptr)), m_tensor_view(m_owned_ptr.get(), y_dim, x_dim, stride) {}

  TensorOwner2D(owned_ptr_type &&owned_ptr, size_type y_dim, size_type x_dim)
      : TensorOwner2D(std::move(owned_ptr), y_dim, x_dim, x_dim) {}

  TensorOwner2D(const TensorOwner2D &) = delete;
  TensorOwner2D &operator=(const TensorOwner2D &) = delete;

  TensorOwner2D(TensorOwner2D &&) noexcept = default;
  TensorOwner2D &operator=(TensorOwner2D &&) noexcept = default;
  ~TensorOwner2D() noexcept = default;

  owned_ptr_type
  release() noexcept {
    return std::move(m_owned_ptr);
  }

  view_type
  tensor_view() const noexcept {
    return m_tensor_view;
  }

private:
  owned_ptr_type m_owned_ptr;
  view_type m_tensor_view;
};

template<typename T>
using TensorOwnerHost2D = TensorOwner2D<T, utils::cu_host_deleter>;

template<typename T>
using TensorOwnerDevice2D = TensorOwner2D<T, utils::cu_memory_deleter>;

template<typename T, typename Deleter>
auto
constructTensorOwner2D(std::unique_ptr<T, Deleter> &&owned_ptr,
                       size_type y_dim,
                       size_type x_dim,
                       size_type stride = -1L) {
  if (stride == -1) {
    stride = x_dim;
  }
  return TensorOwner2D<T, Deleter>(std::move(owned_ptr), y_dim, x_dim, stride);
}

template<typename T>
auto
constructTensorOwnerHost2D(size_type y_dim,
                           size_type x_dim,
                           size_type stride = -1L) {
  if (stride == -1) {
    stride = x_dim;
  }
  auto ptr = utils::cu_make_host_memory_unique<T>(y_dim * stride);
  return TensorOwner2D<T, utils::cu_host_deleter>(std::move(ptr),
                                                  y_dim, x_dim, stride);
}

template<typename T>
auto
constructTensorOwnerDevice2D(size_type y_dim,
                             size_type x_dim,
                             size_type stride = -1L) {
  if (stride == -1) {
    std::size_t pitch{};
    auto ptr = utils::cu_make_pitched_memory_unique<T>(y_dim, x_dim, pitch);
    stride = static_cast<size_type>(pitch);
    return TensorOwner2D<T, utils::cu_memory_deleter>(std::move(ptr),
                                                      y_dim, x_dim, stride);
  } else {
    auto ptr = utils::cu_make_memory_unique<T>(y_dim * stride);
    return TensorOwner2D<T, utils::cu_memory_deleter>(std::move(ptr),
                                                      y_dim, x_dim, stride);
  }
}

} // perceptron
} // tensors

#endif //PERCEPTRON_TENSORS_TENSOR2D_H
