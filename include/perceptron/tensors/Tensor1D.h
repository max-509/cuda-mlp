#ifndef PERCEPTRON_TENSORS_TENSOR1D_H
#define PERCEPTRON_TENSORS_TENSOR1D_H

#include "perceptron/common/Common.h"
#include "perceptron/tensors/TensorHelper.h"
#include "perceptron/tensors/Tensor2D.h"

#include <cstddef>
#include <type_traits>
#include <memory>

namespace perceptron {
namespace tensors {

template<typename T>
class TensorReadOnly1D final {
private:
  using TensorHelper = details::TensorHelper<T>;
public:
  using value_type = typename TensorHelper::value_type;
  using pointer_type = typename TensorHelper::pointer_type;
  using creference_type = typename TensorHelper::creference_type;

  DEVICE_CALLABLE
  TensorReadOnly1D(const value_type *p_array, size_type size, size_type stride)
      : m_p_array(p_array), m_size(size), m_stride(stride) {}
  DEVICE_CALLABLE
  TensorReadOnly1D(const value_type *p_array, size_type size) : TensorReadOnly1D(p_array, size, 1) {}
  DEVICE_CALLABLE
  TensorReadOnly1D(const TensorReadOnly1D &) = default;
  DEVICE_CALLABLE
  TensorReadOnly1D &operator=(const TensorReadOnly1D &) = default;
  DEVICE_CALLABLE
  TensorReadOnly1D &operator=(TensorReadOnly1D &&) noexcept = default;
  DEVICE_CALLABLE
  TensorReadOnly1D(TensorReadOnly1D &&) noexcept = default;
  DEVICE_CALLABLE
  ~TensorReadOnly1D() noexcept = default;

  DEVICE_CALLABLE
  inline creference_type
  operator[](size_type x) const noexcept {
    return m_p_array[x * m_stride];
  }

  DEVICE_CALLABLE
  inline creference_type
  operator()(size_type x) const noexcept {
    return m_p_array[x * m_stride];
  }

  DEVICE_CALLABLE
  inline creference_type
  operator()() const noexcept {
    return *m_p_array;
  }

  DEVICE_CALLABLE
  inline const value_type *
  get(size_type x) const noexcept {
    return m_p_array + x * m_stride;
  }

  DEVICE_CALLABLE
  inline const value_type *
  get() const noexcept {
    return m_p_array;
  }

  DEVICE_CALLABLE
  inline size_type
  get_size() const noexcept { return m_size; }

  DEVICE_CALLABLE
  inline size_type
  get_stride() const noexcept { return m_stride; }

  DEVICE_CALLABLE
  inline TensorReadOnly2D<T, false>
  to_2d() const noexcept {
    return TensorReadOnly2D<T, false>(m_p_array, m_size, 1, m_stride);
  }

  DEVICE_CALLABLE
  inline TensorReadOnly2D<T, true>
  to_2d_t() const noexcept {
    return TensorReadOnly2D<T, true>(m_p_array, 1, m_size, m_stride);
  }

private:
  const value_type *m_p_array{nullptr};
  size_type m_size{};
  size_type m_stride{};
};

template<typename T>
class TensorWriteable1D final {
private:
  using TensorHelper = details::TensorHelper<T>;
public:
  using value_type = typename TensorHelper::value_type;
  using pointer_type = typename TensorHelper::pointer_type;
  using creference_type = typename TensorHelper::creference_type;
  using reference_type = typename TensorHelper::reference_type;
  using vref_type = typename TensorHelper::vref_type;

  DEVICE_CALLABLE
  TensorWriteable1D(pointer_type p_array, size_type size, size_type stride)
      : m_p_array(p_array), m_size(size), m_stride(stride) {}
  DEVICE_CALLABLE
  TensorWriteable1D(pointer_type p_array, size_type size) : TensorWriteable1D(p_array, size, 1) {}
  DEVICE_CALLABLE
  TensorWriteable1D(const TensorWriteable1D &) = default;
  DEVICE_CALLABLE
  TensorWriteable1D &operator=(const TensorWriteable1D &) = default;
  DEVICE_CALLABLE
  TensorWriteable1D &operator=(TensorWriteable1D &&) noexcept = default;
  DEVICE_CALLABLE
  TensorWriteable1D(TensorWriteable1D &&) noexcept = default;
  DEVICE_CALLABLE
  ~TensorWriteable1D() noexcept = default;

  DEVICE_CALLABLE
  inline vref_type
  operator[](size_type x) noexcept {
    return m_p_array[x * m_stride];
  }

  DEVICE_CALLABLE
  inline creference_type
  operator[](size_type x) const noexcept {
    return m_p_array[x * m_stride];
  }

  DEVICE_CALLABLE
  inline vref_type
  operator()(size_type x) noexcept {
    return m_p_array[x * m_stride];
  }

  DEVICE_CALLABLE
  inline vref_type
  operator()() noexcept {
    return *m_p_array;
  }

  DEVICE_CALLABLE
  inline creference_type
  operator()(size_type x) const noexcept {
    return m_p_array[x * m_stride];
  }

  DEVICE_CALLABLE
  inline creference_type
  operator()() const noexcept {
    return *m_p_array;
  }

  DEVICE_CALLABLE
  inline const value_type *
  get(size_type x) const noexcept {
    return m_p_array + x * m_stride;
  }

  DEVICE_CALLABLE
  inline const value_type *
  get() const noexcept {
    return m_p_array;
  }

  DEVICE_CALLABLE
  inline pointer_type
  get(size_type x) noexcept {
    return m_p_array + x * m_stride;
  }

  DEVICE_CALLABLE
  inline pointer_type
  get() noexcept {
    return m_p_array;
  }

  DEVICE_CALLABLE
  inline size_type
  get_size() const noexcept { return m_size; }

  DEVICE_CALLABLE
  inline size_type
  get_stride() const noexcept { return m_stride; }

  DEVICE_CALLABLE
  inline auto
  to_read_only() const noexcept {
    return TensorReadOnly1D<T>(m_p_array, m_size, m_stride);
  }

  DEVICE_CALLABLE
  inline TensorWriteable2D<T>
  to_2d() const noexcept {
    return TensorWriteable2D<T>(m_p_array, m_size, 1, m_stride);
  }

  DEVICE_CALLABLE
  inline
  operator TensorReadOnly1D<T>() const {
    return to_read_only();
  }
private:
  pointer_type m_p_array{nullptr};
  size_type m_size{};
  size_type m_stride{};
};

template<typename T, typename Deleter>
class TensorOwner1D final {
private:
  using owned_ptr_type = std::unique_ptr<T, Deleter>;
  using view_type = TensorWriteable1D<std::remove_extent_t<T>>;
public:
  TensorOwner1D()
      : m_owned_ptr(owned_ptr_type{nullptr}), m_tensor_view(m_owned_ptr.get(), 0, 0) {}

  TensorOwner1D(owned_ptr_type &&owned_ptr, size_type size, size_type stride)
      : m_owned_ptr(std::move(owned_ptr)), m_tensor_view(m_owned_ptr.get(), size, stride) {}

  TensorOwner1D(owned_ptr_type &&owned_ptr, size_type size)
      : TensorOwner1D(std::move(owned_ptr), size, 1) {}

  TensorOwner1D(const TensorOwner1D &) = delete;
  TensorOwner1D &operator=(const TensorOwner1D &) = delete;

  TensorOwner1D(TensorOwner1D &&) noexcept = default;
  TensorOwner1D &operator=(TensorOwner1D &&) noexcept = default;
  ~TensorOwner1D() noexcept = default;

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
auto
constructTensorReadOnly1D(const T *p_tensor, size_type size, size_type stride = 1) {
  return TensorReadOnly1D<T>{p_tensor, size, stride};
}

template<typename T>
auto
constructTensorWriteable1D(T *p_tensor, size_type size, size_type stride = 1) {
  return TensorWriteable1D<T>{p_tensor, size, stride};
}

template<typename T, typename Deleter>
auto
constructTensorOwner1D(std::unique_ptr<T, Deleter> &&owned_ptr, size_type size, size_type stride = 1) {
  return TensorOwner1D<T, Deleter>(std::move(owned_ptr), size, stride);
}

template<typename T>
auto
constructTensorOwnerHost1D(size_type size, size_type stride = 1) {
  auto ptr = utils::cu_make_host_memory_unique<T>(size * stride);
  return TensorOwner1D<T, utils::cu_host_deleter>(std::move(ptr), size, stride);
}

template<typename T>
auto
constructTensorOwnerDevice1D(size_type size, size_type stride = 1) {
  auto ptr = utils::cu_make_memory_unique<T>(size * stride);
  return TensorOwner1D<T, utils::cu_memory_deleter>(std::move(ptr), size, stride);
}

template<typename T>
using TensorOwnerHost1D = TensorOwner1D<T, utils::cu_host_deleter>;

template<typename T>
using TensorOwnerDevice1D = TensorOwner1D<T, utils::cu_memory_deleter>;

} // perceptron
} // tensors

#endif //PERCEPTRON_TENSORS_TENSOR1D_H
