#ifndef PERCEPTRON_TENSORS_TENSOR1D_H
#define PERCEPTRON_TENSORS_TENSOR1D_H

#include "perceptron/common/Common.h"
#include "perceptron/tensors/TensorHelper.h"
#include "perceptron/common/utils/MemoryUtils.h"

#include <cstddef>
#include <type_traits>
#include <memory>

namespace perceptron {
namespace tensors {

template<typename T, bool is_transposed>
class TensorReadOnly2D;

template<typename T>
class TensorWriteable2D;

template<typename T>
class TensorReadOnly1D final {
private:
  using TensorHelper = details::TensorHelper<T>;
public:
  using value_type = typename TensorHelper::value_type;
  using pointer_type = typename TensorHelper::pointer_type;
  using creference_type = typename TensorHelper::creference_type;

  DEVICE_CALLABLE
  TensorReadOnly1D(const value_type *p_array, size_type size, size_type stride);
  DEVICE_CALLABLE
  TensorReadOnly1D(const value_type *p_array, size_type size);
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
  operator[](size_type x) const noexcept;

  DEVICE_CALLABLE
  inline creference_type
  operator()(size_type x) const noexcept;

  DEVICE_CALLABLE
  inline creference_type
  operator()() const noexcept;

  DEVICE_CALLABLE
  inline const value_type *
  get(size_type x) const noexcept;

  DEVICE_CALLABLE
  inline const value_type *
  get() const noexcept;

  [[nodiscard]] DEVICE_CALLABLE
  inline size_type
  get_size() const noexcept;

  [[nodiscard]] DEVICE_CALLABLE
  inline size_type
  get_stride() const noexcept;

  DEVICE_CALLABLE
  inline TensorReadOnly2D<T, false>
  to_2d() const noexcept;

  DEVICE_CALLABLE
  inline TensorReadOnly2D<T, true>
  to_2d_t() const noexcept;

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
  TensorWriteable1D(pointer_type p_array, size_type size, size_type stride);
  DEVICE_CALLABLE
  TensorWriteable1D(pointer_type p_array, size_type size);
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
  operator[](size_type x) noexcept;

  DEVICE_CALLABLE
  inline creference_type
  operator[](size_type x) const noexcept;

  DEVICE_CALLABLE
  inline vref_type
  operator()(size_type x) noexcept;

  DEVICE_CALLABLE
  inline vref_type
  operator()() noexcept;

  DEVICE_CALLABLE
  inline creference_type
  operator()(size_type x) const noexcept;

  DEVICE_CALLABLE
  inline creference_type
  operator()() const noexcept;

  DEVICE_CALLABLE
  inline const value_type *
  get(size_type x) const noexcept;

  DEVICE_CALLABLE
  inline const value_type *
  get() const noexcept;

  DEVICE_CALLABLE
  inline pointer_type
  get(size_type x) noexcept;

  DEVICE_CALLABLE
  inline pointer_type
  get() noexcept;

  [[nodiscard]] DEVICE_CALLABLE
  inline size_type
  get_size() const noexcept;

  [[nodiscard]] DEVICE_CALLABLE
  inline size_type
  get_stride() const noexcept;

  DEVICE_CALLABLE
  inline auto
  to_read_only() const noexcept;

  DEVICE_CALLABLE
  inline TensorWriteable2D<T>
  to_2d() const noexcept;

  DEVICE_CALLABLE
  inline
  operator TensorReadOnly1D<T>() const;
private:
  pointer_type m_p_array{nullptr};
  size_type m_size{};
  size_type m_stride{};
};

template<typename T>
class TensorOwner1D final {
private:
  using owned_ptr_type = std::unique_ptr<T, std::function<void(void *)>>;
  using view_type = TensorWriteable1D<std::remove_extent_t<T>>;
public:
  TensorOwner1D();
  TensorOwner1D(owned_ptr_type &&owned_ptr, size_type size, size_type stride);
  TensorOwner1D(owned_ptr_type &&owned_ptr, size_type size);

  TensorOwner1D(const TensorOwner1D &) = delete;
  TensorOwner1D &operator=(const TensorOwner1D &) = delete;

  TensorOwner1D(TensorOwner1D &&) noexcept = default;
  TensorOwner1D &operator=(TensorOwner1D &&) noexcept = default;
  ~TensorOwner1D() noexcept = default;

  owned_ptr_type
  release() noexcept;

  view_type
  tensor_view() const noexcept;

  void
  to_host(cudaStream_t stream = nullptr);

  void
  to_device(cudaStream_t stream = nullptr);

  void
  to_pinned(cudaStream_t stream = nullptr);

private:
  owned_ptr_type m_owned_ptr;
  view_type m_tensor_view;
};

inline constexpr size_type DEFAULT_1D_STRIDE = 1;

template<typename T>
auto
constructTensorReadOnly1D(const T *p_tensor, size_type size, size_type stride = DEFAULT_1D_STRIDE);

template<typename T>
auto
constructTensorWriteable1D(T *p_tensor, size_type size, size_type stride = DEFAULT_1D_STRIDE);

template<typename T, typename Deleter>
auto
constructTensorOwner1D(std::unique_ptr<T, Deleter> &&owned_ptr, size_type size, size_type stride = DEFAULT_1D_STRIDE);

template<typename T>
auto
constructTensorOwnerHost1D(size_type size, size_type stride = DEFAULT_1D_STRIDE, cudaStream_t stream = nullptr);

template<typename T>
auto
constructTensorOwnerHost1D(size_type size, cudaStream_t stream);

template<typename T>
auto
constructTensorOwnerDevice1D(size_type size, size_type stride = DEFAULT_1D_STRIDE, cudaStream_t stream = nullptr);

template<typename T>
auto
constructTensorOwnerDevice1D(size_type size, cudaStream_t stream);

} // perceptron
} // tensors

#include "perceptron/tensors/Tensor1DImpl.hpp"

#endif //PERCEPTRON_TENSORS_TENSOR1D_H
