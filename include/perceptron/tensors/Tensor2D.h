#ifndef PERCEPTRON_TENSORS_TENSOR2D_H
#define PERCEPTRON_TENSORS_TENSOR2D_H

#include "perceptron/common/Common.h"
#include "perceptron/common/utils/MemoryUtils.h"
#include "perceptron/tensors/TensorHelper.h"

#include <cstddef>
#include <type_traits>
#include <memory>
#include <sstream>
#include <string>

#include <cuda_runtime.h>

namespace perceptron {
namespace tensors {

template<typename T>
class TensorReadOnly1D;

template<typename T>
class TensorWriteable1D;

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
                   size_type stride);
  DEVICE_CALLABLE
  TensorReadOnly2D(const value_type *p_array, size_type y_dim, size_type x_dim);
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
  operator()(size_type y, size_type x) const noexcept;

  DEVICE_CALLABLE
  inline creference_type
  operator()(size_type y) const noexcept;

  DEVICE_CALLABLE
  inline creference_type
  operator()() const noexcept;

  DEVICE_CALLABLE
  inline const value_type *
  get(size_type y, size_type x) const noexcept;

  DEVICE_CALLABLE
  inline const value_type *
  get(size_type y) const noexcept;

  DEVICE_CALLABLE
  inline const value_type *
  get() const noexcept;

  DEVICE_CALLABLE
  inline TensorReadOnly1D<T>
  get_row(size_type i) const noexcept;

  DEVICE_CALLABLE
  inline TensorReadOnly1D<T>
  get_col(size_type i) const noexcept;

  [[nodiscard]] DEVICE_CALLABLE
  inline size_type
  get_nrows() const noexcept;

  [[nodiscard]] DEVICE_CALLABLE
  inline size_type
  get_ncols() const noexcept;

  [[nodiscard]] DEVICE_CALLABLE
  inline size_type
  get_y_dim() const noexcept;

  [[nodiscard]] DEVICE_CALLABLE
  inline size_type
  get_x_dim() const noexcept;

  [[nodiscard]] DEVICE_CALLABLE
  inline size_type
  get_stride() const noexcept;

  [[nodiscard]]
  inline std::string
  shape_repr() const;

  [[nodiscard]] DEVICE_CALLABLE
  constexpr size_type
  transposed() const noexcept;

  DEVICE_CALLABLE
  inline TensorReadOnly2D<T, !is_transposed>
  t() const noexcept;

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
                    size_type stride);
  DEVICE_CALLABLE
  TensorWriteable2D(pointer_type p_array, size_type y_dim, size_type x_dim);
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
  operator()(size_type y, size_type x) noexcept;

  DEVICE_CALLABLE
  inline vref_type
  operator()(size_type y) noexcept;

  DEVICE_CALLABLE
  inline vref_type
  operator()() noexcept;

  DEVICE_CALLABLE
  inline creference_type
  operator()(size_type y, size_type x) const noexcept;

  DEVICE_CALLABLE
  inline creference_type
  operator()(size_type y) const noexcept;

  DEVICE_CALLABLE
  inline creference_type
  operator()() const noexcept;

  DEVICE_CALLABLE
  inline pointer_type
  get(size_type y, size_type x) noexcept;

  DEVICE_CALLABLE
  inline pointer_type
  get(size_type y) noexcept;

  DEVICE_CALLABLE
  inline pointer_type
  get() noexcept;

  DEVICE_CALLABLE
  inline const value_type *
  get(size_type y, size_type x) const noexcept;

  DEVICE_CALLABLE
  inline const value_type *
  get(size_type y) const noexcept;

  DEVICE_CALLABLE
  inline const value_type *
  get() const noexcept;

  DEVICE_CALLABLE
  inline TensorWriteable1D<T>
  get_row(size_type i) const noexcept;

  DEVICE_CALLABLE
  inline TensorWriteable1D<T>
  get_col(size_type i) const noexcept;

  [[nodiscard]] DEVICE_CALLABLE
  inline size_type
  get_nrows() const noexcept;

  [[nodiscard]] DEVICE_CALLABLE
  inline size_type
  get_ncols() const noexcept;

  [[nodiscard]] DEVICE_CALLABLE
  inline size_type
  get_y_dim() const noexcept;

  [[nodiscard]] DEVICE_CALLABLE
  inline size_type
  get_x_dim() const noexcept;

  [[nodiscard]] DEVICE_CALLABLE
  inline size_type
  get_stride() const noexcept;

  [[nodiscard]]
  inline std::string
  shape_repr() const;

  DEVICE_CALLABLE
  inline auto
  to_read_only() const noexcept;

  DEVICE_CALLABLE
  inline
  operator TensorReadOnly2D<T, false>() const;

private:
  pointer_type m_p_array{nullptr};
  size_type m_y_dim{};
  size_type m_x_dim{};
  size_type m_stride{};
};

template<typename T>
class TensorOwner2D final {
private:
  using owned_ptr_type = std::unique_ptr<T, std::function<void(void *)>>;
  using view_type = TensorWriteable2D<std::remove_extent_t<T>>;
public:
  TensorOwner2D(owned_ptr_type &&owned_ptr,
                size_type y_dim, size_type x_dim,
                size_type stride);

  TensorOwner2D(owned_ptr_type &&owned_ptr, size_type y_dim, size_type x_dim);

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

inline constexpr size_type DEFAULT_2D_STRIDE = -1L;

template<typename T>
auto
constructTensorReadOnly2D(const T *p_tensor,
                          size_type y_dim,
                          size_type x_dim,
                          size_type stride = DEFAULT_2D_STRIDE);

template<typename T>
auto
constructTensorWriteable2D(T *p_tensor,
                           size_type y_dim,
                           size_type x_dim,
                           size_type stride = DEFAULT_2D_STRIDE);

template<typename T, typename Deleter>
auto
constructTensorOwner2D(std::unique_ptr<T, Deleter> &&owned_ptr,
                       size_type y_dim,
                       size_type x_dim,
                       size_type stride = DEFAULT_2D_STRIDE);

template<typename T>
auto
constructTensorOwnerHost2D(size_type y_dim,
                           size_type x_dim,
                           size_type stride = DEFAULT_2D_STRIDE,
                           cudaStream_t stream = nullptr);

template<typename T>
auto
constructTensorOwnerHost2D(size_type y_dim,
                           size_type x_dim,
                           cudaStream_t stream);

template<typename T>
auto
constructTensorOwnerDevice2D(size_type y_dim,
                             size_type x_dim,
                             size_type stride = DEFAULT_2D_STRIDE,
                             cudaStream_t stream = nullptr);

template<typename T>
auto
constructTensorOwnerDevice2D(size_type y_dim,
                             size_type x_dim,
                             cudaStream_t stream);

} // perceptron
} // tensors

#include "perceptron/tensors/Tensor2DImpl.hpp"

#endif //PERCEPTRON_TENSORS_TENSOR2D_H
