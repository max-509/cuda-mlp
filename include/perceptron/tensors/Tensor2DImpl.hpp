#ifndef PERCEPTRON_TENSORS_TENSOR2DIMPL_HPP
#define PERCEPTRON_TENSORS_TENSOR2DIMPL_HPP

#include "perceptron/tensors/Tensor1D.h"

namespace perceptron {
namespace tensors {

template<typename T, bool is_transposed>
DEVICE_CALLABLE
TensorReadOnly2D<T, is_transposed>::TensorReadOnly2D(const value_type *p_array,
                                                     size_type y_dim,
                                                     size_type x_dim,
                                                     size_type stride)
    : m_p_array(p_array), m_y_dim(y_dim), m_x_dim(x_dim), m_stride(stride) {}

template<typename T, bool is_transposed>
DEVICE_CALLABLE
TensorReadOnly2D<T, is_transposed>::TensorReadOnly2D(const value_type *p_array, size_type y_dim, size_type x_dim)
    : TensorReadOnly2D(p_array, y_dim, x_dim, x_dim) {}

template<typename T, bool is_transposed>
DEVICE_CALLABLE
inline typename TensorReadOnly2D<T, is_transposed>::creference_type
TensorReadOnly2D<T, is_transposed>::operator()(size_type y, size_type x) const noexcept {
  return *get(y, x);
}

template<typename T, bool is_transposed>
DEVICE_CALLABLE
inline typename TensorReadOnly2D<T, is_transposed>::creference_type
TensorReadOnly2D<T, is_transposed>::operator()(size_type y) const noexcept {
  return *get(y, 0);
}

template<typename T, bool is_transposed>
DEVICE_CALLABLE
inline typename TensorReadOnly2D<T, is_transposed>::creference_type
TensorReadOnly2D<T, is_transposed>::operator()() const noexcept {
  return *m_p_array;
}

template<typename T, bool is_transposed>
DEVICE_CALLABLE
inline const typename TensorReadOnly2D<T, is_transposed>::value_type *
TensorReadOnly2D<T, is_transposed>::get(size_type y, size_type x) const noexcept {
  if constexpr (is_transposed) {
    return m_p_array + y + m_stride * x;
  } else {
    return m_p_array + m_stride * y + x;
  }
}

template<typename T, bool is_transposed>
DEVICE_CALLABLE
inline const typename TensorReadOnly2D<T, is_transposed>::value_type *
TensorReadOnly2D<T, is_transposed>::get(size_type y) const noexcept {
  return get(y, 0);
}

template<typename T, bool is_transposed>
DEVICE_CALLABLE
inline const typename TensorReadOnly2D<T, is_transposed>::value_type *
TensorReadOnly2D<T, is_transposed>::get() const noexcept {
  return m_p_array;
}

template<typename T, bool is_transposed>
DEVICE_CALLABLE
inline TensorReadOnly1D<T>
TensorReadOnly2D<T, is_transposed>::get_row(size_type i) const noexcept {
  if constexpr (is_transposed) {
    return TensorReadOnly1D<T>(m_p_array + i, get_ncols(), get_stride());
  } else {
    return TensorReadOnly1D<T>(m_p_array + i * get_stride(), get_ncols(), 1);
  }
}

template<typename T, bool is_transposed>
DEVICE_CALLABLE
inline TensorReadOnly1D<T>
TensorReadOnly2D<T, is_transposed>::get_col(size_type i) const noexcept {
  if constexpr (is_transposed) {
    return TensorReadOnly1D<T>(m_p_array + i * get_stride(), get_nrows(), 1);
  } else {
    return TensorReadOnly1D<T>(m_p_array + i, get_nrows(), get_stride());
  }
}

template<typename T, bool is_transposed>
[[nodiscard]] DEVICE_CALLABLE
inline size_type
TensorReadOnly2D<T, is_transposed>::get_nrows() const noexcept {
  if constexpr (!is_transposed) {
    return get_y_dim();
  } else {
    return get_x_dim();
  }
}

template<typename T, bool is_transposed>
[[nodiscard]] DEVICE_CALLABLE
inline size_type
TensorReadOnly2D<T, is_transposed>::get_ncols() const noexcept {
  if constexpr (!is_transposed) {
    return get_x_dim();
  } else {
    return get_y_dim();
  }
}

template<typename T, bool is_transposed>
[[nodiscard]] DEVICE_CALLABLE
inline size_type
TensorReadOnly2D<T, is_transposed>::get_y_dim() const noexcept { return m_y_dim; }

template<typename T, bool is_transposed>
[[nodiscard]] DEVICE_CALLABLE
inline size_type
TensorReadOnly2D<T, is_transposed>::get_x_dim() const noexcept { return m_x_dim; }

template<typename T, bool is_transposed>
[[nodiscard]] DEVICE_CALLABLE
inline size_type
TensorReadOnly2D<T, is_transposed>::get_stride() const noexcept { return m_stride; }

template<typename T, bool is_transposed>
[[nodiscard]]
inline std::string
TensorReadOnly2D<T, is_transposed>::shape_repr() const {
  std::stringstream out_stream;
  out_stream << "(" << get_nrows() << ", " << get_ncols() << ")";
  return out_stream.str();
}

template<typename T, bool is_transposed>
[[nodiscard]] DEVICE_CALLABLE
constexpr size_type
TensorReadOnly2D<T, is_transposed>::transposed() const noexcept { return is_transposed; }

template<typename T, bool is_transposed>
DEVICE_CALLABLE
inline TensorReadOnly2D<T, !is_transposed>
TensorReadOnly2D<T, is_transposed>::t() const noexcept {
  return TensorReadOnly2D<T, !is_transposed>(m_p_array, m_x_dim, m_y_dim, m_stride);
}

template<typename T>
DEVICE_CALLABLE
TensorWriteable2D<T>::TensorWriteable2D(pointer_type p_array,
                                        size_type y_dim,
                                        size_type x_dim,
                                        size_type stride)
    : m_p_array(p_array), m_y_dim(y_dim), m_x_dim(x_dim), m_stride(stride) {}

template<typename T>
DEVICE_CALLABLE
TensorWriteable2D<T>::TensorWriteable2D(pointer_type p_array, size_type y_dim, size_type x_dim)
    : TensorWriteable2D(p_array, y_dim, x_dim, x_dim) {}

template<typename T>
DEVICE_CALLABLE
inline typename TensorWriteable2D<T>::vref_type
TensorWriteable2D<T>::operator()(size_type y, size_type x) noexcept {
  return *get(y, x);
}

template<typename T>
DEVICE_CALLABLE
inline typename TensorWriteable2D<T>::vref_type
TensorWriteable2D<T>::operator()(size_type y) noexcept {
  return *get(y, 0);
}

template<typename T>
DEVICE_CALLABLE
inline typename TensorWriteable2D<T>::vref_type
TensorWriteable2D<T>::operator()() noexcept {
  return *m_p_array;
}

template<typename T>
DEVICE_CALLABLE
inline typename TensorWriteable2D<T>::creference_type
TensorWriteable2D<T>::operator()(size_type y, size_type x) const noexcept {
  return *get(y, x);
}

template<typename T>
DEVICE_CALLABLE
inline typename TensorWriteable2D<T>::creference_type
TensorWriteable2D<T>::operator()(size_type y) const noexcept {
  return *get(y, 0);
}

template<typename T>
DEVICE_CALLABLE
inline typename TensorWriteable2D<T>::creference_type
TensorWriteable2D<T>::operator()() const noexcept {
  return *m_p_array;
}

template<typename T>
DEVICE_CALLABLE
inline typename TensorWriteable2D<T>::pointer_type
TensorWriteable2D<T>::get(size_type y, size_type x) noexcept {
  return m_p_array + m_stride * y + x;
}

template<typename T>
DEVICE_CALLABLE
inline typename TensorWriteable2D<T>::pointer_type
TensorWriteable2D<T>::get(size_type y) noexcept {
  return get(y, 0);
}

template<typename T>
DEVICE_CALLABLE
inline typename TensorWriteable2D<T>::pointer_type
TensorWriteable2D<T>::get() noexcept {
  return m_p_array;
}

template<typename T>
DEVICE_CALLABLE
inline const typename TensorWriteable2D<T>::value_type *
TensorWriteable2D<T>::get(size_type y, size_type x) const noexcept {
  return m_p_array + m_stride * y + x;
}

template<typename T>
DEVICE_CALLABLE
inline const typename TensorWriteable2D<T>::value_type *
TensorWriteable2D<T>::get(size_type y) const noexcept {
  return *get(y, 0);
}

template<typename T>
DEVICE_CALLABLE
inline const typename TensorWriteable2D<T>::value_type *
TensorWriteable2D<T>::get() const noexcept {
  return m_p_array;
}

template<typename T>
DEVICE_CALLABLE
inline TensorWriteable1D<T>
TensorWriteable2D<T>::get_row(size_type i) const noexcept {
  return TensorReadOnly1D<T>(m_p_array + i * get_stride(), get_ncols(), 1);
}

template<typename T>
DEVICE_CALLABLE
inline TensorWriteable1D<T>
TensorWriteable2D<T>::get_col(size_type i) const noexcept {
  return TensorReadOnly1D<T>(m_p_array + i, get_nrows(), get_stride());
}

template<typename T>
[[nodiscard]] DEVICE_CALLABLE
inline size_type
TensorWriteable2D<T>::get_nrows() const noexcept { return get_y_dim(); }

template<typename T>
[[nodiscard]] DEVICE_CALLABLE
inline size_type
TensorWriteable2D<T>::get_ncols() const noexcept { return get_x_dim(); }

template<typename T>
[[nodiscard]] DEVICE_CALLABLE
inline size_type
TensorWriteable2D<T>::get_y_dim() const noexcept {
  return m_y_dim;
}

template<typename T>
[[nodiscard]] DEVICE_CALLABLE
inline size_type
TensorWriteable2D<T>::get_x_dim() const noexcept { return m_x_dim; }

template<typename T>
[[nodiscard]] DEVICE_CALLABLE
inline size_type
TensorWriteable2D<T>::get_stride() const noexcept { return m_stride; }

template<typename T>
[[nodiscard]]
inline std::string
TensorWriteable2D<T>::shape_repr() const {
  std::stringstream out_stream;
  out_stream << "(" << get_nrows() << ", " << get_ncols() << ")";
  return out_stream.str();
}

template<typename T>
DEVICE_CALLABLE
inline auto
TensorWriteable2D<T>::to_read_only() const noexcept {
  return TensorReadOnly2D<T, false>(m_p_array, m_y_dim, m_x_dim, m_stride);
}

template<typename T>
DEVICE_CALLABLE
inline
TensorWriteable2D<T>::operator TensorReadOnly2D<T, false>() const {
  return to_read_only();
}

template<typename T, typename Deleter>
TensorOwner2D<T, Deleter>::TensorOwner2D(owned_ptr_type &&owned_ptr,
                                         size_type y_dim, size_type x_dim,
                                         size_type stride)
    : m_owned_ptr(std::move(owned_ptr)), m_tensor_view(m_owned_ptr.get(), y_dim, x_dim, stride) {}

template<typename T, typename Deleter>
TensorOwner2D<T, Deleter>::TensorOwner2D(owned_ptr_type &&owned_ptr, size_type y_dim, size_type x_dim)
    : TensorOwner2D(std::move(owned_ptr), y_dim, x_dim, x_dim) {}

template<typename T, typename Deleter>
typename TensorOwner2D<T, Deleter>::owned_ptr_type
TensorOwner2D<T, Deleter>::release() noexcept {
  return std::move(m_owned_ptr);
}

template<typename T, typename Deleter>
typename TensorOwner2D<T, Deleter>::view_type
TensorOwner2D<T, Deleter>::tensor_view() const noexcept {
  return m_tensor_view;
}

template<typename T>
auto
constructTensorReadOnly2D(const T *p_tensor,
                          size_type y_dim,
                          size_type x_dim,
                          size_type stride) {
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
                           size_type stride) {
  if (stride == -1) {
    stride = x_dim;
  }
  return TensorWriteable2D<T>{p_tensor, y_dim, x_dim, stride};
}

template<typename T, typename Deleter>
auto
constructTensorOwner2D(std::unique_ptr<T, Deleter> &&owned_ptr,
                       size_type y_dim,
                       size_type x_dim,
                       size_type stride) {
  if (stride == -1) {
    stride = x_dim;
  }
  return TensorOwner2D<T, Deleter>(std::move(owned_ptr), y_dim, x_dim, stride);
}

template<typename T>
auto
constructTensorOwnerHost2D(size_type y_dim,
                           size_type x_dim,
                           size_type stride) {
  if (stride == -1) {
    stride = x_dim;
  }
  auto ptr = utils::cu_make_host_memory_unique<T>(y_dim * stride);
  return TensorOwner2D<T, utils::cu_host_deleter_t>(std::move(ptr),
                                                    y_dim, x_dim, stride);
}

template<typename T>
auto
constructTensorOwnerDevice2D(size_type y_dim,
                             size_type x_dim,
                             size_type stride) {
  if (stride == -1) {
    std::size_t pitch{};
    auto ptr = utils::cu_make_pitched_memory_unique<T>(y_dim, x_dim, pitch);
    stride = static_cast<size_type>(pitch);
    return TensorOwner2D<T, utils::cu_memory_deleter_t>(std::move(ptr),
                                                        y_dim, x_dim, stride);
  } else {
    auto ptr = utils::cu_make_memory_unique<T>(y_dim * stride);
    return TensorOwner2D<T, utils::cu_memory_deleter_t>(std::move(ptr),
                                                        y_dim, x_dim, stride);
  }
}

} // perceptron
} // tensors

#endif //PERCEPTRON_TENSORS_TENSOR2DIMPL_HPP
