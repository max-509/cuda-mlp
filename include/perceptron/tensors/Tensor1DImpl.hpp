#ifndef PERCEPTRON_TENSORS_TENSOR1DIMPL_HPP
#define PERCEPTRON_TENSORS_TENSOR1DIMPL_HPP

#include "perceptron/tensors/Tensor2D.h"
#include "perceptron/tensors/TensorGetter.h"

namespace perceptron {
namespace tensors {

template<typename T>
DEVICE_CALLABLE
TensorReadOnly1D<T>::TensorReadOnly1D(const value_type *p_array, size_type size, size_type stride)
    : m_p_array(p_array), m_size(size), m_stride(stride) {}

template<typename T>
DEVICE_CALLABLE
TensorReadOnly1D<T>::TensorReadOnly1D(const value_type *p_array, size_type size) : TensorReadOnly1D(p_array, size, 1) {}

template<typename T>
DEVICE_CALLABLE
inline typename TensorReadOnly1D<T>::creference_type
TensorReadOnly1D<T>::operator[](size_type x) const noexcept {
  return *get_elem(m_p_array, x, m_stride);
}

template<typename T>
DEVICE_CALLABLE
inline typename TensorReadOnly1D<T>::creference_type
TensorReadOnly1D<T>::operator()(size_type x) const noexcept {
  return *get_elem(m_p_array, x, m_stride);
}

template<typename T>
DEVICE_CALLABLE
inline typename TensorReadOnly1D<T>::creference_type
TensorReadOnly1D<T>::operator()() const noexcept {
  return *m_p_array;
}

template<typename T>
DEVICE_CALLABLE
inline const typename TensorReadOnly1D<T>::value_type *
TensorReadOnly1D<T>::get(size_type x) const noexcept {
  return get_elem(m_p_array, x, m_stride);
}

template<typename T>
DEVICE_CALLABLE
inline const typename TensorReadOnly1D<T>::value_type *
TensorReadOnly1D<T>::get() const noexcept {
  return m_p_array;
}

template<typename T>
DEVICE_CALLABLE
inline size_type
TensorReadOnly1D<T>::get_size() const noexcept { return m_size; }

template<typename T>
DEVICE_CALLABLE
inline size_type
TensorReadOnly1D<T>::get_stride() const noexcept { return m_stride; }

template<typename T>
DEVICE_CALLABLE
inline TensorReadOnly2D<T, false>
TensorReadOnly1D<T>::to_2d() const noexcept {
  return TensorReadOnly2D<T, false>(m_p_array, m_size, 1, m_stride);
}

template<typename T>
DEVICE_CALLABLE
inline TensorReadOnly2D<T, true>
TensorReadOnly1D<T>::to_2d_t() const noexcept {
  return TensorReadOnly2D<T, true>(m_p_array, m_size, 1, m_stride);
}

template<typename T>
DEVICE_CALLABLE
TensorWriteable1D<T>::TensorWriteable1D(pointer_type p_array, size_type size, size_type stride)
    : m_p_array(p_array), m_size(size), m_stride(stride) {}

template<typename T>
DEVICE_CALLABLE
TensorWriteable1D<T>::TensorWriteable1D(pointer_type
                                        p_array,
                                        size_type size) : TensorWriteable1D(p_array, size, 1) {}

template<typename T>
DEVICE_CALLABLE
inline typename TensorWriteable1D<T>::vref_type
TensorWriteable1D<T>::operator[](size_type x) noexcept {
  return m_p_array[x * m_stride];
}

template<typename T>
DEVICE_CALLABLE
inline typename TensorWriteable1D<T>::creference_type
TensorWriteable1D<T>::operator[](size_type x) const noexcept {
  return m_p_array[x * m_stride];
}

template<typename T>
DEVICE_CALLABLE
inline typename TensorWriteable1D<T>::vref_type
TensorWriteable1D<T>::operator()(size_type x) noexcept {
  return m_p_array[x * m_stride];
}

template<typename T>
DEVICE_CALLABLE
inline typename TensorWriteable1D<T>::vref_type
TensorWriteable1D<T>::operator()() noexcept {
  return *m_p_array;
}

template<typename T>
DEVICE_CALLABLE
inline typename TensorWriteable1D<T>::creference_type
TensorWriteable1D<T>::operator()(size_type x) const noexcept {
  return m_p_array[x * m_stride];
}

template<typename T>
DEVICE_CALLABLE
inline typename TensorWriteable1D<T>::creference_type
TensorWriteable1D<T>::operator()() const noexcept {
  return *m_p_array;
}

template<typename T>
DEVICE_CALLABLE
inline const typename TensorWriteable1D<T>::value_type *
TensorWriteable1D<T>::get(size_type x) const noexcept {
  return m_p_array + x * m_stride;
}

template<typename T>
DEVICE_CALLABLE
inline const typename TensorWriteable1D<T>::value_type *
TensorWriteable1D<T>::get() const noexcept {
  return m_p_array;
}

template<typename T>
DEVICE_CALLABLE
inline typename TensorWriteable1D<T>::pointer_type
TensorWriteable1D<T>::get(size_type x) noexcept {
  return m_p_array + x * m_stride;
}

template<typename T>
DEVICE_CALLABLE
inline typename TensorWriteable1D<T>::pointer_type
TensorWriteable1D<T>::get() noexcept {
  return m_p_array;
}

template<typename T>
DEVICE_CALLABLE
inline size_type
TensorWriteable1D<T>::get_size() const noexcept { return m_size; }

template<typename T>
DEVICE_CALLABLE
inline size_type
TensorWriteable1D<T>::get_stride() const noexcept { return m_stride; }

template<typename T>
DEVICE_CALLABLE
inline auto
TensorWriteable1D<T>::to_read_only() const noexcept {
  return TensorReadOnly1D<T>(m_p_array, m_size, m_stride);
}

template<typename T>
DEVICE_CALLABLE
inline TensorWriteable2D<T>
TensorWriteable1D<T>::to_2d() const noexcept {
  return TensorWriteable2D<T>(m_p_array, m_size, 1, m_stride);
}

template<typename T>
DEVICE_CALLABLE
inline
TensorWriteable1D<T>::operator TensorReadOnly1D<T>() const {
  return to_read_only();
}

template<typename T>
TensorOwner1D<T>::TensorOwner1D()
    : m_owned_ptr(owned_ptr_type{nullptr}), m_tensor_view(m_owned_ptr.get(), 0, 0) {}

template<typename T>
TensorOwner1D<T>::TensorOwner1D(owned_ptr_type &&owned_ptr, size_type size, size_type stride)
    : m_owned_ptr(std::move(owned_ptr)), m_tensor_view(m_owned_ptr.get(), size, stride) {}

template<typename T>
TensorOwner1D<T>::TensorOwner1D(owned_ptr_type &&owned_ptr, size_type size)
    : TensorOwner1D(std::move(owned_ptr), size, 1) {}

template<typename T>
typename TensorOwner1D<T>::owned_ptr_type
TensorOwner1D<T>::release() noexcept {
  return std::move(m_owned_ptr);
}

template<typename T>
typename TensorOwner1D<T>::view_type
TensorOwner1D<T>::tensor_view() const noexcept {
  return m_tensor_view;
}

template<typename T>
void
TensorOwner1D<T>::to_host(cudaStream_t stream) {
  auto attrs = utils::cu_get_pointer_attrs(m_owned_ptr.get());
  if (utils::is_device(attrs)) {
    auto host_owner = utils::cu_make_host_memory_unique<T>(m_tensor_view.get_size() * m_tensor_view.get_stride(), stream);
    utils::cu_memcpy_async(host_owner.get(), m_owned_ptr.get(),
                           m_tensor_view.get_size() * m_tensor_view.get_stride(),
                           cudaMemcpyDefault, stream);
    (void) std::unique_ptr<T, utils::cu_memory_deleter_t>{m_owned_ptr.release(), utils::cu_memory_deleter_t{stream}};
    m_owned_ptr = std::move(host_owner);
    m_tensor_view = constructTensorWriteable1D(m_owned_ptr.get(), m_tensor_view.get_size(), m_tensor_view.get_stride());
  }
}

template<typename T>
void
TensorOwner1D<T>::to_device(cudaStream_t stream) {
  // TODO: finish
  auto attrs = utils::cu_get_pointer_attrs(m_owned_ptr.get());
  if (utils::is_device(attrs)) {
    auto host_owner = utils::cu_make_host_memory_unique<T>(stream);
    utils::cu_memcpy_async(host_owner.get(), m_owned_ptr.get(), m_tensor_view.get_size(),
                           cudaMemcpyDefault, stream);
    (void) std::unique_ptr<T, utils::cu_memory_deleter_t>{m_owned_ptr.release(), utils::cu_memory_deleter_t{stream}};
    m_owned_ptr = std::move(host_owner);
    m_tensor_view = constructTensorWriteable1D(m_owned_ptr.get(), m_tensor_view.get_size(), m_tensor_view.get_stride());
  }
}

template<typename T>
void
TensorOwner1D<T>::to_pinned(cudaStream_t stream) {
  // TODO: finish
}

template<typename T>
auto
constructTensorReadOnly1D(const T *p_tensor, size_type size, size_type stride) {
  return TensorReadOnly1D<T>{p_tensor, size, stride};
}

template<typename T>
auto
constructTensorWriteable1D(T *p_tensor, size_type size, size_type stride) {
  return TensorWriteable1D<T>{p_tensor, size, stride};
}

template<typename T, typename Deleter>
auto
constructTensorOwner1D(std::unique_ptr<T, Deleter> &&owned_ptr, size_type size, size_type stride) {
  return TensorOwner1D<T>(std::move(owned_ptr), size, stride);
}

template<typename T>
auto
constructTensorOwnerHost1D(size_type size, size_type stride, cudaStream_t stream) {
  auto ptr = utils::cu_make_host_memory_unique<T>(size * stride, stream);
  return TensorOwner1D<T>(std::move(ptr), size, stride);
}

template<typename T>
auto
constructTensorOwnerHost1D(size_type size, cudaStream_t stream) {
  return constructTensorOwnerHost1D<T>(size, DEFAULT_1D_STRIDE, stream);
}

template<typename T>
auto
constructTensorOwnerDevice1D(size_type size, size_type stride, cudaStream_t stream) {
  auto ptr = utils::cu_make_memory_unique<T>(size * stride, stream);
  return TensorOwner1D<T>(std::move(ptr), size, stride);
}

template<typename T>
auto
constructTensorOwnerDevice1D(size_type size, cudaStream_t stream) {
  return constructTensorOwnerDevice1D<T>(size, DEFAULT_1D_STRIDE, stream);
}

} // perceptron
} // tensors

#endif //PERCEPTRON_TENSORS_TENSOR1DIMPL_HPP
