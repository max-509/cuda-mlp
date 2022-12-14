#ifndef PERCEPTRON_TENSORS_OPS_KERNELS_MATHOPSKERNELS_CUH
#define PERCEPTRON_TENSORS_OPS_KERNELS_MATHOPSKERNELS_CUH

#include "perceptron/common/Common.h"
#include "perceptron/tensors/Tensor1D.h"
#include "perceptron/tensors/Tensor2D.h"

#include <cuda_runtime.h>

namespace perceptron {
namespace tensors {
namespace ops {
namespace kernels {

void
scal_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
            float alpha, TensorWriteable2D<float> x);

void
scal_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
            double alpha, TensorWriteable2D<double> x);

void
reverse_scal_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
                    float alpha, TensorWriteable2D<float> x);

void
reverse_scal_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
                    double alpha, TensorWriteable2D<double> x);

void
add_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
           float alpha, TensorWriteable2D<float> x);

void
add_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
           double alpha, TensorWriteable2D<double> x);

void
add_negative_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
                    float alpha, TensorWriteable2D<float> x);

void
add_negative_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
                    double alpha, TensorWriteable2D<double> x);

void
element_wise_mul_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
                        TensorReadOnly2D<float, false> t1,
                        TensorReadOnly2D<float, false> t2,
                        TensorWriteable2D<float> dst);

void
element_wise_mul_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
                        TensorReadOnly2D<float, false> t1,
                        TensorReadOnly2D<float, true> t2,
                        TensorWriteable2D<float> dst);

void
element_wise_mul_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
                        TensorReadOnly2D<float, true> t1,
                        TensorReadOnly2D<float, false> t2,
                        TensorWriteable2D<float> dst);

void
element_wise_mul_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
                        TensorReadOnly2D<float, true> t1,
                        TensorReadOnly2D<float, true> t2,
                        TensorWriteable2D<float> dst);

void
element_wise_mul_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
                        TensorReadOnly2D<double, false> t1,
                        TensorReadOnly2D<double, false> t2,
                        TensorWriteable2D<double> dst);

void
element_wise_mul_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
                        TensorReadOnly2D<double, false> t1,
                        TensorReadOnly2D<double, true> t2,
                        TensorWriteable2D<double> dst);

void
element_wise_mul_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
                        TensorReadOnly2D<double, true> t1,
                        TensorReadOnly2D<double, false> t2,
                        TensorWriteable2D<double> dst);

void
element_wise_mul_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
                        TensorReadOnly2D<double, true> t1,
                        TensorReadOnly2D<double, true> t2,
                        TensorWriteable2D<double> dst);

void
element_wise_mul_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
                        TensorReadOnly2D<float, false> t1,
                        TensorWriteable2D<float> dst);

void
element_wise_mul_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
                        TensorReadOnly2D<float, true> t1,
                        TensorWriteable2D<float> dst);

void
element_wise_mul_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
                        TensorReadOnly2D<double, false> t1,
                        TensorWriteable2D<double> dst);

void
element_wise_mul_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
                        TensorReadOnly2D<double, true> t1,
                        TensorWriteable2D<double> dst);

void
exp_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
           TensorReadOnly2D<float, false> src, TensorWriteable2D<float> dst);

void
exp_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
           TensorReadOnly2D<float, true> src, TensorWriteable2D<float> dst);

void
exp_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
           TensorReadOnly2D<double, false> src, TensorWriteable2D<double> dst);

void
exp_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
           TensorReadOnly2D<double, true> src, TensorWriteable2D<double> dst);

void
exp_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
           TensorWriteable2D<float> dst);

void
exp_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
           TensorWriteable2D<double> dst);

void
negative_exp_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
                    TensorReadOnly2D<float, false> src, TensorWriteable2D<float> dst);

void
negative_exp_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
                    TensorReadOnly2D<float, true> src, TensorWriteable2D<float> dst);

void
negative_exp_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
                    TensorReadOnly2D<double, false> src, TensorWriteable2D<double> dst);

void
negative_exp_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
                    TensorReadOnly2D<double, true> src, TensorWriteable2D<double> dst);

void
negative_exp_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
                    TensorWriteable2D<float> dst);

void
negative_exp_kernel(dim3 threads, dim3 blocks, size_type shared_mem, cudaStream_t stream,
                    TensorWriteable2D<double> dst);

} // perceptron
} // tensors
} // ops
} // kernels

#endif //PERCEPTRON_TENSORS_OPS_KERNELS_MATHOPSKERNELS_CUH
