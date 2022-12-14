set(HDRS_PATH ${mlp_SOURCE_DIR}/include)
set(SRCS_PATH ${mlp_SOURCE_DIR}/src)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

include(CheckLanguage)
check_language(CUDA)

set(HDRS

        ${HDRS_PATH}/perceptron/MultilayerPerceptron.h

        ${HDRS_PATH}/perceptron/losses/ILoss.h
        ${HDRS_PATH}/perceptron/losses/SquaredLoss.h

        ${HDRS_PATH}/perceptron/optimizers/IOptimizer.h
        ${HDRS_PATH}/perceptron/optimizers/SGD.h

        ${HDRS_PATH}/perceptron/activations/IActivation.h
        ${HDRS_PATH}/perceptron/activations/IdentityActivation.h
        ${HDRS_PATH}/perceptron/activations/SigmoidActivation.h

        ${HDRS_PATH}/perceptron/tensors/ops/MemoryOps.h
        ${HDRS_PATH}/perceptron/tensors/ops/kernels/MemoryOpsKernels.cuh
        ${HDRS_PATH}/perceptron/tensors/ops/kernels/impl/MathOpsKernelsImpl.cuh
        ${HDRS_PATH}/perceptron/tensors/ops/MathOps.h
        ${HDRS_PATH}/perceptron/tensors/ops/kernels/MathOpsKernels.cuh
        ${HDRS_PATH}/perceptron/tensors/ops/kernels/impl/MemoryOpsKernelsImpl.cuh
        ${HDRS_PATH}/perceptron/tensors/TensorHelper.h
        ${HDRS_PATH}/perceptron/tensors/Tensor1D.h
        ${HDRS_PATH}/perceptron/tensors/Tensor2D.h

        ${HDRS_PATH}/perceptron/common/Common.h
        ${HDRS_PATH}/perceptron/common/utils/MemoryUtils.h
        ${HDRS_PATH}/perceptron/common/utils/StreamUtils.h
        ${HDRS_PATH}/perceptron/common/utils/CuBLASUtils.h
        ${HDRS_PATH}/perceptron/common/utils/CuDNNUtils.h
        ${HDRS_PATH}/perceptron/common/utils/CudaUtils.h
        ${HDRS_PATH}/perceptron/common/functions/MathFunctions.cuh)

set(SRCS

        ${SRCS_PATH}/perceptron/MultilayerPerceptron.cpp

        ${SRCS_PATH}/perceptron/losses/ILoss.cpp
        ${SRCS_PATH}/perceptron/losses/SquaredLoss.cpp

        ${SRCS_PATH}/perceptron/optimizers/IOptimizer.cpp
        ${SRCS_PATH}/perceptron/optimizers/SGD.cpp

        ${SRCS_PATH}/perceptron/activations/IActivation.cpp
        ${SRCS_PATH}/perceptron/activations/IdentityActivation.cpp
        ${SRCS_PATH}/perceptron/activations/SigmoidActivation.cpp

        ${SRCS_PATH}/perceptron/tensors/ops/kernels/MathOpsKernels.cu
        ${SRCS_PATH}/perceptron/tensors/ops/kernels/MemoryOpsKernels.cu

        ${SRCS_PATH}/perceptron/common/utils/StreamUtils.cpp
        ${SRCS_PATH}/perceptron/common/utils/CuBLASUtils.cpp
        ${SRCS_PATH}/perceptron/common/utils/CudaUtils.cpp
        ${SRCS_PATH}/perceptron/common/utils/CuDNNUtils.cpp
        ${SRCS_PATH}/perceptron/common/utils/MemoryUtils.cpp)

add_library(mlp SHARED ${SRCS} ${HDRS})

target_include_directories(mlp PUBLIC ${HDRS_PATH})

target_link_directories(mlp PRIVATE ${CUDAToolkit_LIBRARY_DIR})
target_link_libraries(mlp PRIVATE CUDA::cudart)
target_link_libraries(mlp PRIVATE CUDA::cuda_driver)
target_link_libraries(mlp PRIVATE CUDA::cublas)
target_link_libraries(mlp PRIVATE cudnn)

target_link_libraries(mlp INTERFACE CUDA::cudart)
target_link_libraries(mlp INTERFACE CUDA::cuda_driver)
target_link_libraries(mlp INTERFACE CUDA::cublas)

set_target_properties(mlp PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        LINKER_LANGUAGE CUDA)

source_group(
        TREE "${HDRS_PATH}"
        PREFIX "Header Files"
        FILES ${HDRS})


