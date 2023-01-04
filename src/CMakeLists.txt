set(HDRS_PATH ${mlp_SOURCE_DIR}/include)
set(SRCS_PATH ${mlp_SOURCE_DIR}/src)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ version selection")
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

include(CheckLanguage)
check_language(CUDA)

set(HDRS

        ${HDRS_PATH}/perceptron/MultilayerPerceptron.h
        ${HDRS_PATH}/perceptron/Layer.h

        ${HDRS_PATH}/perceptron/losses/ILoss.h
        ${HDRS_PATH}/perceptron/losses/SquaredLoss.h

        ${HDRS_PATH}/perceptron/optimizers/OptimizerFactory.h
        ${HDRS_PATH}/perceptron/optimizers/IOptimizer.h
        ${HDRS_PATH}/perceptron/optimizers/SGD.h

        ${HDRS_PATH}/perceptron/activations/IActivation.h
        ${HDRS_PATH}/perceptron/activations/IdentityActivation.h
        ${HDRS_PATH}/perceptron/activations/SigmoidActivation.h
        ${HDRS_PATH}/perceptron/activations/ReLUActivation.h
        ${HDRS_PATH}/perceptron/activations/LeakyReLUActivation.h

        ${HDRS_PATH}/perceptron/tensors/ops/MemoryOps.h
        ${HDRS_PATH}/perceptron/tensors/ops/kernels/MemoryOpsKernels.cuh
        ${HDRS_PATH}/perceptron/tensors/ops/kernels/impl/MathOpsKernelsImpl.cuh
        ${HDRS_PATH}/perceptron/tensors/ops/MathOps.h
        ${HDRS_PATH}/perceptron/tensors/ops/kernels/MathOpsKernels.cuh
        ${HDRS_PATH}/perceptron/tensors/ops/kernels/impl/MemoryOpsKernelsImpl.cuh
        ${HDRS_PATH}/perceptron/tensors/ops/ConstructOps.h
        ${HDRS_PATH}/perceptron/tensors/ops/kernels/ConstructOpsKernels.cuh
        ${HDRS_PATH}/perceptron/tensors/ops/kernels/impl/ConstructOpsKernelsImpl.cuh
        ${HDRS_PATH}/perceptron/tensors/ops/Ops.h
        ${HDRS_PATH}/perceptron/tensors/shufflers/IShuffler.h
        ${HDRS_PATH}/perceptron/tensors/shufflers/DummyShuffler.h
        ${HDRS_PATH}/perceptron/tensors/shufflers/BatchShuffler.h
        ${HDRS_PATH}/perceptron/tensors/TensorHelper.h
        ${HDRS_PATH}/perceptron/tensors/Tensor1D.h
        ${HDRS_PATH}/perceptron/tensors/Tensor1DImpl.hpp
        ${HDRS_PATH}/perceptron/tensors/Tensor2D.h
        ${HDRS_PATH}/perceptron/tensors/Tensor2DImpl.hpp
        ${HDRS_PATH}/perceptron/tensors/TensorGetter.h

        ${HDRS_PATH}/perceptron/common/Common.h
        ${HDRS_PATH}/perceptron/common/utils/MemoryUtils.h
        ${HDRS_PATH}/perceptron/common/utils/StreamUtils.h
        ${HDRS_PATH}/perceptron/common/utils/CuBLASUtils.h
        ${HDRS_PATH}/perceptron/common/utils/CurandUtils.h
        ${HDRS_PATH}/perceptron/common/utils/CudaUtils.h
        ${HDRS_PATH}/perceptron/common/functions/MathFunctions.cuh)

set(SRCS

        ${SRCS_PATH}/perceptron/MultilayerPerceptron.cu
        ${SRCS_PATH}/perceptron/Layer.cu

        ${SRCS_PATH}/perceptron/losses/ILoss.cu
        ${SRCS_PATH}/perceptron/losses/SquaredLoss.cu

        ${SRCS_PATH}/perceptron/optimizers/OptimizerFactory.cu
        ${SRCS_PATH}/perceptron/optimizers/IOptimizer.cu
        ${SRCS_PATH}/perceptron/optimizers/SGD.cu

        ${SRCS_PATH}/perceptron/activations/IActivation.cu
        ${SRCS_PATH}/perceptron/activations/IdentityActivation.cu
        ${SRCS_PATH}/perceptron/activations/SigmoidActivation.cu
        ${SRCS_PATH}/perceptron/activations/LeakyReLUActivation.cu
        ${SRCS_PATH}/perceptron/activations/ReLUActivation.cu

        ${SRCS_PATH}/perceptron/tensors/ops/kernels/ConstructOpsKernels.cu
        ${SRCS_PATH}/perceptron/tensors/ops/kernels/MathOpsKernels.cu
        ${SRCS_PATH}/perceptron/tensors/ops/kernels/MemoryOpsKernels.cu
        ${SRCS_PATH}/perceptron/tensors/shufflers/IShuffler.cu
        ${SRCS_PATH}/perceptron/tensors/shufflers/DummyShuffler.cu
        ${SRCS_PATH}/perceptron/tensors/shufflers/BatchShuffler.cu

        ${SRCS_PATH}/perceptron/common/utils/StreamUtils.cu
        ${SRCS_PATH}/perceptron/common/utils/CuBLASUtils.cu
        ${SRCS_PATH}/perceptron/common/utils/CudaUtils.cu
        ${SRCS_PATH}/perceptron/common/utils/CurandUtils.cu
        ${SRCS_PATH}/perceptron/common/utils/MemoryUtils.cu)

add_library(mlp STATIC ${SRCS} ${HDRS})

target_compile_options(mlp PRIVATE --extended-lambda)

target_include_directories(mlp PUBLIC ${HDRS_PATH})

target_link_directories(mlp PUBLIC ${CUDAToolkit_LIBRARY_DIR})
target_link_libraries(mlp PUBLIC CUDA::cudart)
target_link_libraries(mlp PUBLIC CUDA::cuda_driver)
target_link_libraries(mlp PUBLIC CUDA::cublas)
target_link_libraries(mlp PUBLIC CUDA::curand)

set_target_properties(mlp PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CUDA_VISIBILITY_PRESET "hidden"
        CUDA_SEPARABLE_COMPILATION ON
        LINKER_LANGUAGE CUDA)

source_group(
        TREE "${HDRS_PATH}"
        PREFIX "Header Files"
        FILES ${HDRS})

# Pybind 11
set(PYTHON_MLP_TARGET cuda_mlp)
pybind11_add_module(${PYTHON_MLP_TARGET} ${SRCS_PATH}/module.cu)

target_link_libraries(${PYTHON_MLP_TARGET} PRIVATE mlp)
target_link_libraries(${PYTHON_MLP_TARGET} PRIVATE pybind11::pybind11)

set_target_properties(${PYTHON_MLP_TARGET} PROPERTIES
        CXX_VISIBILITY_PRESET "hidden"
        INTERPROCEDURAL_OPTIMIZATION TRUE
        PREFIX "${PYTHON_MODULE_PREFIX}"
        SUFFIX "${PYTHON_MODULE_EXTENSION}")


