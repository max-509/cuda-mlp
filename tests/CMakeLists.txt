cmake_minimum_required(VERSION 3.15)
project(mlp_tests LANGUAGES CXX CUDA)

set(TESTS_PATH ${mlp_SOURCE_DIR}/tests)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ version selection")
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
        catch2lib
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v2.12.0
)
FetchContent_MakeAvailable(catch2lib)

set(TARGET mlp_tests)

macro(MlpCreateTest TestName)
    set(TEST_TARGET "${TestName}_${TARGET}")
    add_executable(${TEST_TARGET}
            ${TESTS_PATH}/perceptron/${TestName}_tests.cu)
    target_link_libraries(${TEST_TARGET} PRIVATE mlp Catch2::Catch2)
    target_compile_options(${TEST_TARGET} PRIVATE --extended-lambda)

    set_target_properties(${TEST_TARGET} PROPERTIES
            CXX_STANDARD 17
            CXX_EXTENSIONS OFF
            CMAKE_CXX_STANDARD_REQUIRED ON
            POSITION_INDEPENDENT_CODE ON
            LINKER_LANGUAGE CUDA
            OUTPUT_NAME "${TEST_TARGET}")
endmacro()

MlpCreateTest(MultilayerPerceptron)
MlpCreateTest(MathOps)
MlpCreateTest(MemoryOps)
