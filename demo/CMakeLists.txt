cmake_minimum_required(VERSION 3.15)
project(mlp_demo LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ version selection")
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TARGET mlp_demo)

include(FetchContent)
FetchContent_Declare(
        csv-parser
        GIT_REPOSITORY https://github.com/vincentlaucsb/csv-parser.git
        GIT_TAG 2.1.3
)
FetchContent_MakeAvailable(csv-parser)

macro(MlpCreateDemo DemoName)
    set(DEMO_TARGET "${DemoName}_${TARGET}")
    add_executable(${DEMO_TARGET} ${DemoName}.cu results_saver.h)
    target_link_libraries(${DEMO_TARGET} PRIVATE mlp csv)
    target_compile_options(${DEMO_TARGET} PRIVATE --extended-lambda)
    target_include_directories(${DEMO_TARGET} PRIVATE ${CMAKE_BINARY_DIR}/_deps/csv-parser-src/include)

    set_target_properties(${DEMO_TARGET} PROPERTIES
            CXX_STANDARD 17
            CXX_EXTENSIONS OFF
            CMAKE_CXX_STANDARD_REQUIRED ON
            POSITION_INDEPENDENT_CODE ON
            LINKER_LANGUAGE CUDA
            OUTPUT_NAME "${DEMO_TARGET}")
endmacro()

MlpCreateDemo(cos_2pi)
MlpCreateDemo(x_sin_2pi)
MlpCreateDemo(polynom)
