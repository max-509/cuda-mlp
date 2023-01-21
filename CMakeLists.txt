cmake_minimum_required(VERSION 3.20)
project(mlp
        LANGUAGES CUDA CXX)

add_subdirectory(src)

if (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    set(CMAKE_CXX_EXTENSIONS OFF)
    set_property(GLOBAL PROPERTY USE_FOLDERS ON)
    add_subdirectory(tests)
    add_subdirectory(demo)
endif()
