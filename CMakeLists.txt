cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

project(soft_npu LANGUAGES CXX)

include(FetchContent)

FetchContent_Declare(json
        GIT_REPOSITORY https://github.com/ArthurSonzogni/nlohmann_json_cmake_fetchcontent
        GIT_TAG v3.7.3)

FetchContent_GetProperties(json)
if(NOT json_POPULATED)
    FetchContent_Populate(json)
    add_subdirectory(${json_SOURCE_DIR} ${json_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

find_package(TBB REQUIRED tbb)

find_package(libcmaes)

if (NOT libcmaes_FOUND)
    FetchContent_Declare(
            libcmaes
            GIT_REPOSITORY https://github.com/beniz/libcmaes.git
            GIT_TAG master)
    FetchContent_GetProperties(libcmaes)
    if(NOT libcmaes_POPULATED)
        FetchContent_Populate(libcmaes)
        file (WRITE "${libcmaes_SOURCE_DIR}/include/libcmaes/cmaes_export.h" "#define CMAES_EXPORT")
        add_subdirectory(${libcmaes_SOURCE_DIR} ${libcmaes_BINARY_DIR})
    endif()
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Wpedantic ${OpenMP_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -Wextra -Wpedantic ${OpenMP_CXX_FLAGS}")

enable_testing()

add_subdirectory(src)
add_subdirectory(test)
