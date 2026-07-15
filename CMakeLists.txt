cmake_minimum_required(VERSION 3.18)
project(ctc-decoder VERSION 0.1.0 LANGUAGES CXX)

# C++17 标准：平衡现代特性与编译器兼容性
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 编译优化
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ─── 依赖：pybind11 ──────────────────────────────────────
# 优先使用系统安装的 pybind11，否则通过 FetchContent 自动下载
find_package(pybind11 QUIET)
if(NOT pybind11_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# ─── 核心库：ctc_core（静态库）───────────────────────────
add_library(ctc_core STATIC
    src/greedy_decoder.cpp
    src/prefix_beam_search.cpp
    src/thread_pool.cpp
    src/batch_decoder.cpp
    src/ctc_api.cpp
)
target_include_directories(ctc_core PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)
target_compile_options(ctc_core PRIVATE
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Release>:-march=native>
)
# 开启警告（Google C++ Style Guide 要求）
target_compile_options(ctc_core PRIVATE
    -Wall -Wextra -Wpedantic -Wshadow
)

# ─── Python 绑定模块 ────────────────────────────────────
pybind11_add_module(_ctc_decoder bindings/pybind.cpp)
target_link_libraries(_ctc_decoder PRIVATE ctc_core)
target_compile_options(_ctc_decoder PRIVATE
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Release>:-march=native>
)

# ─── 选项：测试与基准（默认关闭，需要时用 -DBUILD_TESTS=ON） ──
option(BUILD_TESTS "Build unit tests" OFF)
option(BUILD_BENCHMARKS "Build performance benchmarks" OFF)
option(BUILD_SHARED_LIB "Build shared library (libctc_decoder.so) with C API" OFF)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# ─── 共享库：libctc_decoder.so（C API） ─────────────────
if(BUILD_SHARED_LIB)
    add_library(ctc_decoder SHARED src/ctc_shim.cpp)
    target_link_libraries(ctc_decoder PRIVATE
        -Wl,--whole-archive ctc_core -Wl,--no-whole-archive
    )
    target_include_directories(ctc_decoder PUBLIC ${CMAKE_SOURCE_DIR}/include)
    set_target_properties(ctc_decoder PROPERTIES
        VERSION ${PROJECT_VERSION}
        SOVERSION 1
        C_VISIBILITY_PRESET default
    )
endif()

# ─── 安装 ──────────────────────────────────────────────
install(TARGETS _ctc_decoder DESTINATION .)
if(BUILD_SHARED_LIB)
    install(TARGETS ctc_decoder DESTINATION lib)
    install(FILES include/ctc/ctc_api.h DESTINATION include/ctc)
endif()
