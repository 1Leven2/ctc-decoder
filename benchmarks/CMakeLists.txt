# ─── 依赖：Google Benchmark ──────────────────────────────
find_package(benchmark QUIET)
if(NOT benchmark_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.8.3
    )
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googlebenchmark)
endif()

add_executable(ctc_bench bench.cpp)
target_link_libraries(ctc_bench PRIVATE ctc_core benchmark::benchmark)
target_compile_options(ctc_bench PRIVATE $<$<CONFIG:Release>:-O3>)
