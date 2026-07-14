# ─── 依赖：GoogleTest ───────────────────────────────────
find_package(GTest QUIET)
if(NOT GTest_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

include(GoogleTest)

# ─── 贪心解码单元测试 ──────────────────────────────────
add_executable(test_greedy test_greedy.cpp)
target_link_libraries(test_greedy PRIVATE ctc_core GTest::gtest_main)
gtest_discover_tests(test_greedy)

# ─── 前缀束搜索单元测试 ────────────────────────────────
add_executable(test_beam test_beam.cpp)
target_link_libraries(test_beam PRIVATE ctc_core GTest::gtest_main)
gtest_discover_tests(test_beam)
