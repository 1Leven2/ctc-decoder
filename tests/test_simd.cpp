/**
 * SIMD 工具函数单元测试
 *
 * 验证 AVX2 实现的 simd_argmax 和 simd_threshold_filter
 * 与标量版本结果完全一致。
 */

#include <gtest/gtest.h>

#include <cstring>
#include <random>
#include <vector>

#include "ctc/simd_utils.h"

using namespace ctc::simd;

/* ═══════════════════════════════════════════════════════════════
 *  辅助：标量参考实现（用于交叉验证）
 * ═══════════════════════════════════════════════════════════════ */

namespace {

/** 标量 argmax */
std::pair<float, int> ScalarArgmax(const float *data, int n) {
  if (n <= 0)
    return {-std::numeric_limits<float>::infinity(), -1};
  float max_val = data[0];
  int max_idx = 0;
  for (int i = 1; i < n; ++i) {
    if (data[i] > max_val) {
      max_val = data[i];
      max_idx = i;
    }
  }
  return {max_val, max_idx};
}

/** 标量阈值过滤 */
std::vector<int> ScalarThresholdFilter(const float *data, int n,
                                        float threshold) {
  std::vector<int> result;
  for (int i = 0; i < n; ++i) {
    if (data[i] >= threshold) {
      result.push_back(i);
    }
  }
  return result;
}

/** 生成随机 float 数组 */
std::vector<float> RandomFloats(int n, unsigned seed = 42) {
  std::vector<float> data(n);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
  for (int i = 0; i < n; ++i) {
    data[i] = dist(rng);
  }
  return data;
}

} // namespace

/* ═══════════════════════════════════════════════════════════════
 *  simd_argmax 测试
 * ═══════════════════════════════════════════════════════════════ */

TEST(SimdArgmax, EmptyArray) {
  auto [val, idx] = simd_argmax(nullptr, 0);
  EXPECT_TRUE(std::isinf(val) && val < 0);
  EXPECT_EQ(idx, -1);
}

TEST(SimdArgmax, SingleElement) {
  float data[] = {3.14f};
  auto [val, idx] = simd_argmax(data, 1);
  EXPECT_FLOAT_EQ(val, 3.14f);
  EXPECT_EQ(idx, 0);
}

TEST(SimdArgmax, AllNegative) {
  float data[] = {-5.0f, -1.0f, -3.0f};
  auto [val, idx] = simd_argmax(data, 3);
  EXPECT_FLOAT_EQ(val, -1.0f);
  EXPECT_EQ(idx, 1);
}

TEST(SimdArgmax, AllSameValue) {
  float data[] = {2.0f, 2.0f, 2.0f, 2.0f};
  auto [val, idx] = simd_argmax(data, 4);
  EXPECT_FLOAT_EQ(val, 2.0f);
  // 全部相同时返回第一个最大值
  EXPECT_EQ(idx, 0);
}

TEST(SimdArgmax, MatchScalarSmall) {
  // 小于 8 个（非对齐 AVX 步长的零头）
  for (int n = 1; n <= 20; ++n) {
    auto data = RandomFloats(n, n);
    auto [simd_val, simd_idx] = simd_argmax(data.data(), n);
    auto [scalar_val, scalar_idx] = ScalarArgmax(data.data(), n);
    EXPECT_FLOAT_EQ(simd_val, scalar_val) << "n=" << n;
    EXPECT_EQ(simd_idx, scalar_idx) << "n=" << n;
  }
}

TEST(SimdArgmax, MatchScalarLarge) {
  // 远大于 AVX 步长（8），覆盖多个向量块
  for (int n : {100, 500, 1000, 5000}) {
    auto data = RandomFloats(n, n);
    auto [simd_val, simd_idx] = simd_argmax(data.data(), n);
    auto [scalar_val, scalar_idx] = ScalarArgmax(data.data(), n);
    EXPECT_FLOAT_EQ(simd_val, scalar_val) << "n=" << n;
    EXPECT_EQ(simd_idx, scalar_idx) << "n=" << n;
  }
}

TEST(SimdArgmax, MaxAtBoundary) {
  // 最大值在 AVX 步长边界附近，覆盖索引追踪的正确性
  std::vector<int> test_sizes = {7, 8, 9, 15, 16, 17, 31, 32, 33};
  for (int n : test_sizes) {
    std::vector<float> data(n, -100.0f);
    data[n - 1] = 100.0f; // 最大值在末尾
    auto [val, idx] = simd_argmax(data.data(), n);
    EXPECT_FLOAT_EQ(val, 100.0f) << "n=" << n;
    EXPECT_EQ(idx, n - 1) << "n=" << n;
  }
}

/* ═══════════════════════════════════════════════════════════════
 *  simd_threshold_filter 测试
 * ═══════════════════════════════════════════════════════════════ */

TEST(SimdThresholdFilter, EmptyArray) {
  auto result = simd_threshold_filter(nullptr, 0, 0.0f);
  EXPECT_TRUE(result.empty());
}

TEST(SimdThresholdFilter, AllPass) {
  float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  auto result = simd_threshold_filter(data, 8, 0.0f);
  EXPECT_EQ(result.size(), 8u);
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(result[i], i);
  }
}

TEST(SimdThresholdFilter, NonePass) {
  float data[] = {0.1f, 0.2f, 0.3f};
  auto result = simd_threshold_filter(data, 3, 100.0f);
  EXPECT_TRUE(result.empty());
}

TEST(SimdThresholdFilter, ExactThreshold) {
  float data[] = {0.5f, 1.0f, -1.0f};
  auto result = simd_threshold_filter(data, 3, 0.5f);
  ASSERT_EQ(result.size(), 2u);
  EXPECT_EQ(result[0], 0);
  EXPECT_EQ(result[1], 1);
}

TEST(SimdThresholdFilter, MatchScalar) {
  for (int n : {1, 5, 10, 50, 100, 500}) {
    auto data = RandomFloats(n, n + 100);
    auto simd_result = simd_threshold_filter(data.data(), n, 0.0f);
    auto scalar_result = ScalarThresholdFilter(data.data(), n, 0.0f);
    EXPECT_EQ(simd_result, scalar_result) << "n=" << n;
  }
}

TEST(SimdThresholdFilter, NegativeThreshold) {
  float data[] = {-5.0f, -3.0f, -1.0f, 1.0f, 3.0f};
  auto result = simd_threshold_filter(data, 5, -2.0f);
  ASSERT_EQ(result.size(), 3u);
  EXPECT_EQ(result[0], 2); // -1.0
  EXPECT_EQ(result[1], 3); // 1.0
  EXPECT_EQ(result[2], 4); // 3.0
}
