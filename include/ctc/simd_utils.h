/**
 * SIMD 加速工具函数（header-only）
 *
 * 对 CTC 解码器中计算密集的循环做 AVX2 手动向量化，
 * 编译器未定义 __AVX2__ 时自动退化为标量循环。
 *
 * 核心函数：
 *   simd_argmax            — 向量化 argmax，同时追踪值和索引
 *   simd_threshold_filter  — 向量化阈值过滤，批量筛选高于 cutoff 的 token
 */

#ifndef CTC_SIMD_UTILS_H_
#define CTC_SIMD_UTILS_H_

#include <limits>
#include <utility>
#include <vector>

#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace ctc {
namespace simd {

/* ═══════════════════════════════════════════════════════════════
 *  simd_argmax: 向量化 argmax
 *
 *  返回 (max_value, max_index)
 *
 *  AVX2 策略：
 *   每次加载 8 个 float，用 _mm256_max_ps 做向量比较，
 *   同时用 _mm256_cmp_ps 追踪最大值位置。
 *   最后对 8 通道做横向归约，找出全局最大值和索引。
 *
 *  标量策略 (fallback)：
 *   退化为 std::max_element 等价循环
 *  普通算法只有一个人找最大值，而 AVX 是 8 个人同时比赛找最大值
 * ═══════════════════════════════════════════════════════════════ */
#ifdef __AVX2__
inline std::pair<float, int> simd_argmax(const float *data, int n) {
  if (n <= 0)
    return {-std::numeric_limits<float>::infinity(), -1};

  int i = 0;

  // 处理非对齐头部（字节对齐到 32）
  // LoadU 已处理非对齐访问，这里只处理不足 8 个的余数
  float max_val = -std::numeric_limits<float>::infinity();
  int max_idx = -1;

  // 每次处理 8 个 float（256-bit / 32-bit = 8）
  constexpr int kStep = 8;

  // 构造索引寄存器：[0, 1, 2, 3, 4, 5, 6, 7]
  __m256i indices = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
  __m256i index_step = _mm256_set1_epi32(kStep); // [8 8 8 8 8 8 8 8]

  __m256 max_reg = _mm256_set1_ps(
      -std::numeric_limits<float>::infinity()); // [-inf, -inf, ..., -inf] 8
                                                // channels
  __m256i max_idx_reg = indices; // 当前 8 通道各自的最大值索引

  for (; i + kStep <= n; i += kStep) {
    __m256 vals =
        _mm256_loadu_ps(data + i); // 从内存加载 8 个 float 到寄存器 vals

    // 比较掩码：vals > max_reg ?
    __m256 cmp = _mm256_cmp_ps(
        vals, max_reg, _CMP_GT_OQ); // 掩码 FFFFFFFF 表示 vals >
                                    // max_reg，00000000 表示 vals <= max_reg

    // max_reg = max(vals, max_reg)
    max_reg = _mm256_max_ps(
        vals, max_reg); // 更新 max_reg 为当前 8 个通道各个通道的最大值

    // 对于 vals > old_max 的通道，更新索引为当前位置
    // _mm256_blendv_ps 根据掩码 cmp 选择 max_idx_reg 或 indices
    // 若掩码为 1（vals > max_reg），则选择 indices（后者），否则选择
    // max_idx_reg
    max_idx_reg = _mm256_castps_si256(_mm256_blendv_ps(
        _mm256_castsi256_ps(max_idx_reg), _mm256_castsi256_ps(indices), cmp));

    // indices += 8
    indices = _mm256_add_epi32(
        indices, index_step); // 对 8 个通道的索引加 8，准备下一轮迭代
  }

  // 横向归约：将 8 通道的 max_reg 和 max_idx_reg 合并为标量
  float max_arr[8];
  int idx_arr[8];
  _mm256_storeu_ps(
      max_arr, max_reg); // 将 max_reg 寄存器的 8 个 float 存储到 max_arr 数组
  _mm256_storeu_si256(
      (__m256i *)idx_arr,
      max_idx_reg); // 将 max_idx_reg 寄存器的 8 个 int 存储到 idx_arr 数组

  // 找出 8 个通道中的最大值和索引
  for (int j = 0; j < kStep; ++j) {
    if (max_arr[j] > max_val) {
      max_val = max_arr[j];
      max_idx = idx_arr[j];
    }
  }

  // 处理剩余不足 8 个的元素
  for (; i < n; ++i) {
    if (data[i] > max_val) {
      max_val = data[i];
      max_idx = i;
    }
  }

  return {max_val, max_idx};
}
#else
// 标量回退
inline std::pair<float, int> simd_argmax(const float *data, int n) {
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
#endif // __AVX2__

/* ═══════════════════════════════════════════════════════════════
 *  simd_threshold_filter: 向量化阈值过滤
 *
 *  返回所有 log_probs[i] >= threshold 的索引列表。
 *  用于 beam search 中替代逐 token 的 if (logp < cutoff) continue。
 *
 *  AVX2 策略：
 *   每次加载 8 个 float，一次 _mm256_cmp_ps 比较 + _mm256_movemask_ps
 *   生成 8-bit 掩码，逐 bit 检查得到命中的索引。
 *   相比标量版本的 N 次分支，向量化版本的数据并行度更高，
 *   且结果可直接用于内层循环迭代，消除内层的 cutoff 分支。
 *
 *  标量策略 (fallback)：
 *   简单 for 循环 + if 判断
 * ═══════════════════════════════════════════════════════════════ */
#ifdef __AVX2__
inline std::vector<int> simd_threshold_filter(const float *log_probs, int n,
                                              float threshold) {
  std::vector<int> result;
  if (n <= 0)
    return result;

  result.reserve(n / 4); // 预估 25% 通过率，减少 reallocation

  int i = 0;
  constexpr int kStep = 8;
  __m256 thresh = _mm256_set1_ps(threshold);

  for (; i + kStep <= n; i += kStep) {
    __m256 vals = _mm256_loadu_ps(log_probs + i);

    // 8 个 float 同时比较：vals >= threshold ?
    __m256 cmp = _mm256_cmp_ps(vals, thresh, _CMP_GE_OQ);

    // movemask 将 8 个比较结果的高位压缩为 8-bit 掩码
    int mask = _mm256_movemask_ps(cmp); // 把8个比较结果压缩成一个8位整数

    // 逐 bit 检查掩码，命中则记录索引
    if (mask != 0) {
      // 开始判断这 8 个通道是否满足阈值条件
      for (int j = 0; j < kStep; ++j) {
        if (mask & (1 << j)) { // 第 j 个元素满足阈值条件吗
          result.push_back(i + j);
        }
      }
    }
  } // 下一轮 for 从下 8 个元素开始处理

  // 处理剩余不足 8 个的元素
  for (; i < n; ++i) {
    if (log_probs[i] >= threshold) {
      result.push_back(i);
    }
  }

  return result;
}
#else
// 标量回退
inline std::vector<int> simd_threshold_filter(const float *log_probs, int n,
                                              float threshold) {
  std::vector<int> result;
  if (n <= 0)
    return result;
  result.reserve(n / 4);
  for (int i = 0; i < n; ++i) {
    if (log_probs[i] >= threshold) {
      result.push_back(i);
    }
  }
  return result;
}
#endif // __AVX2__

} // namespace simd
} // namespace ctc

#endif // CTC_SIMD_UTILS_H_
