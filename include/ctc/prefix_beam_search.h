#ifndef CTC_PREFIX_BEAM_SEARCH_H_
#define CTC_PREFIX_BEAM_SEARCH_H_

#include <limits>
#include <unordered_map>
#include <vector>

#include "decoder.h"
#include "types.h"

namespace ctc {

/* ─── 内部类型 ─────────────────────────────────────────────── */

/**
 * CTC 前缀的分数结构
 *
 * CTC 的核心难题：同一条 label 序列对应无数条对齐路径（blank 可随意插入）。
 * 通过维护两个分数 "结尾于 blank" 和 "结尾于非 blank"，用动态规划合并所有路径，
 * 避免指数级枚举。
 *
 * 递推公式（每帧输入 logp 向量时）：
 *
 *   扩展到 blank（前缀不变）：
 *     P_b'(l) = log_add(P_b(l), P_nb(l)) + logp[blank]
 *
 *   扩展到非 blank 字符 c：
 *     - 若 c != l 的最后一个字符 或 l 为空：
 *         新前缀 l' = l + [c]
 *         P_nb'(l') = log_add(P_b(l), P_nb(l)) + logp[c]
 *     - 若 c == l 的最后一个字符：
 *         前缀不变（CTC 合并规则将连续相同 token 合并为一个）
 *         P_nb'(l) = P_nb(l) + logp[c]
 *         （仅从 "结尾于非 blank" 的路径扩展，
 *           从 P_b(l) 加 c 会开始新行程，与上述合并路径属不同对齐方式）
 *         注：理论上还应有 P_b(l[:-1]) + logp[c] 的贡献（从更短前缀开始新行程），
 *         但此项通常远小于 P_nb(l) + logp[c]，省略后偏差可忽略。
 */
struct PrefixScore {
  float prob_b = -std::numeric_limits<float>::infinity();
  float prob_nb = -std::numeric_limits<float>::infinity();

  /** 前缀的总 log 概率 = log(exp(prob_b) + exp(prob_nb)) */
  float Total() const;
};

/**
 * 前缀（token ID 序列）的哈希函数
 *
 * 使用 vector<int> 作为 unordered_map 的 key，
 * 组合每个元素的 hash 值得到最终 hash。
 */
struct PrefixHash {
  size_t operator()(const std::vector<int>& prefix) const;
};

/* ─── 解码器类 ────────────────────────────────────────────── */

/**
 * CTC 前缀束搜索解码器
 *
 * 与贪心解码不同，前缀束搜索在每一步维护 beam_size 个最优的文本前缀假设，
 * 可以有效纠正贪心解码中的局部决策错误，显著提高解码准确率。
 *
 * 算法流程（每帧）：
 * 1. 遍历当前 beam 中每个前缀，尝试扩展到 blank 和所有非 blank token。
 * 2. 用 PrefixScore 的递推公式计算每个扩展的 CTC 概率。
 * 3. 新产生的前缀合并到 hash map 中（相同前缀自动去重累加概率）。
 * 4. 按总概率排序，保留前 beam_size 个假设，其余剪枝丢弃。
 *
 * ── 复杂度 ──
 * 每帧 O(beam_size × V)，总 O(T × beam_size × V)。
 * 空间 O(beam_size × 最大序列长度)。
 *
 * ── 使用示例 ──
 * @code
 *   DecoderOptions opts;
 *   opts.beam_size = 10;
 *   PrefixBeamSearch decoder(opts);
 *   auto results = decoder.Decode(log_probs, num_frames, vocab_size);
 *   for (const auto& r : results) {
 *     // r.tokens 为解码 token 序列，r.score 为 log 概率
 *   }
 * @endcode
 */
class PrefixBeamSearch : public CtcDecoder {
 public:
  explicit PrefixBeamSearch(const DecoderOptions& opts);

  /* ─── 非流式接口 ─────────────────────────────── */

  std::vector<DecodeResult> Decode(const float* log_probs, int num_frames,
                                    int vocab_size) override;

  /* ─── 流式接口 ───────────────────────────────── */

  void Step(const float* log_probs, int vocab_size) override;

  std::vector<DecodeResult> Results(int n) const override;

  void Reset() override;

 private:
  /**
   * 对一帧 log 概率执行一次束扩展
   * @param log_probs 当前帧的 log 概率向量 [vocab_size]
   * @param vocab_size 词表大小
   * @param frame_idx 当前帧号（用于记录时间戳）
   */
  void AdvanceDecoding(const float* log_probs, int vocab_size, int frame_idx);

  /** 当前 beam 中的所有假设：前缀 → 分数 */
  using HypothesisMap =
      std::unordered_map<std::vector<int>, PrefixScore, PrefixHash>;
  HypothesisMap cur_hyps_;

  DecoderOptions opts_;
  int frame_count_;
};

}  // namespace ctc

#endif  // CTC_PREFIX_BEAM_SEARCH_H_
