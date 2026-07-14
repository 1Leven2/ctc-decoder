#ifndef CTC_GREEDY_DECODER_H_
#define CTC_GREEDY_DECODER_H_

#include <vector>

#include "decoder.h"
#include "types.h"

namespace ctc {

/**
 * CTC 贪心解码器
 *
 * 原理：每一帧直接选取概率最大的 token（argmax），
 *       然后应用 CTC 的空白符与重复合并规则得到最终序列。
 *
 * ── 合并规则 ──
 * 1. 去除所有 blank token（默认 id=0）
 * 2. 相邻重复的 token 只保留一个
 *    例：序列 [a, a, <b>, a, a] → [a, a]（b 为 blank 被去掉，相邻 a 合并）
 *
 * ── 复杂度 ──
 * 时间复杂度 O(T·V)，空间复杂度 O(T_result)，T_result ≤ T。
 * 因为每帧只取 argmax 不做搜索，速度极快，适合作为 baseline。
 *
 * ── 使用示例 ──
 * @code
 *   GreedyDecoder decoder;
 *   auto results = decoder.Decode(log_probs, num_frames, vocab_size);
 *   // results[0].tokens 即为解码的 token ID 序列
 * @endcode
 */
class GreedyDecoder : public CtcDecoder {
 public:
  explicit GreedyDecoder(int blank_id = 0);

  /* ─── 非流式接口 ─────────────────────────────── */

  std::vector<DecodeResult> Decode(const float* log_probs, int num_frames,
                                    int vocab_size) override;

  /* ─── 流式接口 ───────────────────────────────── */

  void Step(const float* log_probs, int vocab_size) override;

  std::vector<DecodeResult> Results(int n) const override;

  void Reset() override;

 private:
  /** 空白符 token 索引 */
  int blank_id_;

  /* ─── 流式解码内部状态 ───────────────────────── */

  /** 上一帧选取的 token ID，用于判断相邻重复。
   *  初始化为 -1（无效 token，确保首帧不会与前一帧重复）。 */
  int prev_token_;

  /** 当前累积的解码结果 */
  std::vector<int> tokens_;

  /** 累积对数概率（每帧 argmax 概率之和） */
  float total_score_;

  /** 每个输出 token 的起始帧号 */
  std::vector<int> timestamps_;

  /** 当前已处理的帧数（流式 Step 调用次数），用于记录时间戳 */
  int frame_count_;
};

}  // namespace ctc

#endif  // CTC_GREEDY_DECODER_H_
