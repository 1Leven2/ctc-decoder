#ifndef CTC_DECODER_H_
#define CTC_DECODER_H_

#include "types.h"

namespace ctc {

/**
 * CTC 解码器抽象接口
 *
 * 所有解码算法（贪心、前缀束搜索等）均实现此接口，
 * 支持两种调用模式：
 * 1) 非流式 — Decode() 一次性传入所有帧
 * 2) 流式   — Step() 逐帧输入，Results() 获取当前最优
 *
 * 设计原则：
 * - log_probs 为二维矩阵 [num_frames, vocab_size]，以行优先 (row-major) 展平存储
 * - 所有概率值均为 log 空间（log_softmax 的输出），调用方须在传入前完成归一化
 * - blank token 索引由 DecoderOptions::blank_id 指定，默认为 0
 */
class CtcDecoder {
 public:
  virtual ~CtcDecoder() = default;

  /* ─── 非流式接口 ─────────────────────────────────────────── */

  /**
   * 一次性输入全部音频帧的 log 概率，返回 n-best 解码结果。
   *
   * @param log_probs  log_softmax 后的概率矩阵，行优先展平。
   *                   shape: [num_frames, vocab_size]，类型 float
   * @param num_frames 音频帧数 T
   * @param vocab_size 词表大小 V
   * @return           按概率降序排列的 n 个最优解码结果
   */
  virtual std::vector<DecodeResult> Decode(const float* log_probs,
                                           int num_frames,
                                           int vocab_size) = 0;

  /* ─── 流式接口 ───────────────────────────────────────────── */

  /**
   * 逐帧输入 log 概率向量。
   * 流式场景下每来一帧调用一次，内部维护解码状态。
   *
   * @param log_probs  当前帧的 log 概率向量，长度 vocab_size
   * @param vocab_size 词表大小
   */
  virtual void Step(const float* log_probs, int vocab_size) = 0;

  /**
   * 获取当前解码状态下的 n 个最佳假设。
   * 流式场景下常在 Step() 若干帧后调用。
   */
  virtual std::vector<DecodeResult> Results(int n) const = 0;

  /**
   * 重置解码器内部状态。
   * 流式场景下结束一段音频后调用，以准备解码下一段。
   */
  virtual void Reset() = 0;
};

}  // namespace ctc

#endif  // CTC_DECODER_H_
