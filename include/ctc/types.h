#ifndef CTC_TYPES_H_
#define CTC_TYPES_H_

#include <cstdint>
#include <vector>

namespace ctc {

/**
 * 解码器配置参数
 *
 * CTC 解码器的通用配置，贪心解码只需要 blank_id，
 * 前缀束搜索需要 beam_size 控制搜索宽度。
 */
struct DecoderOptions {
  /** 前缀束搜索的 beam 宽度，即每帧保留的假设数量上限。
   *  取值越大解码越准确，但计算量与 beam_size 成正比。
   *  工业界常用 10~100，设为 1 时退化为贪心解码。 */
  int beam_size = 10;

  /** CTC 空白符 (blank) 的 token 索引，默认为 0。
   *  此约定与 Wenet / DeepSpeech 一致，即词表中第 0 个位置为 blank。 */
  int blank_id = 0;

  /** 返回的最佳假设数量，默认为 1（只返回最优结果）。 */
  int n_best = 1;

  /** 剪枝概率阈值（log 空间）。每帧结束后，概率低于
   *  (当前最优概率 - cutoff_threshold) 的假设会被丢弃。
   *  设为 0.0 表示不启用阈值剪枝，仅按 beam_size 保留。 */
  float cutoff_threshold = 0.0;
};

/**
 * 解码结果
 *
 * 包含一次解码产出的 token 序列及其评分信息。
 * Python 侧会额外提供 text 字段（通过词表映射得到可读文本）。
 */
struct DecodeResult {
  /** 解码出的 token ID 序列（已去除 blank 并经过去重合并）。 */
  std::vector<int> tokens;

  /** 该序列在 CTC 下的对数概率（log P(序列 | 声学输入)）。
   *  值域 (-inf, 0]，越接近 0 表示置信度越高。 */
  float score = 0.0f;

  /** 每个输出 token 对应的音频帧号（用于时间对齐）。
   *  长度与 tokens 相同，timestamps[i] 表示 tokens[i] 被触发的帧索引。
   *  不需要时间信息时可留空。 */
  std::vector<int> timestamps;
};

} // namespace ctc

#endif // CTC_TYPES_H_
