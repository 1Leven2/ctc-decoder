/**
 * 批量 CTC 解码器实现
 *
 * DecodeBatch 核心流程：
 *   1. 预计算每条语音在扁平数据中的偏移量
 *   2. 预分配结果容器（不同索引间无竞争）
 *   3. 为每条语音向线程池提交独立解码任务
 *   4. 等待所有任务完成，收集结果
 *
 * 线程安全策略：每条语音独立创建 PrefixBeamSearch 实例，
 * 写入 results 的不同索引——全程无共享可变状态，无需显式加锁。
 */

#include "ctc/batch_decoder.h"

#include "ctc/prefix_beam_search.h"

namespace ctc {

BatchDecoder::BatchDecoder(const DecoderOptions &opts, size_t num_threads)
    : opts_(opts), pool_(num_threads) {}

std::vector<std::vector<DecodeResult>>
BatchDecoder::DecodeBatch(const float *batch_data, const int *frame_counts,
                          int num_utterances, int vocab_size) {
  // 边界检查
  if (num_utterances <= 0 || vocab_size <= 0)
    return {};

  // 步骤 1：预计算每条语音的起始偏移量
  // offset[i] 表示第 i 条语音首帧在 batch_data 中的 float 索引
  std::vector<int> offsets(num_utterances);
  int accum = 0;
  for (int i = 0; i < num_utterances; ++i) {
    offsets[i] = accum;
    accum += frame_counts[i] * vocab_size;
  }

  // 步骤 2：预分配结果容器
  // results[i] 由第 i 个任务独占写入，无需同步
  std::vector<std::vector<DecodeResult>> results(num_utterances);

  // 步骤 3：为每条语音提交解码任务
  std::vector<std::future<void>> futures;
  futures.reserve(num_utterances);

  for (int i = 0; i < num_utterances; ++i) {
    int frames = frame_counts[i];
    int offset = offsets[i];

    futures.push_back(pool_.Enqueue(
        [&results, i, batch_data, offset, frames, vocab_size, this]() {
          // 零帧语音 → 空结果
          if (frames <= 0) {
            results[i] = {};
            return;
          }

          // 每条语音独立创建解码器实例——无共享状态
          PrefixBeamSearch decoder(opts_);
          results[i] = decoder.Decode(batch_data + offset, frames, vocab_size);
          // decoder 在此析构，释放该语音的 Trie 树内存
        }));
  }

  // 步骤 4：等待所有任务完成
  // future.get() 若任务抛出异常，此处重新抛出
  for (auto &f : futures) {
    f.get();
  }

  return results;
}

} // namespace ctc
