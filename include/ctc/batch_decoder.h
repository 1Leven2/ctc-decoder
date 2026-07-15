/**
 * 批量 CTC 解码器
 *
 * 对 N 条语音同时执行前缀束搜索，每条语音独立解码实例，
 * 通过线程池并发执行，无共享状态，天然线程安全。
 *
 * 使用示例：
 *   DecoderOptions opts;
 *   opts.beam_size = 10;
 *   BatchDecoder decoder(opts, 4);  // 4 线程
 *
 *   // batch_data = [utt0_frames..., utt1_frames...] 平坦拼接
 *   // frame_counts = [T0, T1, ..., TN-1]
 *   auto results = decoder.DecodeBatch(data, frame_counts, N, V);
 *   // results[i][j] = 第 i 条语音的第 j 个最佳假设
 */

#ifndef CTC_BATCH_DECODER_H_
#define CTC_BATCH_DECODER_H_

#include <vector>

#include "ctc/thread_pool.h"
#include "ctc/types.h"

namespace ctc {

class BatchDecoder {
public:
  /**
   * 构造批量解码器
   *
   * @param opts        解码配置（beam_size、blank_id、n_best 等），
   *                    批量中每条语音共用同一配置
   * @param num_threads 线程池大小，0 表示自动取硬件并发数
   */
  explicit BatchDecoder(const DecoderOptions &opts, size_t num_threads = 0);

  /**
   * 批量解码
   *
   * @param batch_data     平坦拼接的 log_softmax 概率矩阵（行优先，按语音拼接）
   *                       布局：utt0[frame0_v0..V-1]...[frame_T0-1_v0..V-1]
   *                             utt1[frame0_v0..V-1]......
   * @param frame_counts   每条语音的帧数 [T0, T1, ..., T_{N-1}]
   * @param num_utterances 语音条数 N
   * @param vocab_size     词表大小 V（批量内所有语音共享相同词表）
   * @return               results[i] 为第 i 条语音的 n_best 个解码结果，
   *                       按概率降序排列。若 frame_counts[i] <= 0，
   *                       对应 results[i] 为空 vector。
   */
  std::vector<std::vector<DecodeResult>>
  DecodeBatch(const float *batch_data, const int *frame_counts,
              int num_utterances, int vocab_size);

  /** 返回内部线程池引用（用于观测负载等） */
  ThreadPool &GetThreadPool() { return pool_; }

  /** 返回当前解码配置（只读） */
  const DecoderOptions &Options() const { return opts_; }

private:
  DecoderOptions opts_;
  ThreadPool pool_;
};

} // namespace ctc

#endif // CTC_BATCH_DECODER_H_
