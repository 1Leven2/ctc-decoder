/**
 * CTC 解码器 C API
 *
 * 纯 C 接口（extern "C"），可被 C、Python ctypes、Go cgo、Rust FFI 等调用。
 * 透明句柄设计——所有内部状态通过不完整类型指针隐藏。
 *
 * 返回值约定：
 *   0（CTC_SUCCESS）— 成功
 *   非零（CTC_ERROR）— 失败，调用 ctc_last_error() 获取错误信息
 *
 * 线程安全：
 *   ctc_last_error() 使用 thread_local 存储，
 *   不同线程的错误信息互不干扰。
 *
 * 使用示例（C 语言）：
 *   ctc_decoder_options_t opts = {
 *       .beam_size = 10, .blank_id = 0, .n_best = 1,
 *       .cutoff_threshold = 0.0f, .num_threads = 4
 *   };
 *   ctc_batch_decoder_t *dec = ctc_batch_decoder_create(&opts);
 *   ctc_batch_results_t *results = NULL;
 *   if (ctc_decode_batch(dec, data, frame_counts, N, V, &results) != 0) {
 *       fprintf(stderr, "error: %s\n", ctc_last_error());
 *       return;
 *   }
 *   // ... 访问 results ...
 *   ctc_batch_results_destroy(results);
 *   ctc_batch_decoder_destroy(dec);
 */

#ifndef CTC_API_H_
#define CTC_API_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

/* ─── 返回值 ──────────────────────────────────────── */
#define CTC_SUCCESS 0
#define CTC_ERROR (-1)

/* ─── 透明句柄（不完整类型，外部只持有指针） ────────── */
typedef struct ctc_batch_decoder_t ctc_batch_decoder_t;
typedef struct ctc_batch_results_t ctc_batch_results_t;

/* ─── 配置结构体 ──────────────────────────────────── */
typedef struct {
  /** beam 宽度，默认 10。取值 1 时退化为贪心解码 */
  int beam_size;
  /** CTC blank token 索引，默认 0 */
  int blank_id;
  /** 返回的最佳假设数，默认 1 */
  int n_best;
  /** 剪枝阈值（log 空间），0.0 表示不启用 */
  float cutoff_threshold;
  /** 线程池大小，0 表示自动取硬件并发数 */
  int num_threads;
} ctc_decoder_options_t;

/* ═══════════════════════════════════════════════════════════════
 *  解码器生命周期
 * ═══════════════════════════════════════════════════════════════ */

/**
 * 创建批量解码器
 *
 * @param opts 解码配置，NULL 则使用默认参数
 * @return     解码器句柄，失败返回 NULL（调用 ctc_last_error() 获取原因）
 */
ctc_batch_decoder_t *
ctc_batch_decoder_create(const ctc_decoder_options_t *opts);

/** 销毁解码器，释放所有资源 */
void ctc_batch_decoder_destroy(ctc_batch_decoder_t *decoder);

/* ═══════════════════════════════════════════════════════════════
 *  批量解码
 * ═══════════════════════════════════════════════════════════════ */

/**
 * 批量解码 N 条语音
 *
 * @param decoder         解码器句柄
 * @param batch_data      平坦拼接的 log_softmax 概率矩阵（行优先，按语音拼接）
 * @param frame_counts    每条语音的帧数数组 [T0, T1, ..., T_{N-1}]
 * @param num_utterances  语音条数 N
 * @param vocab_size      词表大小 V（批量内共享）
 * @param results_out     输出：解码结果句柄，调用者负责释放
 * @return                CTC_SUCCESS 或 CTC_ERROR
 */
int ctc_decode_batch(ctc_batch_decoder_t *decoder, const float *batch_data,
                     const int *frame_counts, int num_utterances,
                     int vocab_size, ctc_batch_results_t **results_out);

/* ═══════════════════════════════════════════════════════════════
 *  结果访问
 * ═══════════════════════════════════════════════════════════════ */

/** 返回批量结果中的语音条数 */
int ctc_batch_results_num_utterances(ctc_batch_results_t *results);

/**
 * 返回第 utterance_idx 条语音的假设数量（通常等于 n_best）
 *
 * @param results        结果句柄
 * @param utterance_idx  语音索引 [0, num_utterances)
 * @return               假设数量，失败返回 CTC_ERROR
 */
int ctc_batch_results_num_hyps(ctc_batch_results_t *results,
                               int utterance_idx);

/**
 * 获取第 utterance_idx 条语音的第 hyp_idx 个假设的 token 序列
 *
 * @param results        结果句柄
 * @param utterance_idx  语音索引
 * @param hyp_idx        假设索引（0 = 最优）
 * @param out_num_tokens 输出：token 序列长度
 * @return               指向 token 数组的指针（生命周期与 results 句柄相同），
 *                       失败返回 NULL
 */
const int *ctc_batch_results_tokens(ctc_batch_results_t *results,
                                    int utterance_idx, int hyp_idx,
                                    int *out_num_tokens);

/**
 * 获取第 utterance_idx 条语音的第 hyp_idx 个假设的分数（log 概率）
 */
float ctc_batch_results_score(ctc_batch_results_t *results, int utterance_idx,
                               int hyp_idx);

/**
 * 获取第 utterance_idx 条语音的第 hyp_idx 个假设的时间戳（帧号）
 *
 * @param out_num_timestamps 输出：时间戳数组长度
 * @return                   指向时间戳数组的指针，
 *                           不需要时间信息时可忽略（返回 NULL 也是合法的）
 */
const int *
ctc_batch_results_timestamps(ctc_batch_results_t *results, int utterance_idx,
                              int hyp_idx, int *out_num_timestamps);

/** 销毁结果句柄，释放内部存储 */
void ctc_batch_results_destroy(ctc_batch_results_t *results);

/* ═══════════════════════════════════════════════════════════════
 *  错误处理
 * ═══════════════════════════════════════════════════════════════ */

/**
 * 获取最后一次错误信息
 *
 * 线程安全：每个线程独立存储，不同线程的错误互不覆盖。
 *
 * @return 错误描述的 C 字符串，无错误时返回空字符串 ""
 */
const char *ctc_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* CTC_API_H_ */
