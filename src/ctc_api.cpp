/**
 * CTC 解码器 C API 实现
 *
 * 将 BatchDecoder（C++）包装为纯 C 接口。透明句柄模式：
 *   ctc_batch_decoder_t → 包装 BatchDecoder*
 *   ctc_batch_results_t → 包装解码结果和元数据
 *
 * 错误处理：通过 thread_local 字符串存储最后一条错误信息，
 * 类似 CUDA cudaGetLastError / libcurl curl_easy_strerror 风格。
 */

#include "ctc/ctc_api.h"

#include <cstring>
#include <string>

#include "ctc/batch_decoder.h"
#include "ctc/types.h"

/* ─── 默认配置 ────────────────────────────────────── */
static const ctc_decoder_options_t kDefaultOptions = {
    10,   // beam_size
    0,    // blank_id
    1,    // n_best
    0.0f, // cutoff_threshold
    0     // num_threads (auto)
};

/* ─── 透明句柄定义 ────────────────────────────────── */

/** 解码器句柄：包装 BatchDecoder */
struct ctc_batch_decoder_t {
  ctc::BatchDecoder *impl;
};

/** 结果句柄：持有解码结果和元数据，供访问器函数读取 */
struct ctc_batch_results_t {
  std::vector<std::vector<ctc::DecodeResult>> data;
  int num_utterances;
  int n_best;
};

/* ─── thread_local 错误状态 ────────────────────────── */

static thread_local std::string tls_last_error;

static const char *SetError(const char *msg) {
  tls_last_error = msg;
  return msg;
}

#define CHECK_NOT_NULL(ptr, name)                                              \
  do {                                                                         \
    if (!(ptr)) {                                                              \
      SetError(name " is NULL");                                               \
      return CTC_ERROR;                                                        \
    }                                                                          \
  } while (0)

// 指针返回版本的 NULL 检查（返回 nullptr 而非 CTC_ERROR）
#define CHECK_NOT_NULL_PTR(ptr, name)                                          \
  do {                                                                         \
    if (!(ptr)) {                                                              \
      SetError(name " is NULL");                                               \
      return nullptr;                                                          \
    }                                                                          \
  } while (0)

/* ═══════════════════════════════════════════════════════════════
 *  解码器生命周期
 * ═══════════════════════════════════════════════════════════════ */

ctc_batch_decoder_t *
ctc_batch_decoder_create(const ctc_decoder_options_t *opts) {
  if (!opts)
    opts = &kDefaultOptions;

  ctc::DecoderOptions cpp_opts;
  cpp_opts.beam_size = opts->beam_size;
  cpp_opts.blank_id = opts->blank_id;
  cpp_opts.n_best = opts->n_best;
  cpp_opts.cutoff_threshold = opts->cutoff_threshold;

  auto *handle = new (std::nothrow) ctc_batch_decoder_t;
  if (!handle) {
    SetError("ctc_batch_decoder_create: out of memory");
    return nullptr;
  }

  try {
    handle->impl = new ctc::BatchDecoder(cpp_opts, opts->num_threads);
  } catch (const std::exception &e) {
    SetError(e.what());
    delete handle;
    return nullptr;
  }

  tls_last_error.clear();
  return handle;
}

void ctc_batch_decoder_destroy(ctc_batch_decoder_t *decoder) {
  if (!decoder)
    return;
  delete decoder->impl;
  delete decoder;
}

/* ═══════════════════════════════════════════════════════════════
 *  批量解码
 * ═══════════════════════════════════════════════════════════════ */

int ctc_decode_batch(ctc_batch_decoder_t *decoder, const float *batch_data,
                     const int *frame_counts, int num_utterances,
                     int vocab_size, ctc_batch_results_t **results_out) {
  CHECK_NOT_NULL(decoder, "decoder");
  CHECK_NOT_NULL(batch_data, "batch_data");
  CHECK_NOT_NULL(frame_counts, "frame_counts");
  CHECK_NOT_NULL(results_out, "results_out");

  if (num_utterances < 0 || vocab_size < 0) {
    SetError("ctc_decode_batch: num_utterances and vocab_size must be >= 0");
    return CTC_ERROR;
  }

  auto *results = new (std::nothrow) ctc_batch_results_t;
  if (!results) {
    SetError("ctc_decode_batch: out of memory");
    return CTC_ERROR;
  }

  try {
    results->data = decoder->impl->DecodeBatch(batch_data, frame_counts,
                                                num_utterances, vocab_size);
    results->num_utterances = num_utterances;
    results->n_best = decoder->impl->Options().n_best;
  } catch (const std::exception &e) {
    SetError(e.what());
    delete results;
    return CTC_ERROR;
  }

  *results_out = results;
  tls_last_error.clear();
  return CTC_SUCCESS;
}

/* ═══════════════════════════════════════════════════════════════
 *  结果访问器
 * ═══════════════════════════════════════════════════════════════ */

int ctc_batch_results_num_utterances(ctc_batch_results_t *results) {
  if (!results) {
    SetError("ctc_batch_results_num_utterances: results is NULL");
    return CTC_ERROR;
  }
  return results->num_utterances;
}

int ctc_batch_results_num_hyps(ctc_batch_results_t *results,
                               int utterance_idx) {
  CHECK_NOT_NULL(results, "results");
  if (utterance_idx < 0 || utterance_idx >= results->num_utterances) {
    SetError("ctc_batch_results_num_hyps: utterance_idx out of range");
    return CTC_ERROR;
  }
  return static_cast<int>(results->data[utterance_idx].size());
}

const int *ctc_batch_results_tokens(ctc_batch_results_t *results,
                                    int utterance_idx, int hyp_idx,
                                    int *out_num_tokens) {
  CHECK_NOT_NULL_PTR(results, "results");
  CHECK_NOT_NULL_PTR(out_num_tokens, "out_num_tokens");

  if (utterance_idx < 0 || utterance_idx >= results->num_utterances) {
    SetError("ctc_batch_results_tokens: utterance_idx out of range");
    return nullptr;
  }
  auto &hyps = results->data[utterance_idx];
  if (hyp_idx < 0 || hyp_idx >= static_cast<int>(hyps.size())) {
    SetError("ctc_batch_results_tokens: hyp_idx out of range");
    return nullptr;
  }
  *out_num_tokens = static_cast<int>(hyps[hyp_idx].tokens.size());
  return hyps[hyp_idx].tokens.data();
}

float ctc_batch_results_score(ctc_batch_results_t *results, int utterance_idx,
                               int hyp_idx) {
  if (!results) {
    SetError("ctc_batch_results_score: results is NULL");
    return 0.0f;
  }
  if (utterance_idx < 0 || utterance_idx >= results->num_utterances)
    return 0.0f;
  auto &hyps = results->data[utterance_idx];
  if (hyp_idx < 0 || hyp_idx >= static_cast<int>(hyps.size()))
    return 0.0f;
  return hyps[hyp_idx].score;
}

const int *
ctc_batch_results_timestamps(ctc_batch_results_t *results, int utterance_idx,
                              int hyp_idx, int *out_num_timestamps) {
  CHECK_NOT_NULL_PTR(results, "results");
  CHECK_NOT_NULL_PTR(out_num_timestamps, "out_num_timestamps");

  if (utterance_idx < 0 || utterance_idx >= results->num_utterances) {
    SetError("ctc_batch_results_timestamps: utterance_idx out of range");
    return nullptr;
  }
  auto &hyps = results->data[utterance_idx];
  if (hyp_idx < 0 || hyp_idx >= static_cast<int>(hyps.size())) {
    SetError("ctc_batch_results_timestamps: hyp_idx out of range");
    return nullptr;
  }
  *out_num_timestamps = static_cast<int>(hyps[hyp_idx].timestamps.size());
  if (hyps[hyp_idx].timestamps.empty())
    return nullptr;
  return hyps[hyp_idx].timestamps.data();
}

void ctc_batch_results_destroy(ctc_batch_results_t *results) { delete results; }

/* ═══════════════════════════════════════════════════════════════
 *  错误处理
 * ═══════════════════════════════════════════════════════════════ */

const char *ctc_last_error(void) { return tls_last_error.c_str(); }
