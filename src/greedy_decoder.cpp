#include "ctc/greedy_decoder.h"

#include <algorithm>
#include <limits>

namespace ctc {

GreedyDecoder::GreedyDecoder(int blank_id)
    : blank_id_(blank_id), prev_token_(-1), total_score_(0.0f),
      frame_count_(0) {}

std::vector<DecodeResult>
GreedyDecoder::Decode(const float *log_probs, int num_frames, int vocab_size) {
  // 重置内部状态，准备新一轮解码
  Reset();

  if (num_frames <= 0 || vocab_size <= 0) {
    return {};
  }

  // 逐帧做 argmax，累积结果
  for (int t = 0; t < num_frames; ++t) {
    Step(log_probs + t * vocab_size, vocab_size);
  }

  return Results(1);
}

void GreedyDecoder::Step(const float *log_probs, int vocab_size) {
  if (vocab_size <= 0)
    return;

  // 找到当前帧概率最大的 token（argmax）
  // std::max_element 对原始数组做 O(V) 线性扫描
  int best_token =
      static_cast<int>(std::max_element(log_probs, log_probs + vocab_size) -
                       log_probs); // 获取的是指针差值，转换为索引
  float best_score = log_probs[best_token];

  // 累加对数概率
  total_score_ += best_score;

  /* ─── CTC 合并规则 ───────────────────────────────────────
   *
   * 条件 1: token != blank  — 过滤所有空白符
   * 条件 2: token != prev   — 相邻重复合并（连续相同非 blank token
   * 只保留第一个）
   *
   * 注意：即使遇到 blank，prev_token_ 也会更新为 blank_id，
   *       这样才能正确处理序列 [a, <b>, a] 的情况 —— blank 打断了重复，
   *       前后两个 a 都应该保留。 */

  if (best_token != blank_id_ && best_token != prev_token_) {
    tokens_.push_back(best_token);
    // 记录该 token 首次发声的帧号
    timestamps_.push_back(frame_count_);
  }

  prev_token_ = best_token;
  ++frame_count_;
}

std::vector<DecodeResult> GreedyDecoder::Results(int /*n*/) const {
  DecodeResult result;
  result.tokens = tokens_;
  result.score = total_score_;

  // 构建帧号序列（当前实现未追踪精确帧号，留用 tokens_.size() 占位）
  // 如需精确时间对齐，流式调用方可在外部从帧计数器推导
  result.timestamps = timestamps_;

  return {result};
}

void GreedyDecoder::Reset() {
  prev_token_ = -1;
  tokens_.clear();
  total_score_ = 0.0f;
  timestamps_.clear();
  frame_count_ = 0;
}

} // namespace ctc
