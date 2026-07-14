/**
 * 前缀束搜索单元测试
 *
 * 覆盖场景：
 * 1. 基本 CTC 行为：blank 去除、相邻合并、blank 打断合并
 * 2. beam_size 影响：beam=1 vs beam>1 的行为差异
 * 3. 贪心等价性：beam=1 时结果应与贪心解码一致
 * 4. n_best 返回正确数量
 * 5. 边界情况：空输入、全 blank
 * 6. 流式与非流式一致性
 * 7. 分数单调性：更优路径应有更高分数
 */

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "ctc/greedy_decoder.h"
#include "ctc/prefix_beam_search.h"

using namespace ctc;

/* ─── 辅助函数 ─────────────────────────────────────────────────── */

/**
 * 从二维概率值构造行优先展平的 log 概率数组
 *
 * 每帧的向量会被 log 化，模拟声学模型 log_softmax 输出。
 * 概率值决定 argmax 的排序，log 值用于 CTC 递推计算。
 */
static std::vector<float> MakeLogProbs(
    const std::vector<std::vector<float>>& probs) {
  int frames = static_cast<int>(probs.size());
  int vocab = frames > 0 ? static_cast<int>(probs[0].size()) : 0;
  std::vector<float> result(frames * vocab);

  for (int t = 0; t < frames; ++t) {
    for (int v = 0; v < vocab; ++v) {
      result[t * vocab + v] = std::log(probs[t][v]);
    }
  }
  return result;
}

/**
 * 从已归一化的 log 概率值构造数据（不需要再做 log）
 */
static std::vector<float> MakeLogProbsDirect(
    const std::vector<std::vector<float>>& log_probs) {
  int frames = static_cast<int>(log_probs.size());
  int vocab = frames > 0 ? static_cast<int>(log_probs[0].size()) : 0;
  std::vector<float> result(frames * vocab);

  for (int t = 0; t < frames; ++t) {
    for (int v = 0; v < vocab; ++v) {
      result[t * vocab + v] = log_probs[t][v];
    }
  }
  return result;
}

/* ═══════════════════════════════════════════════════════════════
 *  测试：基本 CTC 行为
 * ═══════════════════════════════════════════════════════════════ */

/**
 * blank 正确去除
 *
 * 序列：token_1 → blank → token_2
 * 期望：[1, 2]
 */
TEST(BeamSearchTest, BlankRemoved) {
  auto log_probs = MakeLogProbs({
      {0.05f, 0.90f, 0.05f},   // token_1 支配
      {0.90f, 0.05f, 0.05f},   // blank 支配
      {0.05f, 0.05f, 0.90f},   // token_2 支配
  });

  DecoderOptions opts;
  opts.beam_size = 3;
  PrefixBeamSearch decoder(opts);
  auto results = decoder.Decode(log_probs.data(), 3, 3);

  ASSERT_GE(results.size(), 1);
  EXPECT_EQ(results[0].tokens.size(), 2);
  if (results[0].tokens.size() == 2) {
    EXPECT_EQ(results[0].tokens[0], 1);
    EXPECT_EQ(results[0].tokens[1], 2);
  }
}

/**
 * 相邻重复 token 被合并
 *
 * token_1 → token_1 → token_2（连续两帧 1 无 blank 隔开）
 * 期望：[1, 2]
 */
TEST(BeamSearchTest, AdjacentDuplicatesMerged) {
  auto log_probs = MakeLogProbs({
      {0.05f, 0.90f, 0.05f},   // token_1
      {0.05f, 0.90f, 0.05f},   // token_1（重复）
      {0.05f, 0.05f, 0.90f},   // token_2
  });

  DecoderOptions opts;
  opts.beam_size = 3;
  PrefixBeamSearch decoder(opts);
  auto results = decoder.Decode(log_probs.data(), 3, 3);

  ASSERT_GE(results.size(), 1);
  // 最佳结果不应包含重复的 1（连续 1 被合并）
  ASSERT_EQ(results[0].tokens.size(), 2);
  EXPECT_EQ(results[0].tokens[0], 1);
  EXPECT_EQ(results[0].tokens[1], 2);
}

/**
 * blank 打断重复合并
 *
 * token_1 → blank → token_1
 * 期望：[1, 1]
 */
TEST(BeamSearchTest, BlankBreaksDuplicates) {
  auto log_probs = MakeLogProbs({
      {0.05f, 0.90f, 0.05f},   // token_1
      {0.90f, 0.05f, 0.05f},   // blank
      {0.05f, 0.90f, 0.05f},   // token_1
  });

  DecoderOptions opts;
  opts.beam_size = 3;
  PrefixBeamSearch decoder(opts);
  auto results = decoder.Decode(log_probs.data(), 3, 3);

  ASSERT_GE(results.size(), 1);
  ASSERT_EQ(results[0].tokens.size(), 2);
  EXPECT_EQ(results[0].tokens[0], 1);
  EXPECT_EQ(results[0].tokens[1], 1);
}

/* ═══════════════════════════════════════════════════════════════
 *  测试：beam_size 行为
 * ═══════════════════════════════════════════════════════════════ */

/**
 * beam=1 的前缀束搜索应等价贪心解码
 *
 * 因为 beam=1 每帧只保留最优假设 = 每帧 argmax。
 * 前缀束搜索的递推公式在 beam=1 时退化为贪心等价行为。
 */
TEST(BeamSearchTest, BeamSize1EquivalentToGreedy) {
  // 构造一组有多条路径的数据，确保贪心和 beam=1 得到相同结果
  auto log_probs = MakeLogProbs({
      {0.05f, 0.80f, 0.10f, 0.05f},   // token_1 占优
      {0.05f, 0.05f, 0.85f, 0.05f},   // token_2 占优
      {0.85f, 0.05f, 0.05f, 0.05f},   // blank 占优
      {0.05f, 0.05f, 0.05f, 0.85f},   // token_3 占优
  });

  // 贪心解码
  GreedyDecoder greedy(0);
  auto greedy_results = greedy.Decode(log_probs.data(), 4, 4);

  // 前缀束搜索 beam=1
  DecoderOptions opts;
  opts.beam_size = 1;
  PrefixBeamSearch beam(opts);
  auto beam_results = beam.Decode(log_probs.data(), 4, 4);

  ASSERT_EQ(greedy_results.size(), 1);
  ASSERT_GE(beam_results.size(), 1);

  // beam=1 的结果应与贪心一致
  EXPECT_EQ(greedy_results[0].tokens, beam_results[0].tokens);
}

/**
 * beam_size > 1 不会比 beam=1 差
 *
 * 更大的 beam 有更多搜索空间，最优假设的分数应不低于 beam=1。
 */
TEST(BeamSearchTest, LargerBeamNotWorse) {
  auto log_probs = MakeLogProbs({
      {0.01f, 0.40f, 0.30f, 0.20f, 0.09f},  // 多个 token 接近
      {0.01f, 0.35f, 0.35f, 0.20f, 0.09f},  // 歧义帧
      {0.60f, 0.10f, 0.10f, 0.10f, 0.10f},  // blank 占优
      {0.01f, 0.09f, 0.20f, 0.40f, 0.30f},  // 又一次歧义
  });

  DecoderOptions opts1;
  opts1.beam_size = 1;
  PrefixBeamSearch beam1(opts1);
  auto r1 = beam1.Decode(log_probs.data(), 4, 5);

  DecoderOptions opts5;
  opts5.beam_size = 5;
  PrefixBeamSearch beam5(opts5);
  auto r5 = beam5.Decode(log_probs.data(), 4, 5);

  ASSERT_GE(r1.size(), 1);
  ASSERT_GE(r5.size(), 1);

  // beam=5 的最优分数应 ≥ beam=1 的最优分数
  EXPECT_GE(r5[0].score, r1[0].score - 0.01f);
}

/* ═══════════════════════════════════════════════════════════════
 *  测试：n_best 输出
 * ═══════════════════════════════════════════════════════════════ */

/** n_best=3 时应返回最多 3 个结果，且分数单调递减 */
TEST(BeamSearchTest, NBestReturnsCorrectCount) {
  auto log_probs = MakeLogProbs({
      {0.05f, 0.40f, 0.30f, 0.25f},
      {0.05f, 0.10f, 0.40f, 0.45f},
      {0.05f, 0.35f, 0.30f, 0.30f},
  });

  DecoderOptions opts;
  opts.beam_size = 5;
  opts.n_best = 3;
  PrefixBeamSearch decoder(opts);
  auto results = decoder.Decode(log_probs.data(), 3, 4);

  ASSERT_GE(results.size(), 1);
  ASSERT_LE(results.size(), 3);

  // 分数应单调递减
  for (size_t i = 1; i < results.size(); ++i) {
    EXPECT_GE(results[i - 1].score, results[i].score);
  }
}

/* ═══════════════════════════════════════════════════════════════
 *  测试：边界情况
 * ═══════════════════════════════════════════════════════════════ */

/** 空输入返回空结果 */
TEST(BeamSearchTest, EmptyInput) {
  DecoderOptions opts;
  PrefixBeamSearch decoder(opts);
  auto results = decoder.Decode(nullptr, 0, 0);
  EXPECT_TRUE(results.empty());
}

/** 全 blank 输入应产生空 token 序列 */
TEST(BeamSearchTest, AllBlank) {
  auto log_probs = MakeLogProbs({
      {0.95f, 0.02f, 0.01f, 0.02f},
      {0.95f, 0.02f, 0.01f, 0.02f},
  });

  DecoderOptions opts;
  opts.beam_size = 3;
  PrefixBeamSearch decoder(opts);
  auto results = decoder.Decode(log_probs.data(), 2, 4);

  ASSERT_GE(results.size(), 1);
  // 最优结果应该是空序列（全 blank）
  EXPECT_TRUE(results[0].tokens.empty());
}

/** 单帧非 blank */
TEST(BeamSearchTest, SingleFrameNonBlank) {
  auto log_probs = MakeLogProbs({
      {0.05f, 0.85f, 0.10f},
  });

  DecoderOptions opts;
  opts.beam_size = 3;
  PrefixBeamSearch decoder(opts);
  auto results = decoder.Decode(log_probs.data(), 1, 3);

  ASSERT_GE(results.size(), 1);
  ASSERT_EQ(results[0].tokens.size(), 1);
  EXPECT_EQ(results[0].tokens[0], 1);
}

/* ═══════════════════════════════════════════════════════════════
 *  测试：流式接口
 * ═══════════════════════════════════════════════════════════════ */

/** 流式与非流式结果一致 */
TEST(BeamSearchTest, StreamingConsistency) {
  auto log_probs = MakeLogProbs({
      {0.05f, 0.85f, 0.10f},
      {0.85f, 0.05f, 0.10f},
      {0.05f, 0.10f, 0.85f},
  });

  // 非流式
  DecoderOptions opts;
  opts.beam_size = 3;
  PrefixBeamSearch batch(opts);
  auto batch_results = batch.Decode(log_probs.data(), 3, 3);

  // 流式
  PrefixBeamSearch stream(opts);
  for (int t = 0; t < 3; ++t) {
    stream.Step(log_probs.data() + t * 3, 3);
  }
  auto stream_results = stream.Results(1);

  ASSERT_GE(batch_results.size(), 1);
  ASSERT_GE(stream_results.size(), 1);
  EXPECT_EQ(batch_results[0].tokens, stream_results[0].tokens);
  EXPECT_FLOAT_EQ(batch_results[0].score, stream_results[0].score);
}

/** Reset 后状态清空 */
TEST(BeamSearchTest, ResetClearsState) {
  auto log_probs = MakeLogProbs({
      {0.05f, 0.85f, 0.10f},
  });

  DecoderOptions opts;
  opts.beam_size = 3;
  PrefixBeamSearch decoder(opts);
  decoder.Step(log_probs.data(), 3);
  decoder.Reset();

  auto results = decoder.Results(1);
  ASSERT_GE(results.size(), 1);
  EXPECT_TRUE(results[0].tokens.empty());
}

/* ═══════════════════════════════════════════════════════════════
 *  测试：分数性质
 * ═══════════════════════════════════════════════════════════════ */

/** log 概率应 ≤ 0 */
TEST(BeamSearchTest, ScoreNonPositive) {
  auto log_probs = MakeLogProbs({
      {0.05f, 0.80f, 0.15f},
      {0.05f, 0.15f, 0.80f},
  });

  DecoderOptions opts;
  opts.beam_size = 3;
  PrefixBeamSearch decoder(opts);
  auto results = decoder.Decode(log_probs.data(), 2, 3);

  ASSERT_GE(results.size(), 1);
  EXPECT_LE(results[0].score, 0.0f);
}
