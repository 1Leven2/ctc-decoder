/**
 * 贪心解码器单元测试
 *
 * 覆盖场景：
 * 1. 基本功能：blank 正确去除、相邻重复正确合并
 * 2. 边界情况：空输入、单帧、全 blank、全重复 token
 * 3. 流式接口：Reset 后状态清空、Step 逐帧累积
 */

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "ctc/greedy_decoder.h"

using namespace ctc;

/* ─── 辅助函数：构造 log 概率矩阵 ────────────────────────────────── */

/**
 * 从二维 vector 构造行优先展平的 float 数组
 *
 * @param probs  每帧的概率向量 [num_frames][vocab_size]
 *               内部值会被 exp 转为对数概率（模拟 log_softmax 输出）
 * @return 展平后的对数概率数组
 */
static std::vector<float> MakeLogProbs(
    const std::vector<std::vector<float>>& probs) {
  int frames = static_cast<int>(probs.size());
  int vocab = frames > 0 ? static_cast<int>(probs[0].size()) : 0;
  std::vector<float> result(frames * vocab);

  for (int t = 0; t < frames; ++t) {
    for (int v = 0; v < vocab; ++v) {
      // 将原始概率转为 log 概率，模拟声学模型的 log_softmax 输出
      result[t * vocab + v] = std::log(probs[t][v]);
    }
  }
  return result;
}

/* ═══════════════════════════════════════════════════════════════
 *  测试：基本 CTC 合并规则
 * ═══════════════════════════════════════════════════════════════ */

/**
 * 测试空白符正确移除
 *
 * 输入序列：frame_0=token_1, frame_1=blank(0), frame_2=token_2
 * 期望输出：[1, 2]，blank(0) 被跳过
 */
TEST(GreedyDecoderTest, BlankRemoved) {
  // V=3: [blank(0), token_1, token_2]
  // T=3: token_1(0.9) → blank(0.9) → token_2(0.9)
  auto log_probs = MakeLogProbs({
      {0.1f, 0.8f, 0.1f},   // frame 0 → token_1 概率最高
      {0.8f, 0.1f, 0.1f},   // frame 1 → blank 概率最高
      {0.1f, 0.1f, 0.8f},   // frame 2 → token_2 概率最高
  });

  GreedyDecoder decoder(/*blank_id=*/0);
  auto results = decoder.Decode(log_probs.data(), 3, 3);

  ASSERT_EQ(results.size(), 1);
  // 期望：[1, 2]（blank 被过滤，1 和 2 都在）
  ASSERT_EQ(results[0].tokens.size(), 2);
  EXPECT_EQ(results[0].tokens[0], 1);  // token_1
  EXPECT_EQ(results[0].tokens[1], 2);  // token_2
}

/**
 * 测试相邻重复 token 合并
 *
 * 输入：连续两帧都输出同一 token（中间无 blank）
 * 期望：只保留一个
 */
TEST(GreedyDecoderTest, AdjacentDuplicateMerge) {
  auto log_probs = MakeLogProbs({
      {0.1f, 0.8f, 0.1f},   // frame 0 → token_1
      {0.1f, 0.8f, 0.1f},   // frame 1 → token_1 (相邻重复)
      {0.1f, 0.1f, 0.8f},   // frame 2 → token_2
  });

  GreedyDecoder decoder(0);
  auto results = decoder.Decode(log_probs.data(), 3, 3);

  ASSERT_EQ(results.size(), 1);
  // 期望：[1, 2]（连续两个 1 合并为一个）
  ASSERT_EQ(results[0].tokens.size(), 2);
  EXPECT_EQ(results[0].tokens[0], 1);
  EXPECT_EQ(results[0].tokens[1], 2);
}

/**
 * 测试 blank 打断重复合并
 *
 * 序列：token_1 → blank → token_1
 * 期望：[1, 1]（blank 打断了重复，两个 1 都应该保留）
 */
TEST(GreedyDecoderTest, BlankBreaksDuplicate) {
  auto log_probs = MakeLogProbs({
      {0.1f, 0.8f, 0.1f},   // frame 0 → token_1
      {0.8f, 0.1f, 0.1f},   // frame 1 → blank
      {0.1f, 0.8f, 0.1f},   // frame 2 → token_1
  });

  GreedyDecoder decoder(0);
  auto results = decoder.Decode(log_probs.data(), 3, 3);

  ASSERT_EQ(results.size(), 1);
  // 期望：[1, 1]（blank 打断，两端 1 都保留）
  ASSERT_EQ(results[0].tokens.size(), 2);
  EXPECT_EQ(results[0].tokens[0], 1);
  EXPECT_EQ(results[0].tokens[1], 1);
}

/* ═══════════════════════════════════════════════════════════════
 *  测试：边界情况
 * ═══════════════════════════════════════════════════════════════ */

/** 空输入应返回空结果 */
TEST(GreedyDecoderTest, EmptyInput) {
  GreedyDecoder decoder(0);
  auto results = decoder.Decode(nullptr, 0, 0);
  EXPECT_TRUE(results.empty());
}

/** 全 blank 输入应返回空 token 序列 */
TEST(GreedyDecoderTest, AllBlank) {
  auto log_probs = MakeLogProbs({
      {0.9f, 0.05f, 0.05f},  // blank 最高
      {0.9f, 0.05f, 0.05f},  // blank 最高
      {0.9f, 0.05f, 0.05f},  // blank 最高
  });

  GreedyDecoder decoder(0);
  auto results = decoder.Decode(log_probs.data(), 3, 3);

  ASSERT_EQ(results.size(), 1);
  EXPECT_TRUE(results[0].tokens.empty());
}

/** 单帧非 blank 输入 */
TEST(GreedyDecoderTest, SingleFrame) {
  auto log_probs = MakeLogProbs({
      {0.1f, 0.8f, 0.1f},
  });

  GreedyDecoder decoder(0);
  auto results = decoder.Decode(log_probs.data(), 1, 3);

  ASSERT_EQ(results.size(), 1);
  ASSERT_EQ(results[0].tokens.size(), 1);
  EXPECT_EQ(results[0].tokens[0], 1);
}

/* ═══════════════════════════════════════════════════════════════
 *  测试：流式接口
 * ═══════════════════════════════════════════════════════════════ */

/** 流式 Step 累积结果应与非流式 Decode 一致 */
TEST(GreedyDecoderTest, StreamingVsNonStreaming) {
  auto log_probs = MakeLogProbs({
      {0.1f, 0.8f, 0.1f},
      {0.8f, 0.1f, 0.1f},
      {0.1f, 0.1f, 0.8f},
  });

  // 非流式
  GreedyDecoder batch_dec(0);
  auto batch_results = batch_dec.Decode(log_probs.data(), 3, 3);

  // 流式
  GreedyDecoder stream_dec(0);
  for (int t = 0; t < 3; ++t) {
    stream_dec.Step(log_probs.data() + t * 3, 3);
  }
  auto stream_results = stream_dec.Results(1);

  ASSERT_EQ(batch_results.size(), stream_results.size());
  EXPECT_EQ(batch_results[0].tokens, stream_results[0].tokens);
  EXPECT_FLOAT_EQ(batch_results[0].score, stream_results[0].score);
}

/** Reset 后状态应完全清空 */
TEST(GreedyDecoderTest, ResetClearsState) {
  auto log_probs = MakeLogProbs({
      {0.1f, 0.8f, 0.1f},
  });

  GreedyDecoder decoder(0);
  decoder.Step(log_probs.data(), 3);
  decoder.Reset();

  auto results = decoder.Results(1);
  ASSERT_EQ(results.size(), 1);
  EXPECT_TRUE(results[0].tokens.empty());
  EXPECT_FLOAT_EQ(results[0].score, 0.0f);
}

/* ═══════════════════════════════════════════════════════════════
 *  测试：分数合理性
 * ═══════════════════════════════════════════════════════════════ */

/** 贪心解码的分数应为每帧 argmax 概率的对数和，≤ 0 */
TEST(GreedyDecoderTest, ScoreNegative) {
  auto log_probs = MakeLogProbs({
      {0.1f, 0.8f, 0.1f},
      {0.1f, 0.1f, 0.8f},
  });

  GreedyDecoder decoder(0);
  auto results = decoder.Decode(log_probs.data(), 2, 3);

  ASSERT_EQ(results.size(), 1);
  // log(probability) ≤ 0，因为概率 ≤ 1
  EXPECT_LT(results[0].score, 0.0f);
}
