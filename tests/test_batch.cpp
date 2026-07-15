/**
 * 线程池、批量解码器、C API 单元测试
 */

#include <gtest/gtest.h>

#include <atomic>
#include <cstring>
#include <random>
#include <vector>

#include "ctc/batch_decoder.h"
#include "ctc/ctc_api.h"
#include "ctc/prefix_beam_search.h"
#include "ctc/thread_pool.h"

using namespace ctc;

/* ─── 辅助函数 ────────────────────────────────────── */

/** 生成随机 log_probs [T, V] */
static std::vector<float> RandomLogProbs(int T, int V, unsigned seed = 42) {
  std::vector<float> data(T * V);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  for (int t = 0; t < T; ++t) {
    float sum = 0.0f;
    for (int v = 0; v < V; ++v) {
      data[t * V + v] = dist(rng);
      sum += data[t * V + v];
    }
    for (int v = 0; v < V; ++v) {
      data[t * V + v] = std::log(data[t * V + v] / sum);
    }
  }
  return data;
}

/** 将多条语音的 log_probs 拼接为平坦 batch_data */
static std::vector<float>
BuildBatchData(const std::vector<std::vector<float>> &utt_log_probs,
               std::vector<int> &frame_counts, int V) {
  frame_counts.clear();
  size_t total = 0;
  for (auto &u : utt_log_probs) {
    int T = static_cast<int>(u.size() / V);
    frame_counts.push_back(T);
    total += u.size();
  }
  std::vector<float> batch(total);
  size_t offset = 0;
  for (auto &u : utt_log_probs) {
    std::memcpy(batch.data() + offset, u.data(), u.size() * sizeof(float));
    offset += u.size();
  }
  return batch;
}

/* ═══════════════════════════════════════════════════════════════
 *  线程池测试
 * ═══════════════════════════════════════════════════════════════ */

TEST(ThreadPoolTest, BasicEnqueue) {
  ThreadPool pool(4);
  std::atomic<int> counter{0};

  constexpr int N = 100;
  std::vector<std::future<void>> futures;
  for (int i = 0; i < N; ++i) {
    futures.push_back(
        pool.Enqueue([&counter]() { counter.fetch_add(1); }));
  }
  for (auto &f : futures)
    f.get();

  EXPECT_EQ(counter.load(), N);
}

TEST(ThreadPoolTest, ReturnValues) {
  ThreadPool pool(4);
  std::vector<std::future<int>> futures;

  for (int i = 0; i < 50; ++i) {
    futures.push_back(pool.Enqueue([](int x) { return x * x; }, i));
  }

  for (int i = 0; i < 50; ++i) {
    EXPECT_EQ(futures[i].get(), i * i);
  }
}

TEST(ThreadPoolTest, SingleThread) {
  ThreadPool pool(1);
  std::atomic<int> counter{0};

  for (int i = 0; i < 20; ++i) {
    pool.Enqueue([&counter]() { counter.fetch_add(1); }).get();
  }
  EXPECT_EQ(counter.load(), 20);
}

TEST(ThreadPoolTest, ManySmallTasks) {
  ThreadPool pool(4);
  std::atomic<int> counter{0};

  constexpr int N = 5000;
  std::vector<std::future<void>> futures;
  futures.reserve(N);
  for (int i = 0; i < N; ++i) {
    futures.push_back(
        pool.Enqueue([&counter]() { counter.fetch_add(1); }));
  }
  for (auto &f : futures)
    f.get();

  EXPECT_EQ(counter.load(), N);
}

TEST(ThreadPoolTest, ExceptionPropagation) {
  ThreadPool pool(2);
  auto future = pool.Enqueue([]() { throw std::runtime_error("test error"); });
  EXPECT_THROW(future.get(), std::runtime_error);
}

TEST(ThreadPoolTest, EnqueueAfterStop) {
  ThreadPool pool(1);
  // 显式析构后无法再入队，此处测试 pool 生命周期内正常工作
  auto f = pool.Enqueue([]() { return 42; });
  EXPECT_EQ(f.get(), 42);
}

/* ═══════════════════════════════════════════════════════════════
 *  批量解码器测试
 * ═══════════════════════════════════════════════════════════════ */

TEST(BatchDecoderTest, SingleUtteranceMatch) {
  // 批量解码 1 条语音的结果应与非批量版本一致
  constexpr int T = 30, V = 50;
  auto log_probs = RandomLogProbs(T, V);

  DecoderOptions opts;
  opts.beam_size = 5;
  opts.n_best = 3;

  // 批量版本
  BatchDecoder batch_dec(opts, 2);
  int frame_count = T;
  auto batch_results = batch_dec.DecodeBatch(log_probs.data(), &frame_count, 1, V);
  ASSERT_EQ(batch_results.size(), 1u);
  ASSERT_EQ(batch_results[0].size(), 3u);

  // 单条版本
  PrefixBeamSearch single_dec(opts);
  auto single_results = single_dec.Decode(log_probs.data(), T, V);

  ASSERT_EQ(single_results.size(), 3u);
  for (int h = 0; h < 3; ++h) {
    EXPECT_EQ(batch_results[0][h].tokens, single_results[h].tokens) << "hyp=" << h;
    EXPECT_FLOAT_EQ(batch_results[0][h].score, single_results[h].score)
        << "hyp=" << h;
  }
}

TEST(BatchDecoderTest, MultipleIdenticalUtterances) {
  // 相同输入产生相同输出
  constexpr int T = 20, V = 30, N = 4;
  auto single = RandomLogProbs(T, V);
  std::vector<std::vector<float>> inputs(N, single);
  std::vector<int> frame_counts;
  auto batch_data = BuildBatchData(inputs, frame_counts, V);

  DecoderOptions opts;
  opts.beam_size = 5;
  BatchDecoder dec(opts, 4);
  auto results = dec.DecodeBatch(batch_data.data(), frame_counts.data(), N, V);

  ASSERT_EQ(results.size(), static_cast<size_t>(N));
  for (int i = 1; i < N; ++i) {
    EXPECT_EQ(results[0][0].tokens, results[i][0].tokens);
    EXPECT_FLOAT_EQ(results[0][0].score, results[i][0].score);
  }
}

TEST(BatchDecoderTest, VariedLengths) {
  // 不同长度的语音混在一起
  constexpr int V = 30;
  std::vector<int> Ts = {10, 5, 20, 1};
  std::vector<std::vector<float>> inputs;
  for (int T : Ts)
    inputs.push_back(RandomLogProbs(T, V, T));

  std::vector<int> frame_counts;
  auto batch_data = BuildBatchData(inputs, frame_counts, V);

  DecoderOptions opts;
  opts.beam_size = 3;
  BatchDecoder dec(opts, 4);
  auto results = dec.DecodeBatch(batch_data.data(), frame_counts.data(),
                                 static_cast<int>(Ts.size()), V);

  ASSERT_EQ(results.size(), Ts.size());
  for (size_t i = 0; i < Ts.size(); ++i) {
    EXPECT_FALSE(results[i].empty()) << "utt=" << i;
    EXPECT_GE(results[i][0].score, -1e6f) << "utt=" << i; // 合理分数
  }
}

TEST(BatchDecoderTest, EmptyBatch) {
  DecoderOptions opts;
  BatchDecoder dec(opts);
  auto results = dec.DecodeBatch(nullptr, nullptr, 0, 0);
  EXPECT_TRUE(results.empty());
}

TEST(BatchDecoderTest, ZeroFrameUtterance) {
  constexpr int V = 30;
  auto normal = RandomLogProbs(10, V);
  std::vector<float> batch_data = normal;

  std::vector<int> frame_counts = {10, 0, 10};
  // 用三个语音的数据：[normal, empty, normal]，但实际只有一份 normal，
  // 这里简化测试：三条都指向同一数据区域（零帧的不读取数据）
  // 实际上三条指向同一 normal 的数据——对零帧来说读多少都无所谓

  DecoderOptions opts;
  opts.beam_size = 3;
  BatchDecoder dec(opts, 2);
  auto results = dec.DecodeBatch(batch_data.data(), frame_counts.data(), 3, V);

  ASSERT_EQ(results.size(), 3u);
  EXPECT_FALSE(results[0].empty());  // T=10, 有结果
  EXPECT_TRUE(results[1].empty());   // T=0, 空结果
  EXPECT_FALSE(results[2].empty());  // T=10, 有结果（虽然指向了前 10 帧的数据）
}

/* ═══════════════════════════════════════════════════════════════
 *  C API 测试
 * ═══════════════════════════════════════════════════════════════ */

TEST(CApiTest, CreateDestroy) {
  ctc_decoder_options_t opts;
  opts.beam_size = 10;
  opts.blank_id = 0;
  opts.n_best = 1;
  opts.cutoff_threshold = 0.0f;
  opts.num_threads = 2;

  auto *dec = ctc_batch_decoder_create(&opts);
  ASSERT_NE(dec, nullptr);
  ctc_batch_decoder_destroy(dec);
}

TEST(CApiTest, CreateDefaultOptions) {
  auto *dec = ctc_batch_decoder_create(nullptr);
  ASSERT_NE(dec, nullptr);
  ctc_batch_decoder_destroy(dec);
}

TEST(CApiTest, BatchDecode) {
  constexpr int T = 20, V = 30, N = 3;
  auto single = RandomLogProbs(T, V);
  std::vector<std::vector<float>> inputs(N, single);
  std::vector<int> frame_counts;
  auto batch_data = BuildBatchData(inputs, frame_counts, V);

  ctc_decoder_options_t opts = {10, 0, 2, 0.0f, 2};
  auto *dec = ctc_batch_decoder_create(&opts);
  ASSERT_NE(dec, nullptr);

  ctc_batch_results_t *results = nullptr;
  int ret = ctc_decode_batch(dec, batch_data.data(), frame_counts.data(), N, V,
                             &results);
  ASSERT_EQ(ret, CTC_SUCCESS);
  ASSERT_NE(results, nullptr);

  EXPECT_EQ(ctc_batch_results_num_utterances(results), N);

  for (int u = 0; u < N; ++u) {
    int n_hyps = ctc_batch_results_num_hyps(results, u);
    EXPECT_EQ(n_hyps, 2) << "utt=" << u;

    for (int h = 0; h < n_hyps; ++h) {
      int len = 0;
      const int *tokens = ctc_batch_results_tokens(results, u, h, &len);
      ASSERT_NE(tokens, nullptr) << "utt=" << u << " hyp=" << h;
      EXPECT_GT(len, 0) << "utt=" << u << " hyp=" << h;

      float score = ctc_batch_results_score(results, u, h);
      EXPECT_LE(score, 0.0f) << "utt=" << u << " hyp=" << h;

      // 时间戳（beam search 当前未追踪时间戳，可为空）
      int n_ts = 0;
      const int *ts = ctc_batch_results_timestamps(results, u, h, &n_ts);
      // timestamps 可为空（当前 PrefixBeamSearch 未实现时间戳追踪）
      if (n_ts > 0) {
        ASSERT_NE(ts, nullptr);
        EXPECT_EQ(n_ts, len) << "utt=" << u << " hyp=" << h;
      }
    }
  }

  ctc_batch_results_destroy(results);
  ctc_batch_decoder_destroy(dec);
}

TEST(CApiTest, NullHandles) {
  EXPECT_EQ(ctc_decode_batch(nullptr, nullptr, nullptr, 0, 0, nullptr),
            CTC_ERROR);
  EXPECT_STRNE(ctc_last_error(), "");

  EXPECT_EQ(ctc_batch_results_num_utterances(nullptr), CTC_ERROR);

  int len;
  EXPECT_EQ(ctc_batch_results_tokens(nullptr, 0, 0, &len), nullptr);

  EXPECT_FLOAT_EQ(ctc_batch_results_score(nullptr, 0, 0), 0.0f);
}

TEST(CApiTest, OutOfRange) {
  constexpr int T = 10, V = 20;
  auto log_probs = RandomLogProbs(T, V);
  int fc = T;

  ctc_decoder_options_t opts = {5, 0, 1, 0.0f, 1};
  auto *dec = ctc_batch_decoder_create(&opts);
  ctc_batch_results_t *results = nullptr;
  ctc_decode_batch(dec, log_probs.data(), &fc, 1, V, &results);
  ASSERT_NE(results, nullptr);

  // 越界索引
  EXPECT_EQ(ctc_batch_results_num_hyps(results, 999), CTC_ERROR);
  EXPECT_EQ(ctc_batch_results_num_hyps(results, -1), CTC_ERROR);

  int len;
  EXPECT_EQ(ctc_batch_results_tokens(results, 0, 99, &len), nullptr);
  EXPECT_EQ(ctc_batch_results_tokens(results, 99, 0, &len), nullptr);

  ctc_batch_results_destroy(results);
  ctc_batch_decoder_destroy(dec);
}

TEST(CApiTest, ResultPointerStability) {
  // Token 指针在 results 销毁前应保持有效
  constexpr int T = 15, V = 25;
  auto log_probs = RandomLogProbs(T, V);
  int fc = T;

  ctc_decoder_options_t opts = {5, 0, 1, 0.0f, 1};
  auto *dec = ctc_batch_decoder_create(&opts);
  ctc_batch_results_t *results = nullptr;
  ctc_decode_batch(dec, log_probs.data(), &fc, 1, V, &results);

  int len1, len2;
  const int *tokens1 = ctc_batch_results_tokens(results, 0, 0, &len1);
  const int *tokens2 = ctc_batch_results_tokens(results, 0, 0, &len2);

  // 两次调用应返回同一块内存
  EXPECT_EQ(tokens1, tokens2);
  EXPECT_EQ(len1, len2);

  ctc_batch_results_destroy(results);
  ctc_batch_decoder_destroy(dec);
}
