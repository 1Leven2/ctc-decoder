/**
 * CTC 解码器性能基准测试
 *
 * 对比贪心解码和前缀束搜索在不同 T×V 组合下的吞吐量。
 * 可与纯 Python 实现（numpy argmax / 手动 beam search）对比加速比。
 */

#include <benchmark/benchmark.h>

#include <cmath>
#include <random>
#include <vector>

#include "ctc/greedy_decoder.h"
#include "ctc/prefix_beam_search.h"

using namespace ctc;

/**
 * 生成随机 log 概率矩阵
 *
 * @param num_frames 帧数 T
 * @param vocab_size 词表大小 V
 * @param blank_id   blank token 索引
 * @return 行优先展平的 log 概率数组
 */
static std::vector<float> GenerateRandomLogProbs(int num_frames, int vocab_size,
                                                   int blank_id = 0) {
  std::vector<float> data(num_frames * vocab_size);
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);

  for (int t = 0; t < num_frames; ++t) {
    // 生成随机概率值
    float sum = 0.0f;
    for (int v = 0; v < vocab_size; ++v) {
      data[t * vocab_size + v] = dist(rng);
      sum += data[t * vocab_size + v];
    }
    // 归一化并转为 log 概率
    for (int v = 0; v < vocab_size; ++v) {
      data[t * vocab_size + v] = std::log(data[t * vocab_size + v] / sum);
    }
  }

  return data;
}

/* ═══════════════════════════════════════════════════════════════
 *  贪心解码基准
 * ═══════════════════════════════════════════════════════════════ */

/**
 * 固定 V=5000 的中文词表大小，变化帧数 T
 *
 * 典型中文 ASR 场景：V≈5000（常用中文字），T=100~500（2~10秒音频）
 */
static void BM_Greedy_T100_V5000(benchmark::State& state) {
  constexpr int T = 100;
  constexpr int V = 5000;
  auto log_probs = GenerateRandomLogProbs(T, V);
  GreedyDecoder decoder(0);

  for (auto _ : state) {
    auto results = decoder.Decode(log_probs.data(), T, V);
    benchmark::DoNotOptimize(results);
  }

  state.SetItemsProcessed(state.iterations() * T);
  state.SetBytesProcessed(state.iterations() * T * V * sizeof(float));
}
BENCHMARK(BM_Greedy_T100_V5000);

static void BM_Greedy_T300_V5000(benchmark::State& state) {
  constexpr int T = 300;
  constexpr int V = 5000;
  auto log_probs = GenerateRandomLogProbs(T, V);
  GreedyDecoder decoder(0);

  for (auto _ : state) {
    auto results = decoder.Decode(log_probs.data(), T, V);
    benchmark::DoNotOptimize(results);
  }

  state.SetItemsProcessed(state.iterations() * T);
  state.SetBytesProcessed(state.iterations() * T * V * sizeof(float));
}
BENCHMARK(BM_Greedy_T300_V5000);

static void BM_Greedy_T500_V5000(benchmark::State& state) {
  constexpr int T = 500;
  constexpr int V = 5000;
  auto log_probs = GenerateRandomLogProbs(T, V);
  GreedyDecoder decoder(0);

  for (auto _ : state) {
    auto results = decoder.Decode(log_probs.data(), T, V);
    benchmark::DoNotOptimize(results);
  }

  state.SetItemsProcessed(state.iterations() * T);
  state.SetBytesProcessed(state.iterations() * T * V * sizeof(float));
}
BENCHMARK(BM_Greedy_T500_V5000);

/* ═══════════════════════════════════════════════════════════════
 *  前缀束搜索基准（不同 beam_size）
 * ═══════════════════════════════════════════════════════════════ */

/**
 * 固定 T=100, V=128，变化 beam_size
 *
 * 较小 V 使 beam search 更可控，方便对比不同 beam 的耗时
 */
static void BM_Beam_T100_V128_Beam5(benchmark::State& state) {
  constexpr int T = 100;
  constexpr int V = 128;
  auto log_probs = GenerateRandomLogProbs(T, V);
  DecoderOptions opts;
  opts.beam_size = 5;

  for (auto _ : state) {
    PrefixBeamSearch decoder(opts);
    auto results = decoder.Decode(log_probs.data(), T, V);
    benchmark::DoNotOptimize(results);
  }

  state.SetItemsProcessed(state.iterations() * T);
}
BENCHMARK(BM_Beam_T100_V128_Beam5);

static void BM_Beam_T100_V128_Beam10(benchmark::State& state) {
  constexpr int T = 100;
  constexpr int V = 128;
  auto log_probs = GenerateRandomLogProbs(T, V);
  DecoderOptions opts;
  opts.beam_size = 10;

  for (auto _ : state) {
    PrefixBeamSearch decoder(opts);
    auto results = decoder.Decode(log_probs.data(), T, V);
    benchmark::DoNotOptimize(results);
  }

  state.SetItemsProcessed(state.iterations() * T);
}
BENCHMARK(BM_Beam_T100_V128_Beam10);

static void BM_Beam_T100_V128_Beam20(benchmark::State& state) {
  constexpr int T = 100;
  constexpr int V = 128;
  auto log_probs = GenerateRandomLogProbs(T, V);
  DecoderOptions opts;
  opts.beam_size = 20;

  for (auto _ : state) {
    PrefixBeamSearch decoder(opts);
    auto results = decoder.Decode(log_probs.data(), T, V);
    benchmark::DoNotOptimize(results);
  }

  state.SetItemsProcessed(state.iterations() * T);
}
BENCHMARK(BM_Beam_T100_V128_Beam20);

BENCHMARK_MAIN();
