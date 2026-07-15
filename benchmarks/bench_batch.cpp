/**
 * 批量束搜索性能基准
 *
 * 对比不同线程数、不同批量大小下的吞吐量。
 */

#include <benchmark/benchmark.h>

#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#include "ctc/batch_decoder.h"

using namespace ctc;

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

/** 构造批量数据：N 条相同 T×V 的语音 */
static std::pair<std::vector<float>, std::vector<int>>
MakeBatch(int N, int T, int V) {
  auto single = RandomLogProbs(T, V, T + N);
  std::vector<float> batch(N * T * V);
  for (int i = 0; i < N; ++i) {
    std::memcpy(batch.data() + i * T * V, single.data(),
                T * V * sizeof(float));
  }
  std::vector<int> frame_counts(N, T);
  return {batch, frame_counts};
}

/* ─── 不同批量大小 × 不同线程数 ──────────────────── */

static void BM_Batch_N8_T50_V128(benchmark::State &state) {
  constexpr int N = 8, T = 50, V = 128;
  auto [data, fc] = MakeBatch(N, T, V);

  DecoderOptions opts;
  opts.beam_size = 10;
  int threads = static_cast<int>(state.range(0));

  for (auto _ : state) {
    BatchDecoder dec(opts, threads);
    auto results = dec.DecodeBatch(data.data(), fc.data(), N, V);
    benchmark::DoNotOptimize(results);
  }
  state.SetItemsProcessed(state.iterations() * N * T);
}
BENCHMARK(BM_Batch_N8_T50_V128)->Arg(1)->Arg(4)->Arg(8);

static void BM_Batch_N16_T50_V128(benchmark::State &state) {
  constexpr int N = 16, T = 50, V = 128;
  auto [data, fc] = MakeBatch(N, T, V);

  DecoderOptions opts;
  opts.beam_size = 10;
  int threads = static_cast<int>(state.range(0));

  for (auto _ : state) {
    BatchDecoder dec(opts, threads);
    auto results = dec.DecodeBatch(data.data(), fc.data(), N, V);
    benchmark::DoNotOptimize(results);
  }
  state.SetItemsProcessed(state.iterations() * N * T);
}
BENCHMARK(BM_Batch_N16_T50_V128)->Arg(1)->Arg(4)->Arg(8)->Arg(16);

static void BM_Batch_N32_T50_V128(benchmark::State &state) {
  constexpr int N = 32, T = 50, V = 128;
  auto [data, fc] = MakeBatch(N, T, V);

  DecoderOptions opts;
  opts.beam_size = 10;
  int threads = static_cast<int>(state.range(0));

  for (auto _ : state) {
    BatchDecoder dec(opts, threads);
    auto results = dec.DecodeBatch(data.data(), fc.data(), N, V);
    benchmark::DoNotOptimize(results);
  }
  state.SetItemsProcessed(state.iterations() * N * T);
}
BENCHMARK(BM_Batch_N32_T50_V128)->Arg(1)->Arg(4)->Arg(8)->Arg(16);

BENCHMARK_MAIN();
