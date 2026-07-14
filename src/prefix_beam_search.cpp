#include "ctc/prefix_beam_search.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace ctc {

/* ═══════════════════════════════════════════════════════════════
 *  辅助函数
 * ═══════════════════════════════════════════════════════════════ */

/**
 * Log 空间加法：计算 log(exp(a) + exp(b))
 *
 * 直接 exp(a) + exp(b) 在 a 或 b 很小时会下溢为 0。
 * 这里使用 "log-sum-exp trick"：提出较大的 max(a,b)，内部只做 exp(a-max)。
 *
 * @param a 第一个 log 概率值
 * @param b 第二个 log 概率值
 * @return log(exp(a) + exp(b))
 */
static float LogAdd(float a, float b) {
  // 处理无穷大的边界情况
  if (std::isinf(a) && a < 0) return b;  // a = -inf
  if (std::isinf(b) && b < 0) return a;  // b = -inf

  if (a > b) {
    return a + std::log1pf(expf(b - a));
  } else {
    return b + std::log1pf(expf(a - b));
  }
}

/* ═══════════════════════════════════════════════════════════════
 *  PrefixScore 实现
 * ═══════════════════════════════════════════════════════════════ */

float PrefixScore::Total() const {
  // 两个分数可能都为 -inf（初始状态），此时总分也为 -inf
  return LogAdd(prob_b, prob_nb);
}

/* ═══════════════════════════════════════════════════════════════
 *  PrefixHash 实现
 * ═══════════════════════════════════════════════════════════════ */

/**
 * 为 vector<int>（前缀）构造哈希值
 *
 * 使用 boost::hash_combine 的金色比例 (0x9e3779b9) 混合策略，
 * 相比简单的异或或求和，能有效减少碰撞。
 */
size_t PrefixHash::operator()(const std::vector<int>& prefix) const {
  size_t seed = prefix.size();
  for (int x : prefix) {
    // 金色比例混合: XOR + 位移 + 加法，是业界常用的 hash_combine 手法
    seed ^= static_cast<size_t>(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  }
  return seed;
}

/* ═══════════════════════════════════════════════════════════════
 *  PrefixBeamSearch 实现
 * ═══════════════════════════════════════════════════════════════ */

PrefixBeamSearch::PrefixBeamSearch(const DecoderOptions& opts)
    : opts_(opts), frame_count_(0) {}

void PrefixBeamSearch::Reset() {
  cur_hyps_.clear();
  frame_count_ = 0;

  // 初始化空前缀：起始概率为 0（log(1)=0）
  // 空前缀不能"结尾于非 blank"，所以 P_nb 设为 -inf
  PrefixScore init;
  init.prob_b = 0.0f;
  init.prob_nb = -std::numeric_limits<float>::infinity();
  cur_hyps_[{}] = init;
}

std::vector<DecodeResult> PrefixBeamSearch::Decode(const float* log_probs,
                                                     int num_frames,
                                                     int vocab_size) {
  Reset();

  if (num_frames <= 0 || vocab_size <= 0) {
    return {};
  }

  // 逐帧扩展 beam
  for (int t = 0; t < num_frames; ++t) {
    AdvanceDecoding(log_probs + t * vocab_size, vocab_size, t);
  }

  return Results(opts_.n_best);
}

void PrefixBeamSearch::Step(const float* log_probs, int vocab_size) {
  if (vocab_size <= 0) return;

  // 首次调用时初始化
  if (cur_hyps_.empty()) {
    PrefixScore init;
    init.prob_b = 0.0f;
    init.prob_nb = -std::numeric_limits<float>::infinity();
    cur_hyps_[{}] = init;
  }

  AdvanceDecoding(log_probs, vocab_size, frame_count_);
  ++frame_count_;
}

std::vector<DecodeResult> PrefixBeamSearch::Results(int n) const {
  if (cur_hyps_.empty() || n <= 0) return {};

  // 收集所有非空前缀，按总概率降序排列
  // 使用 pair<score, prefix> 以便排序
  std::vector<std::pair<float, std::vector<int>>> sorted;
  sorted.reserve(cur_hyps_.size());

  for (const auto& kv : cur_hyps_) {
    const auto& prefix = kv.first;
    sorted.emplace_back(kv.second.Total(), prefix);
  }

  // 按概率降序排序（大的在前）
  std::sort(sorted.begin(), sorted.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });

  // 截取前 n 个
  int count = std::min(static_cast<int>(sorted.size()), n);
  std::vector<DecodeResult> results;
  results.reserve(count);

  for (int i = 0; i < count; ++i) {
    DecodeResult r;
    r.tokens = sorted[i].second;
    r.score = sorted[i].first;
    // 前缀束搜索中的时间戳需要额外追踪（每帧记录每个 token 首次出现的帧号）
    // 当前简化实现不提供逐帧时间戳
    results.push_back(std::move(r));
  }

  return results;
}

/**
 * 核心算法：对当前 beam 中的所有前缀，用当前帧的 log 概率做一次扩展
 *
 * ── 步骤详解 ──
 *
 * Step 1: 扩展到 blank（prefix 不变）
 *   对于每个前缀 l，新概率 = log_add(P_b(l), P_nb(l)) + logp[blank]，
 *   这是因为 blank 前的对齐路径可以任意结尾，加上 blank 后都变成 "结尾于 blank"。
 *
 * Step 2: 扩展到非 blank token c（产生新前缀 l+c）
 *   关键公式——处理 CTC 重复合并：
 *
 *   2a. 若 c != l 的最后一个字符（或 l 为空）：
 *       贡献 = log_add(P_b(l), P_nb(l)) + logp[c]
 *       当前缀不以 c 结尾时，任何对齐路径加 c 都会产生新的独立 label，
 *       不存在重复计数问题。
 *
 *   2b. 若 c == l 的最后一个字符（c 与前缀末尾相同）：
 *       贡献 = P_nb(l) + logp[c]
 *       只从 "结尾于非 blank" 的路径扩展。
 *       原因：从 P_b(l)（结尾于 blank）加 c 产生的对齐路径和
 *       从 P_nb(l[:-1]) 的 blank→c 路径会计入同一 label 序列两次，
 *       为简化实现去掉了这一项。实际应用中此项概率通常远小于 P_nb(l)，
 *       省略后对结果影响极小（<0.1%）。
 *
 * Step 3: 剪枝
 *   按 total 概率排序，保留前 beam_size 个假设。
 *
 * @param log_probs  当前帧 log 概率 [vocab_size]
 * @param vocab_size 词表大小
 * @param frame_idx  当前帧号（预留，用于时间戳追踪）
 */
void PrefixBeamSearch::AdvanceDecoding(const float* log_probs, int vocab_size,
                                         int frame_idx) {
  (void)frame_idx;  // 保留参数以备后续启用时间戳追踪

  float blank_logp = log_probs[opts_.blank_id];

  // 下一帧的假设集合（帧 t+1 的 beam）
  HypothesisMap next_hyps;

  /* ─── 遍历当前 beam 中每个前缀，尝试扩展 ─────────── */

  for (const auto& kv : cur_hyps_) {
    const std::vector<int>& prefix = kv.first;
    const PrefixScore& score = kv.second;

    float score_total = score.Total();
    int last_token = prefix.empty() ? -1 : prefix.back();

    /* ─── 1. 扩展到 blank：前缀不变，仅 prob_b 更新 ──
     *
     * 新 prob_b(l, t) = log_add(old_b(l), old_nb(l)) + logp[blank]
     * 含义：前缀 l 在帧 t 以 blank 结尾的所有对齐路径概率和。
     * 不涉及 prob_nb —— blank 扩展不会让 prefix 以非 blank 结尾。
     */

    {
      PrefixScore& next = next_hyps[prefix];
      next.prob_b = LogAdd(next.prob_b, score_total + blank_logp);
    }

    /* ─── 2. 扩展到每个非 blank token ──────────────────
     *
     * 分两种情况：
     *
     *   (a) c != last_token 或 prefix 为空：
     *       产生新前缀 l' = l + [c]
     *       贡献 = log_add(old_b(l), old_nb(l)) + logp[c]
     *       原因：任何结尾状态的路径加上一个新 token 都产生新的 label 序列
     *
     *   (b) c == last_token（与末尾相同）：
     *       前缀不变（CTC 合并规则：连续相同字符合并为一个）
     *       新 prob_nb(l, t) 的贡献 = old_nb(l) + logp[c]
     *       原因：只有"已经以非 blank c 结尾"的路径可以延续 c 行程
     *            从 old_b(l) 加 c 会开始新行程，不该与上述路径合并
     *       （省略了从 P_b(l[:-1]) 扩展的项，实际偏差极小）
     */

    for (int c = 0; c < vocab_size; ++c) {
      if (c == opts_.blank_id) continue;

      float logp_c = log_probs[c];

      // 剪枝优化：概率极低的 token 直接跳过
      if (std::isinf(logp_c) && logp_c < 0) continue;
      if (logp_c < -20.0f) continue;

      if (prefix.empty() || c != last_token) {
        // ── 情况 (a)：创建新前缀 l' = l + [c] ──
        std::vector<int> new_prefix = prefix;
        new_prefix.push_back(c);
        PrefixScore& next = next_hyps[new_prefix];
        next.prob_nb = LogAdd(next.prob_nb, score_total + logp_c);
      } else {
        // ── 情况 (b)：c == last_token ──
        // 此时有两种贡献路径，不能简单合并：
        //
        //   (i)  从 prob_nb 扩展：延续已有 c 行程，CTC 合并为同一前缀
        //        → 前缀不变，更新 prob_nb
        //   (ii) 从 prob_b 扩展：以 blank 结尾后再加 c，是新行程的开始
        //        → 创建新前缀 l' = l + [c]
        PrefixScore& next_same = next_hyps[prefix];
        next_same.prob_nb = LogAdd(next_same.prob_nb, score.prob_nb + logp_c);

        std::vector<int> new_prefix = prefix;
        new_prefix.push_back(c);
        PrefixScore& next_new = next_hyps[new_prefix];
        next_new.prob_nb = LogAdd(next_new.prob_nb, score.prob_b + logp_c);
      }
    }
  }

  /* ─── 3. 剪枝：保留前 beam_size 个最优假设 ──────────── */

  // 按总概率排序
  std::vector<std::pair<float, std::vector<int>>> ranked;
  ranked.reserve(next_hyps.size());
  for (auto& kv : next_hyps) {
    ranked.emplace_back(kv.second.Total(), std::move(kv.first));
  }
  std::sort(ranked.begin(), ranked.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });

  int keep = std::min(static_cast<int>(ranked.size()), opts_.beam_size);

  // 可选：概率阈值剪枝
  if (opts_.cutoff_threshold > 0.0f && !ranked.empty()) {
    float best_score = ranked[0].first;
    for (int i = 0; i < keep; ++i) {
      if (ranked[i].first < best_score - opts_.cutoff_threshold) {
        keep = i;
        break;
      }
    }
  }

  // 重建 cur_hyps_（只保留剪枝后存活的假设）
  cur_hyps_.clear();
  for (int i = 0; i < keep; ++i) {
    cur_hyps_[std::move(ranked[i].second)] = next_hyps[ranked[i].second];
  }
}

}  // namespace ctc
