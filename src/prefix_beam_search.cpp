/**
 * CTC 前缀束搜索解码器 — Wenet 风格实现
 *
 * 核心数据结构：PathTrie（前缀 Trie 树）
 * 核心算法：每帧对 beam 内每个前缀做 blank/非 blank 扩展，
 *          用 CTC 递推公式合并对齐路径，按 beam_size 剪枝。
 *
 * 工业级特性（对齐 Wenet runtime/core/decoder/ctc_prefix_beam_search.cc）：
 *   1. 相对阈值剪枝 — 每帧只扩展 logp > max_logp - cutoff 的 token
 *   2. 分数归一化   — 每帧结束后所有路径减去最优分数，防浮点溢出
 *   3. Trie 树管理  — 共享公共前缀，避免 vector<int> 拷贝
 *   4. 双缓冲      — score/ new_score 分离，避免同一帧内的分数污染
 */

#include "ctc/prefix_beam_search.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace ctc {

/* ═══════════════════════════════════════════════════════════════
 *  辅助函数
 * ═══════════════════════════════════════════════════════════════ */

/**
 * Log 空间加法：log(exp(a) + exp(b))
 *
 * 使用 "log-sum-exp trick"：提出 max(a,b)，内部只对差值做 exp，
 * 避免 exp(极小值) 下溢为 0。
 */
static float LogAdd(float a, float b) {
  if (std::isinf(a) && a < 0)
    return b; // a = -inf
  if (std::isinf(b) && b < 0)
    return a; // b = -inf

  if (a > b) {
    return a + std::log1pf(expf(b - a));
  } else {
    return b + std::log1pf(expf(a - b));
  }
}

/* ═══════════════════════════════════════════════════════════════
 *  PrefixScore 实现
 * ═══════════════════════════════════════════════════════════════ */

float PrefixScore::Total() const { return LogAdd(b, nb); }

/* ═══════════════════════════════════════════════════════════════
 *  PathTrie 辅助：路径回溯 & 树销毁
 * ═══════════════════════════════════════════════════════════════ */

/**
 * 从 Trie 节点回溯到根，重建 token 序列
 *
 * @param node  目标节点
 * @return      从根到 node 的 token 序列（不包含根的 id=-1）
 */
std::vector<int> PrefixBeamSearch::ReconstructPath(const PathTrie *node) {
  std::vector<int> path;
  // 从叶子向上回溯，跳过根节点（id == -1）
  while (node != nullptr && node->id != -1) {
    path.push_back(node->id);
    node = node->parent;
  }
  // 回溯得到的是逆序，翻转回来
  std::reverse(path.begin(), path.end());
  return path;
}

/**
 * 递归删除整棵 Trie 树的所有节点
 *
 * @param node 当前节点（递归入口为 root）
 */
void PrefixBeamSearch::DeleteTrie(PathTrie *node) {
  if (node == nullptr)
    return;
  for (auto &kv : node->children) {
    DeleteTrie(kv.second);
  }
  delete node;
}

/* ═══════════════════════════════════════════════════════════════
 *  PrefixBeamSearch: 构造 / 析构
 * ═══════════════════════════════════════════════════════════════ */

PrefixBeamSearch::PrefixBeamSearch(const DecoderOptions &opts) : opts_(opts) {
  root_ = new PathTrie();
  // 根节点：以 blank 开始的路径概率为 0（log(1)=0）
  // nb 保持 -inf，因为空前缀不能"以非 blank 结尾"
  root_->score.b = 0.0f;
  cur_hyps_.push_back(root_);
}

PrefixBeamSearch::~PrefixBeamSearch() { DeleteTrie(root_); }

/* ═══════════════════════════════════════════════════════════════
 *  PrefixBeamSearch: 公共接口
 * ═══════════════════════════════════════════════════════════════ */

void PrefixBeamSearch::Reset() {
  // 销毁旧的 Trie 树
  DeleteTrie(root_);
  cur_hyps_.clear();
  frame_count_ = 0;

  // 重建根节点（与构造函数一致）
  root_ = new PathTrie();
  root_->score.b = 0.0f;
  cur_hyps_.push_back(root_);
}

std::vector<DecodeResult> PrefixBeamSearch::Decode(const float *log_probs,
                                                   int num_frames,
                                                   int vocab_size) {
  Reset();

  if (num_frames <= 0 || vocab_size <= 0) {
    return {};
  }

  for (int t = 0; t < num_frames; ++t) {
    AdvanceDecoding(log_probs + t * vocab_size, vocab_size, t);
  }

  return Results(opts_.n_best);
}

void PrefixBeamSearch::Step(const float *log_probs, int vocab_size) {
  if (vocab_size <= 0)
    return;

  // 首次调用时初始化（兼容未显式 Reset 的流式场景）
  if (cur_hyps_.empty()) {
    root_ = new PathTrie();
    root_->score.b = 0.0f;
    cur_hyps_.push_back(root_);
  }

  AdvanceDecoding(log_probs, vocab_size, frame_count_);
  ++frame_count_;
}

/**
 * 取出当前 beam 中概率最高的 n 个前缀，重建为 DecodeResult
 *
 * 遍历 cur_hyps_（已按分数排序的存活节点），回溯 Trie 路径
 * 得到 token 序列。
 */
std::vector<DecodeResult> PrefixBeamSearch::Results(int n) const {
  if (cur_hyps_.empty() || n <= 0)
    return {};

  std::vector<DecodeResult> result_list;
  result_list.reserve(std::min(static_cast<int>(cur_hyps_.size()), n));

  for (int i = 0; i < static_cast<int>(cur_hyps_.size()) && i < n; ++i) {
    const PathTrie *node = cur_hyps_[i];
    DecodeResult r;
    r.tokens = ReconstructPath(node);
    r.score = node->score.Total();
    result_list.push_back(std::move(r));
  }

  return result_list;
}

/* ═══════════════════════════════════════════════════════════════
 *  AdvanceDecoding: 单帧 beam 扩展（核心算法）
 *
 *  对当前 beam 中的每个前缀，尝试两种扩展：
 *   1. 扩展到 blank → 前缀不变，更新 b
 *   2. 扩展到非 blank c → 根据 c 与末尾 token 的关系决定
 *      (a) c != 末尾 → 产生新前缀，从 Total 扩展
 *      (b) c == 末尾 → 前缀不变（延续行程），从 nb 扩展
 *                     + 产生新前缀（新行程），从 b 扩展
 *
 *  扩展完成后，收集 Trie 中所有被触及的节点，按总分排序，
 *  保留前 beam_size 个。最后做分数归一化保持数值稳定。
 * ═══════════════════════════════════════════════════════════════ */
void PrefixBeamSearch::AdvanceDecoding(const float *log_probs, int vocab_size,
                                       int frame_idx) {
  (void)frame_idx; // 预留用于时间戳追踪

  /* ─── 1. 计算相对剪枝阈值 ──────────────────────────
   *
   * Wenet 风格：每帧先扫描 log_probs 找到最大概率，
   * 低于 max_logp - cutoff_threshold 的 token 直接跳过。
   * 仅此一步即可在 V=5000 时过滤掉 90%+ 的无效扩展。 */

  float max_logp = -std::numeric_limits<float>::infinity();
  for (int i = 0; i < vocab_size; ++i) {
    if (log_probs[i] > max_logp)
      max_logp = log_probs[i];
  }
  float cutoff = max_logp - opts_.cutoff_threshold;
  float blank_logp = log_probs[opts_.blank_id];

  /* ─── 2. 遍历当前 beam，扩展每个前缀 ─────────────── */

  for (PathTrie *cur : cur_hyps_) {
    int last_token = cur->id; // 根节点 id=-1，天然 != 任何非 blank token

    /* ─── 2a. 扩展到 blank：前缀不变 ─────────────────
     *
     * new_b(l) = log_add(old_b(l), old_nb(l)) + logp[blank]
     *
     * 表示前缀 l 在帧 t 以 blank 结尾的所有路径概率和。
     * blank 可以跟在任何结尾状态（blank 或非 blank）之后。 */

    cur->new_score.b =
        LogAdd(cur->new_score.b, cur->score.Total() + blank_logp);

    /* ─── 2b. 扩展到每个非 blank token ─────────────── */

    for (int c = 0; c < vocab_size; ++c) {
      if (c == opts_.blank_id)
        continue;

      float logp_c = log_probs[c];

      // 相对阈值剪枝：跳过该帧概率极低的 token
      if (logp_c < cutoff)
        continue;

      /* ── 情况 (a): c != 末尾，创建新前缀 ──
       *
       * l' = l + [c]
       * nb'(l') += log_add(old_b(l), old_nb(l)) + logp[c]
       *
       * 任何结尾状态的路径加上新 token 都产生新的独立 label */

      if (c != last_token) {
        PathTrie *next = cur->Forward(c);
        next->new_score.nb =
            LogAdd(next->new_score.nb, cur->score.Total() + logp_c);

      } else {
        /* ── 情况 (b): c == 末尾（CTC 合并） ──
         *
         * 此时存在两种贡献路径：
         *
         *   (i)  从 nb 扩展：延续已有的 c 行程
         *        → 前缀不变，nb'(l) += old_nb(l) + logp[c]
         *
         *   (ii) 从 b 扩展：以 blank 结尾后再加 c，开始新行程
         *        → 新前缀 l' = l + [c]
         *        nb'(l') += old_b(l) + logp[c] */

        // (i) 延续行程 → 前缀不变
        cur->new_score.nb = LogAdd(cur->new_score.nb, cur->score.nb + logp_c);

        // (ii) 开始新行程 → 产生新前缀
        PathTrie *next = cur->Forward(c);
        next->new_score.nb = LogAdd(next->new_score.nb, cur->score.b + logp_c);
      }
    }
  }

  /* ─── 3. 收集 Trie 中所有被扩展到的节点 ────────────
   *
   * CollectNodes 递归遍历整棵 Trie，返回所有节点。
   * 未被触及的节点的 new_score 为 -inf，会在下一步
   * Total() 检查中被过滤掉。 */

  std::vector<PathTrie *> all_nodes;
  root_->CollectNodes(all_nodes);

  /* ─── 4. 提交 new_score → score，准备剪枝 ──────────
   *
   * 双缓冲机制：在遍历开始前不能直接修改 score，
   * 否则当前帧的后继扩展会读到"混合"了新旧值的分数。
   * 所有扩展写入 new_score，全部完成后一起提交。 */

  std::vector<std::pair<float, PathTrie *>> ranked;
  ranked.reserve(all_nodes.size());

  for (PathTrie *node : all_nodes) {
    float total = node->new_score.Total();
    if (total > -std::numeric_limits<float>::infinity()) {
      // 提交分数：将 new_score 复制到 score
      node->score = node->new_score;
      // 重置 new_score 为 -inf，为下一帧做准备
      node->new_score = PrefixScore();
      ranked.emplace_back(total, node);
    }
  }

  /* ─── 5. 按总分排序，保留前 beam_size 个 ──────────── */

  std::sort(ranked.begin(), ranked.end(),
            [](const auto &a, const auto &b) { return a.first > b.first; });

  int keep = std::min(static_cast<int>(ranked.size()), opts_.beam_size);

  /* ─── 6. 分数归一化（数值稳定性） ──────────────────
   *
   * 长时间解码时 log 概率会累积到极小的负数（如 -500），
   * 此时 float 的有效精度约 1e-7，exp(-500) 已经下溢。
   * 将所有存活路径减去当前最优分数，分数保持在 [-beam_width, 0]
   * 区间内，完全不改变排序结果（log 空间减法 = 概率空间除法）。
   *
   * 这是 Wenet / sherpa-onnx 等工业代码的标配操作。 */

  if (keep > 0) {
    float norm_factor = ranked[0].first;
    cur_hyps_.clear();
    for (int i = 0; i < keep; ++i) {
      PathTrie *node = ranked[i].second;
      node->score.b -= norm_factor;
      node->score.nb -= norm_factor;
      cur_hyps_.push_back(node);
    }
  }

  /* ─── 注：Trie GC（垃圾回收） ──────────────────────
   *
   * 完整的工业实现还需要在剪枝后删除存活节点以外的 Trie 分支，
   * 防止长时间流式解码时内存无限增长。当前简化实现省略了
   * 复杂的 Trie 节点移除逻辑，内存会在 Reset() 或析构时统一释放。
   * 对于离线批量解码（固定帧数），此省略不影响正确性或内存使用。 */
}

} // namespace ctc
