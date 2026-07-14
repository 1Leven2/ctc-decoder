#ifndef CTC_PREFIX_BEAM_SEARCH_H_
#define CTC_PREFIX_BEAM_SEARCH_H_

#include <limits>
#include <unordered_map>
#include <vector>

#include "decoder.h"

namespace ctc {

/* ═══════════════════════════════════════════════════════════════
 *  PrefixScore: CTC 前缀的双通道概率分数
 *
 *  CTC 的核心问题：同一条 label 序列对应无数条对齐路径
 * （blank 可插入任意位置）。通过维护"以 blank 结尾"和
 * "以非 blank 结尾"两个分数，用动态规划在 log 空间合并
 * 所有路径，避免指数级枚举。
 *
 *  递推公式：
 *   扩展到 blank（前缀不变）：
 *     b'(l) = log_add(b(l), nb(l)) + logp[blank]
 *
 *   扩展到非 blank 字符 c：
 *     - c != l 的最后一个字符：新前缀 l' = l + [c]
 *         nb'(l') = log_add(b(l), nb(l)) + logp[c]
 *     - c == l 的最后一个字符（CTC 合并）：
 *         (i)  从 nb 扩展：延续已有的 c 行程 → 前缀不变
 *               nb'(l) += nb(l) + logp[c]
 *         (ii) 从 b 扩展：blank 后开始新行程 → 新前缀 l' = l + [c]
 *               nb'(l') += b(l) + logp[c]
 * ═══════════════════════════════════════════════════════════════ */
struct PrefixScore {
  float b = -std::numeric_limits<float>::infinity(); // 以 blank 结尾的路径
  float nb = -std::numeric_limits<float>::infinity(); // 以非 blank 结尾的路径

  /** 总概率 = log(exp(b) + exp(nb)) */
  float Total() const;
};

/* ═══════════════════════════════════════════════════════════════
 *  PathTrie: 前缀 Trie 树节点
 *
 *  Wenet 管理 beam 中所有前缀的核心数据结构。
 *  相比 unordered_map<vector<int>, PrefixScore> 方案，Trie 树
 *  天然共享公共前缀，减少内存分配和 key 拷贝开销。
 *
 *  每个节点代表一个前缀（从根到该节点的路径），
 *  存储该前缀在当前帧的 CTC 分数。
 *
 *  Wenet 源码参考：
 *   runtime/core/decoder/ctc_prefix_beam_search.h (PathTrie)
 * ═══════════════════════════════════════════════════════════════ */
class PathTrie {
public:
  int id = -1; // 当前节点对应的 token id，根节点为 -1
  PathTrie *parent = nullptr;
  std::unordered_map<int, PathTrie *> children;

  PrefixScore score;     // 当前帧的累积分数
  PrefixScore new_score; // 下一帧计算中的暂存分数

  /**
   * 向下扩展一个 token，若子节点不存在则创建
   * @param token 扩展的 token id
   * @return 子节点指针
   */
  PathTrie *Forward(int token) {
    auto it = children.find(token);
    if (it == children.end()) {
      PathTrie *node = new PathTrie();
      node->id = token;
      node->parent = this;
      children[token] = node;
      return node;
    }
    return it->second;
  }

  /**
   * 递归收集整棵 Trie 中的所有节点（用于剪枝排序）
   * @param nodes 输出：所有节点的指针
   */
  void CollectNodes(std::vector<PathTrie *> &nodes) {
    nodes.push_back(this);
    for (auto &kv : children) {
      kv.second->CollectNodes(nodes);
    }
  }
};

/* ═══════════════════════════════════════════════════════════════
 *  PrefixBeamSearch: CTC 前缀束搜索解码器（Wenet 风格实现）
 *
 *  与贪心解码不同，前缀束搜索在每一步维护 beam_size 个最优
 *  文本前缀假设，通过搜索多条路径来纠正局部最优陷阱。
 *
 *  工业级特性（Wenet 对齐）：
 *   — Trie 树管理前缀，共享公共前缀，节省内存与拷贝
 *   — 相对阈值剪枝：每帧只扩展 max_logp - cutoff_threshold
 *     以上的 token，跳过极小概率 token，显著加速
 *   — 分数归一化：每帧结束后所有存活路径减去最优分数，
 *     保持数值在合理区间，避免长序列累积导致浮点精度丢失
 *
 *  时间复杂度：每帧 O(beam_size × V)，总 O(T × beam_size × V)
 *  空间复杂度：O(beam_size × 最大前缀长度)
 * ═══════════════════════════════════════════════════════════════ */
class PrefixBeamSearch : public CtcDecoder {
public:
  explicit PrefixBeamSearch(const DecoderOptions &opts);
  ~PrefixBeamSearch() override;

  std::vector<DecodeResult> Decode(const float *log_probs, int num_frames,
                                   int vocab_size) override;
  void Step(const float *log_probs, int vocab_size) override;
  std::vector<DecodeResult> Results(int n) const override;
  void Reset() override;

private:
  /** 对一帧 log 概率执行一次 beam 扩展（核心算法） */
  void AdvanceDecoding(const float *log_probs, int vocab_size, int frame_idx);

  /** 从 Trie 节点回溯到根，重建 token 序列 */
  static std::vector<int> ReconstructPath(const PathTrie *node);

  /** 递归删除整棵 Trie 树（析构/Reset 时用） */
  static void DeleteTrie(PathTrie *node);

  DecoderOptions opts_;
  PathTrie *root_ = nullptr;
  std::vector<PathTrie *> cur_hyps_; // 当前 beam 中存活的节点
  int frame_count_ = 0;
};

} // namespace ctc

#endif // CTC_PREFIX_BEAM_SEARCH_H_
