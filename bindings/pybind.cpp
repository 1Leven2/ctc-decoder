/**
 * CTC 解码器的 Python 封装（pybind11）
 *
 * 将 C++ 高性能解码器暴露为 Python 可调用的模块，
 * 输入 numpy 数组 (float32)，输出包含 token 序列和分数的 Python 对象。
 *
 * 使用方式：
 *   import ctc_decoder
 *   result = ctc_decoder.greedy_decode(log_probs)       # 贪心
 *   results = ctc_decoder.beam_search(log_probs, beam=10) # 束搜索
 */

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <vector>

#include "ctc/greedy_decoder.h"
#include "ctc/prefix_beam_search.h"
#include "ctc/types.h"

namespace py = pybind11;

/* ═══════════════════════════════════════════════════════════════
 *  辅助：从 C++ DecodeResult 构造 Python dict
 * ═══════════════════════════════════════════════════════════════ */

/**
 * 将 C++ DecodeResult 转换为 Python dict
 *
 * Python 侧看到的字段：
 *   tokens     — list[int]，解码的 token ID 序列（已去 blank/合并）
 *   score      — float，该序列的 CTC log 概率
 *   timestamps — list[int]，每个 token 对应的帧号
 */
static py::dict ResultToDict(const ctc::DecodeResult& result) {
  py::dict d;
  d["tokens"] = result.tokens;
  d["score"] = result.score;
  d["timestamps"] = result.timestamps;
  return d;
}

/* ═══════════════════════════════════════════════════════════════
 *  非流式接口：一次性输入全部帧，返回解码结果
 * ═══════════════════════════════════════════════════════════════ */

/**
 * 贪心解码（非流式）
 *
 * @param log_probs  numpy 二维数组 [num_frames, vocab_size]，dtype=float32
 *                   必须是 log_softmax 后的对数概率
 * @return           dict: {"tokens": [...], "score": float, "timestamps": [...]}
 */
py::dict GreedyDecode(py::array_t<float, py::array::c_style | py::array::forcecast> log_probs) {
  // 获取数组基本信息
  auto buf = log_probs.request();
  if (buf.ndim != 2) {
    throw std::runtime_error("log_probs 必须是二维数组 [T, V]");
  }

  int num_frames = static_cast<int>(buf.shape[0]);
  int vocab_size = static_cast<int>(buf.shape[1]);
  const float* data = static_cast<const float*>(buf.ptr);

  ctc::GreedyDecoder decoder(/*blank_id=*/0);

  std::vector<ctc::DecodeResult> results;
  {
    // 释放 GIL，允许 Python 其他线程并行运行
    // GreedyDecoder::Decode 是纯 C++ 计算，不涉及 Python 对象
    py::gil_scoped_release release;
    results = decoder.Decode(data, num_frames, vocab_size);
  }  // GIL 在此处自动恢复（RAII）

  if (results.empty()) return py::dict();
  return ResultToDict(results[0]);
}

/**
 * 前缀束搜索解码（非流式）
 *
 * @param log_probs  numpy 二维数组 [num_frames, vocab_size]，dtype=float32
 * @param beam_size  beam 宽度（默认 10），越大越准确但越慢
 * @param n_best     返回的最佳结果数量（默认 1）
 * @return           list[dict]，每个 dict 同贪心解码的返回格式，
 *                   按概率降序排列
 */
py::list BeamSearch(py::array_t<float, py::array::c_style | py::array::forcecast> log_probs,
                     int beam_size = 10, int n_best = 1) {
  auto buf = log_probs.request();
  if (buf.ndim != 2) {
    throw std::runtime_error("log_probs 必须是二维数组 [T, V]");
  }

  int num_frames = static_cast<int>(buf.shape[0]);
  int vocab_size = static_cast<int>(buf.shape[1]);
  const float* data = static_cast<const float*>(buf.ptr);

  ctc::DecoderOptions opts;
  opts.beam_size = beam_size;
  opts.n_best = n_best;
  opts.blank_id = 0;

  ctc::PrefixBeamSearch decoder(opts);

  std::vector<ctc::DecodeResult> results;
  {
    py::gil_scoped_release release;
    results = decoder.Decode(data, num_frames, vocab_size);
  }

  py::list out;
  for (const auto& r : results) {
    out.append(ResultToDict(r));
  }
  return out;
}

/* ═══════════════════════════════════════════════════════════════
 *  流式解码类封装
 * ═══════════════════════════════════════════════════════════════ */

/**
 * Python 可见的贪心解码器（流式）
 *
 * 使用示例：
 *   dec = ctc_decoder.GreedyDecoder()
 *   for frame_logp in stream:
 *       dec.step(frame_logp)          # 逐帧喂入
 *   result = dec.result()             # 获取当前最佳
 *   dec.reset()                       # 重置以解码下一段
 */
class PyGreedyDecoder {
 public:
  PyGreedyDecoder() : decoder_(0) {}

  void Step(py::array_t<float, py::array::c_style | py::array::forcecast> log_probs) {
    auto buf = log_probs.request();
    if (buf.ndim != 1) {
      throw std::runtime_error("step() 输入须为一维向量 [V]");
    }
    int vocab_size = static_cast<int>(buf.shape[0]);
    const float* data = static_cast<const float*>(buf.ptr);

    {
      py::gil_scoped_release release;
      decoder_.Step(data, vocab_size);
    }
  }

  py::dict Result() {
    auto results = decoder_.Results(1);
    if (results.empty()) return py::dict();
    return ResultToDict(results[0]);
  }

  void Reset() { decoder_.Reset(); }

 private:
  ctc::GreedyDecoder decoder_;
};

/**
 * Python 可见的前缀束搜索解码器（流式）
 *
 * 使用示例：
 *   dec = ctc_decoder.PrefixBeamSearchDecoder(beam_size=10)
 *   for frame_logp in stream:
 *       dec.step(frame_logp)
 *   results = dec.results(n_best=3)
 *   dec.reset()
 */
class PyPrefixBeamSearch {
 public:
  explicit PyPrefixBeamSearch(int beam_size = 10, int n_best = 1)
      : opts_{}, decoder_(opts_) {
    opts_.beam_size = beam_size;
    opts_.n_best = n_best;
    opts_.blank_id = 0;
    decoder_ = ctc::PrefixBeamSearch(opts_);
  }

  void Step(py::array_t<float, py::array::c_style | py::array::forcecast> log_probs) {
    auto buf = log_probs.request();
    if (buf.ndim != 1) {
      throw std::runtime_error("step() 输入须为一维向量 [V]");
    }
    int vocab_size = static_cast<int>(buf.shape[0]);
    const float* data = static_cast<const float*>(buf.ptr);

    {
      py::gil_scoped_release release;
      decoder_.Step(data, vocab_size);
    }
  }

  py::list Results(int n = 1) {
    auto cpp_results = decoder_.Results(n);
    py::list out;
    for (const auto& r : cpp_results) {
      out.append(ResultToDict(r));
    }
    return out;
  }

  void Reset() { decoder_.Reset(); }

 private:
  ctc::DecoderOptions opts_;
  ctc::PrefixBeamSearch decoder_;
};

/* ═══════════════════════════════════════════════════════════════
 *  模块注册
 * ═══════════════════════════════════════════════════════════════ */

PYBIND11_MODULE(_ctc_decoder, m) {
  m.doc() = R"pbdoc(
    CTC 解码器 (C++ 高性能实现)

    提供贪心解码和前缀束搜索两种算法，输入声学模型的 log_softmax 概率矩阵，
    输出解码的 token 序列。

    核心函数:
      greedy_decode(log_probs)  → dict  贪心解码（最快）
      beam_search(log_probs)    → list  前缀束搜索（更准确）

    流式解码类:
      GreedyDecoder        逐帧贪心解码
      PrefixBeamSearchDecoder  逐帧前缀束搜索
  )pbdoc";

  /* ─── 非流式函数 ───────────────────────────────── */

  m.def("greedy_decode", &GreedyDecode,
        py::arg("log_probs"),
        R"pbdoc(
          贪心解码（非流式）

          每一帧取概率最大的 token，然后用 CTC 规则去除 blank 并合并相邻重复。

          Args:
              log_probs: numpy 二维数组 [T, V], dtype=float32,
                         log_softmax 后的对数概率矩阵

          Returns:
              dict: {"tokens": [int, ...], "score": float, "timestamps": [int, ...]}

          Example:
              >>> log_probs = model.encode(audio)  # [T, V]
              >>> result = ctc_decoder.greedy_decode(log_probs)
              >>> print(result["tokens"])  # [10, 45, 67, ...]
        )pbdoc");

  m.def("beam_search", &BeamSearch,
        py::arg("log_probs"), py::arg("beam_size") = 10, py::arg("n_best") = 1,
        R"pbdoc(
          前缀束搜索解码（非流式）

          在每一步维护 beam_size 个最优前缀假设，有效纠正贪心解码的局部最优陷阱。

          Args:
              log_probs: numpy 二维数组 [T, V], dtype=float32
              beam_size: beam 宽度，默认 10。越大越准确，耗时正比增长
              n_best:    返回的最佳结果数，默认 1

          Returns:
              list[dict]: 按概率降序排列的结果列表，
                         每个 dict 格式同 greedy_decode 返回值

          Example:
              >>> results = ctc_decoder.beam_search(log_probs, beam_size=20, n_best=3)
              >>> for r in results:
              ...     print(r["score"], r["tokens"])
        )pbdoc");

  /* ─── 流式解码类 ────────────────────────────────── */

  py::class_<PyGreedyDecoder>(m, "GreedyDecoder")
      .def(py::init<>())
      .def("step", &PyGreedyDecoder::Step, py::arg("log_probs"),
           "输入一帧 log 概率向量 [V]，更新内部解码状态")
      .def("result", &PyGreedyDecoder::Result,
           "获取当前累积的最优解码结果")
      .def("reset", &PyGreedyDecoder::Reset,
           "重置解码器状态，准备解码下一段音频");

  py::class_<PyPrefixBeamSearch>(m, "PrefixBeamSearchDecoder")
      .def(py::init<int, int>(), py::arg("beam_size") = 10, py::arg("n_best") = 1)
      .def("step", &PyPrefixBeamSearch::Step, py::arg("log_probs"),
           "输入一帧 log 概率向量 [V]，执行一次 beam 扩展")
      .def("results", &PyPrefixBeamSearch::Results, py::arg("n") = 1,
           "获取当前 beam 中最佳的 n 个解码结果")
      .def("reset", &PyPrefixBeamSearch::Reset,
           "重置解码器状态");
}
