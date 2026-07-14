# CTC Decoder

高性能 CTC（Connectionist Temporal Classification）解码器，C++17 实现 + pybind11 Python 封装。

提供**贪心解码**和**前缀束搜索**两种算法，输入声学模型 logits 概率矩阵，输出解码文本序列。

## 特性

- 纯 CTC 声学解码（blank=0，log_softmax 输入）
- 贪心解码：O(T·V) 极速解码，~2000x 实时
- 前缀束搜索：beam_size 可配，完整 CTC 递推公式实现
- Python 绑定：numpy 数组输入，dict/list 输出
- 流式解码：支持逐帧 Step/Results 调用
- 详尽中文注释：每个数据结构和算法步骤均有注释说明
- 单元测试全覆盖（GoogleTest，21 个测试用例）
- 性能基准（Google Benchmark）

## 安装

```bash
# CMake 构建（推荐）
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# 导入 Python 模块
cd build
python -c "import _ctc_decoder"
```

## 快速开始

```python
import numpy as np
import _ctc_decoder as ctc

# 准备 logits [T=帧数, V=词表大小]
log_probs = np.random.randn(100, 5000).astype(np.float32)
log_probs = log_probs - log_probs.max(axis=1, keepdims=True)  # log_softmax 近似

# 贪心解码（最快）
result = ctc.greedy_decode(log_probs)
print(result["tokens"])   # [10, 45, 67, ...]
print(result["score"])    # -3.14  (log 概率)

# 前缀束搜索（更准确）
results = ctc.beam_search(log_probs, beam_size=10, n_best=3)
for r in results:
    print(r["tokens"], r["score"])

# 流式解码
decoder = ctc.GreedyDecoder()
for frame in stream:
    decoder.step(frame)
result = decoder.result()
```

## 算法

### 贪心解码

每帧取 argmax，利用 CTC 规则去 blank、合并相邻重复。O(T·V)，极快。

### 前缀束搜索

维护 beam 内多个前缀假设，使用 CTC 前缀概率递推公式合并对齐路径：

- P_b(l)：前缀 l 以 blank 结尾的概率
- P_nb(l)：前缀 l 以非 blank 结尾的概率

每帧 beam 扩展后用 log-sum-exp 去重合并，按 beam_size 剪枝。

## 性能

| 算法 | 配置 | 耗时 | 实时比 |
|------|------|------|--------|
| 贪心 | T=500, V=5000 | 2.3ms | ~2000x |
| 束搜索 | T=100, V=128, beam=5 | 14.7ms | ~68x |
| 束搜索 | T=100, V=128, beam=20 | 60.2ms | ~16x |

基准测试平台：Intel 16核 3.1GHz，GCC 13.3，-O3 -march=native

## 运行测试

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=ON
make -j$(nproc)

ctest --output-on-failure        # 单元测试
./benchmarks/ctc_bench           # 性能基准
```

## 依赖

- C++17 编译器 (GCC 8+ / Clang 7+)
- CMake ≥ 3.18
- pybind11（通过 FetchContent 自动下载）
- numpy（Python 侧调用）
- GoogleTest / Google Benchmark（测试，通过 FetchContent 自动下载）

## 许可证

MIT
