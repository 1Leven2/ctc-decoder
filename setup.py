"""
CTC 解码器 Python 包

提供贪心解码和前缀束搜索，C++ 高性能实现 + pybind11 封装。

安装方式:
    pip install -e .          # 开发模式安装
    pip install .             # 正式安装

使用示例:
    import ctc_decoder
    result = ctc_decoder.greedy_decode(log_probs)
"""

from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "_ctc_decoder",
        [
            "src/greedy_decoder.cpp",
            "src/prefix_beam_search.cpp",
            "bindings/pybind.cpp",
        ],
        include_dirs=["include"],
        cxx_std=17,
        extra_compile_args=["-O3", "-march=native", "-Wall", "-Wextra"],
    ),
]

setup(
    name="ctc-decoder",
    version="0.1.0",
    author="CTC Decoder Project",
    description="高性能 CTC 解码器（贪心解码 + 前缀束搜索）",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    packages=find_packages(),
    python_requires=">=3.8",
)
