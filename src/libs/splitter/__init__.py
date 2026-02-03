"""
Splitter 库模块。

提供统一的 Splitter 抽象接口和工厂，支持多种切分策略。
"""

from .base_splitter import BaseSplitter, TextChunk, SplitterError
from .splitter_factory import SplitterFactory
from .recursive_splitter import RecursiveSplitter

# 注册提供商到工厂
SplitterFactory.register("recursive", RecursiveSplitter)

__all__ = [
    "BaseSplitter",
    "TextChunk",
    "SplitterError",
    "SplitterFactory",
    "RecursiveSplitter",
]
