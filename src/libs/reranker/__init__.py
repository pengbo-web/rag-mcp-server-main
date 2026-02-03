"""
Reranker 模块 - 结果重排序抽象。

本模块包含：
- BaseReranker: Reranker 抽象基类
- RerankCandidate: 重排序候选项数据类
- RerankResult: 重排序结果数据类
- RerankerError: Reranker 错误基类
- RerankerFactory: Reranker 工厂类
- NoneReranker: 不进行重排序的实现（保持原顺序）
- LLMReranker: 使用 LLM 进行重排序的实现
"""

from .base_reranker import BaseReranker, RerankCandidate, RerankResult, RerankerError
from .none_reranker import NoneReranker
from .llm_reranker import LLMReranker
from .reranker_factory import RerankerFactory

# 注册 NoneReranker
RerankerFactory.register("none", NoneReranker)
# 注册 LLMReranker
RerankerFactory.register("llm", LLMReranker)

__all__ = [
    "BaseReranker",
    "RerankCandidate",
    "RerankResult",
    "RerankerError",
    "RerankerFactory",
    "NoneReranker",
    "LLMReranker",
]
