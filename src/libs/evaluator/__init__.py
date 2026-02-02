"""
Evaluator 模块 - RAG 评估抽象。

本模块包含：
- BaseEvaluator: Evaluator 抽象基类
- EvaluationQuery: 评估查询数据类
- EvaluationResult: 评估结果数据类
- EvaluationMetrics: 评估指标结果数据类
- EvaluatorError: Evaluator 错误基类
- EvaluatorFactory: Evaluator 工厂类
- CustomEvaluator: 自定义评估器实现（hit_rate, mrr, precision@k, recall@k, ndcg@k）
"""

from .base_evaluator import (
    BaseEvaluator,
    EvaluationQuery,
    EvaluationResult,
    EvaluationMetrics,
    EvaluatorError,
)
from .custom_evaluator import CustomEvaluator
from .evaluator_factory import EvaluatorFactory

__all__ = [
    "BaseEvaluator",
    "EvaluationQuery",
    "EvaluationResult",
    "EvaluationMetrics",
    "EvaluatorError",
    "CustomEvaluator",
    "EvaluatorFactory",
]
