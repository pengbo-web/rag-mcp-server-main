"""
Evaluator 基类与相关数据模型。

定义统一的评估接口，支持多种评估后端（自定义指标、Ragas、DeepEval 等）。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


class EvaluatorError(Exception):
    """Evaluator 相关错误的基类。"""
    pass


@dataclass
class EvaluationQuery:
    """评估查询数据。
    
    Attributes:
        query_id: 查询唯一标识符
        query_text: 查询文本
        golden_ids: 黄金标准文档 ID 列表（相关文档）
        metadata: 附加元数据（可选）
    """
    query_id: str
    query_text: str
    golden_ids: list[str]
    metadata: Optional[dict[str, Any]] = None


@dataclass
class EvaluationResult:
    """评估结果数据。
    
    Attributes:
        query_id: 查询 ID
        retrieved_ids: 检索到的文档 ID 列表
        scores: 检索分数列表（与 retrieved_ids 对应）
        metadata: 附加元数据（可选）
    """
    query_id: str
    retrieved_ids: list[str]
    scores: Optional[list[float]] = None
    metadata: Optional[dict[str, Any]] = None


@dataclass
class EvaluationMetrics:
    """评估指标结果。
    
    Attributes:
        metrics: 指标名称到值的映射（如 {"hit_rate": 0.8, "mrr": 0.65}）
        query_level_metrics: 每个查询的详细指标（可选）
        metadata: 附加元数据（可选）
    """
    metrics: dict[str, float]
    query_level_metrics: Optional[dict[str, dict[str, float]]] = None
    metadata: Optional[dict[str, Any]] = None
    
    def get_metric(self, name: str) -> Optional[float]:
        """获取指定指标的值。
        
        Args:
            name: 指标名称
            
        Returns:
            指标值，如果不存在则返回 None
        """
        return self.metrics.get(name)
    
    def get_average_metric(self, name: str) -> Optional[float]:
        """获取指定指标的平均值（跨所有查询）。
        
        Args:
            name: 指标名称
            
        Returns:
            平均指标值，如果不存在则返回 None
        """
        if not self.query_level_metrics:
            return self.get_metric(name)
        
        values = [
            query_metrics.get(name)
            for query_metrics in self.query_level_metrics.values()
            if name in query_metrics
        ]
        
        if not values:
            return None
        
        return sum(values) / len(values)


class BaseEvaluator(ABC):
    """Evaluator 抽象基类。
    
    定义统一的评估接口，所有具体实现必须继承此类。
    
    Args:
        provider: 评估提供商类型（如 "custom", "ragas", "deepeval"）
        metrics: 要计算的指标列表
        **kwargs: 提供商特定配置参数
    """
    
    def __init__(self, provider: str, metrics: list[str], **kwargs):
        """初始化 Evaluator。
        
        Args:
            provider: 评估提供商类型
            metrics: 要计算的指标列表
            **kwargs: 提供商特定配置参数
        """
        self.provider = provider
        self.metrics = metrics
        self.config = kwargs
        
        # 验证指标
        self.validate_metrics(metrics)
    
    @abstractmethod
    def evaluate(
        self,
        queries: list[EvaluationQuery],
        results: list[EvaluationResult],
        **kwargs
    ) -> EvaluationMetrics:
        """评估检索结果质量。
        
        Args:
            queries: 查询列表（包含黄金标准）
            results: 检索结果列表
            **kwargs: 额外参数（如 trace context）
            
        Returns:
            评估指标结果
            
        Raises:
            EvaluatorError: 评估失败时抛出
        """
        pass
    
    def validate_metrics(self, metrics: list[str]) -> None:
        """验证指标列表的有效性。
        
        Args:
            metrics: 指标列表
            
        Raises:
            EvaluatorError: 指标无效时抛出
        """
        if not metrics:
            raise EvaluatorError("指标列表不能为空")
        
        if not all(isinstance(m, str) for m in metrics):
            raise EvaluatorError("所有指标必须是字符串")
        
        # 检查重复
        if len(metrics) != len(set(metrics)):
            raise EvaluatorError("指标列表包含重复项")
    
    def validate_inputs(
        self,
        queries: list[EvaluationQuery],
        results: list[EvaluationResult]
    ) -> None:
        """验证输入数据的有效性。
        
        Args:
            queries: 查询列表
            results: 结果列表
            
        Raises:
            EvaluatorError: 输入无效时抛出
        """
        if not queries:
            raise EvaluatorError("查询列表不能为空")
        
        if not results:
            raise EvaluatorError("结果列表不能为空")
        
        if len(queries) != len(results):
            raise EvaluatorError(
                f"查询数量 ({len(queries)}) 与结果数量 ({len(results)}) 不匹配"
            )
        
        # 检查类型
        if not all(isinstance(q, EvaluationQuery) for q in queries):
            raise EvaluatorError("所有查询必须是 EvaluationQuery 实例")
        
        if not all(isinstance(r, EvaluationResult) for r in results):
            raise EvaluatorError("所有结果必须是 EvaluationResult 实例")
        
        # 检查 ID 匹配
        query_ids = [q.query_id for q in queries]
        result_ids = [r.query_id for r in results]
        
        if query_ids != result_ids:
            raise EvaluatorError("查询 ID 与结果 ID 不匹配")
    
    def get_supported_metrics(self) -> list[str]:
        """获取支持的指标列表。
        
        Returns:
            支持的指标名称列表
        """
        return []
    
    def __repr__(self) -> str:
        """返回 Evaluator 的字符串表示。"""
        return f"{self.__class__.__name__}(provider='{self.provider}', metrics={self.metrics})"
