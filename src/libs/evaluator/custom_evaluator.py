"""
CustomEvaluator 实现。

实现轻量级自定义评估指标，包括：
- hit_rate: 命中率（至少有一个相关文档被检索到的查询比例）
- mrr: 平均倒数排名（Mean Reciprocal Rank）
- precision@k: 前 k 个结果的精确率
- recall@k: 前 k 个结果的召回率
- ndcg@k: 归一化折损累积增益（Normalized Discounted Cumulative Gain）
"""

from typing import Optional

from .base_evaluator import (
    BaseEvaluator,
    EvaluationQuery,
    EvaluationResult,
    EvaluationMetrics,
    EvaluatorError,
)


class CustomEvaluator(BaseEvaluator):
    """自定义评估器。
    
    实现基于信息检索的经典评估指标。
    
    支持的指标：
    - hit_rate: 命中率
    - mrr: 平均倒数排名
    - precision@k: 精确率（k 由配置或结果决定）
    - recall@k: 召回率
    - ndcg@k: 归一化折损累积增益
    
    Args:
        metrics: 要计算的指标列表
        k: 前 k 个结果（默认为 10）
        **kwargs: 其他配置参数
    """
    
    SUPPORTED_METRICS = [
        "hit_rate",
        "mrr",
        "precision@k",
        "recall@k",
        "ndcg@k",
    ]
    
    def __init__(self, metrics: list[str], k: int = 10, **kwargs):
        """初始化 CustomEvaluator。
        
        Args:
            metrics: 要计算的指标列表
            k: 前 k 个结果
            **kwargs: 其他配置参数
        """
        super().__init__(provider="custom", metrics=metrics, **kwargs)
        self.k = k
    
    def validate_metrics(self, metrics: list[str]) -> None:
        """验证指标是否支持。
        
        Args:
            metrics: 指标列表
            
        Raises:
            EvaluatorError: 包含不支持的指标时抛出
        """
        super().validate_metrics(metrics)
        
        unsupported = set(metrics) - set(self.SUPPORTED_METRICS)
        if unsupported:
            raise EvaluatorError(
                f"不支持的指标: {', '.join(unsupported)}。"
                f"支持的指标: {', '.join(self.SUPPORTED_METRICS)}"
            )
    
    def evaluate(
        self,
        queries: list[EvaluationQuery],
        results: list[EvaluationResult],
        **kwargs
    ) -> EvaluationMetrics:
        """评估检索结果。
        
        Args:
            queries: 查询列表
            results: 结果列表
            **kwargs: 额外参数
            
        Returns:
            评估指标结果
        """
        # 验证输入
        self.validate_inputs(queries, results)
        
        # 计算每个查询的指标
        query_level_metrics = {}
        
        for query, result in zip(queries, results):
            query_metrics = {}
            
            # 计算各项指标
            if "hit_rate" in self.metrics:
                query_metrics["hit_rate"] = self._calculate_hit_rate(
                    query.golden_ids, result.retrieved_ids
                )
            
            if "mrr" in self.metrics:
                query_metrics["mrr"] = self._calculate_mrr(
                    query.golden_ids, result.retrieved_ids
                )
            
            if "precision@k" in self.metrics:
                query_metrics["precision@k"] = self._calculate_precision_at_k(
                    query.golden_ids, result.retrieved_ids, self.k
                )
            
            if "recall@k" in self.metrics:
                query_metrics["recall@k"] = self._calculate_recall_at_k(
                    query.golden_ids, result.retrieved_ids, self.k
                )
            
            if "ndcg@k" in self.metrics:
                query_metrics["ndcg@k"] = self._calculate_ndcg_at_k(
                    query.golden_ids, result.retrieved_ids, self.k
                )
            
            query_level_metrics[query.query_id] = query_metrics
        
        # 计算整体平均指标
        overall_metrics = {}
        for metric_name in self.metrics:
            values = [
                qm[metric_name]
                for qm in query_level_metrics.values()
                if metric_name in qm
            ]
            if values:
                overall_metrics[metric_name] = sum(values) / len(values)
        
        return EvaluationMetrics(
            metrics=overall_metrics,
            query_level_metrics=query_level_metrics,
            metadata={"k": self.k}
        )
    
    def _calculate_hit_rate(
        self,
        golden_ids: list[str],
        retrieved_ids: list[str]
    ) -> float:
        """计算命中率（是否至少命中一个相关文档）。
        
        Args:
            golden_ids: 黄金标准 ID 列表
            retrieved_ids: 检索到的 ID 列表
            
        Returns:
            命中率（0 或 1）
        """
        golden_set = set(golden_ids)
        retrieved_set = set(retrieved_ids)
        
        return 1.0 if golden_set & retrieved_set else 0.0
    
    def _calculate_mrr(
        self,
        golden_ids: list[str],
        retrieved_ids: list[str]
    ) -> float:
        """计算倒数排名（第一个相关文档的排名倒数）。
        
        Args:
            golden_ids: 黄金标准 ID 列表
            retrieved_ids: 检索到的 ID 列表
            
        Returns:
            倒数排名（0 到 1 之间）
        """
        golden_set = set(golden_ids)
        
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in golden_set:
                return 1.0 / rank
        
        return 0.0
    
    def _calculate_precision_at_k(
        self,
        golden_ids: list[str],
        retrieved_ids: list[str],
        k: int
    ) -> float:
        """计算前 k 个结果的精确率。
        
        Args:
            golden_ids: 黄金标准 ID 列表
            retrieved_ids: 检索到的 ID 列表
            k: 前 k 个结果
            
        Returns:
            精确率（0 到 1 之间）
        """
        golden_set = set(golden_ids)
        top_k = retrieved_ids[:k]
        
        if not top_k:
            return 0.0
        
        relevant_count = sum(1 for doc_id in top_k if doc_id in golden_set)
        return relevant_count / len(top_k)
    
    def _calculate_recall_at_k(
        self,
        golden_ids: list[str],
        retrieved_ids: list[str],
        k: int
    ) -> float:
        """计算前 k 个结果的召回率。
        
        Args:
            golden_ids: 黄金标准 ID 列表
            retrieved_ids: 检索到的 ID 列表
            k: 前 k 个结果
            
        Returns:
            召回率（0 到 1 之间）
        """
        if not golden_ids:
            return 0.0
        
        golden_set = set(golden_ids)
        top_k = retrieved_ids[:k]
        
        relevant_count = sum(1 for doc_id in top_k if doc_id in golden_set)
        return relevant_count / len(golden_ids)
    
    def _calculate_ndcg_at_k(
        self,
        golden_ids: list[str],
        retrieved_ids: list[str],
        k: int
    ) -> float:
        """计算归一化折损累积增益。
        
        Args:
            golden_ids: 黄金标准 ID 列表
            retrieved_ids: 检索到的 ID 列表
            k: 前 k 个结果
            
        Returns:
            NDCG 值（0 到 1 之间）
        """
        golden_set = set(golden_ids)
        top_k = retrieved_ids[:k]
        
        # 计算 DCG
        dcg = 0.0
        for i, doc_id in enumerate(top_k, start=1):
            relevance = 1.0 if doc_id in golden_set else 0.0
            dcg += relevance / self._log2(i + 1)
        
        # 计算理想 DCG（IDCG）
        ideal_relevances = [1.0] * min(len(golden_ids), k)
        idcg = sum(
            rel / self._log2(i + 2)
            for i, rel in enumerate(ideal_relevances)
        )
        
        # 归一化
        if idcg == 0.0:
            return 0.0
        
        return dcg / idcg
    
    @staticmethod
    def _log2(x: float) -> float:
        """计算以 2 为底的对数。"""
        import math
        return math.log2(x) if x > 0 else 0.0
    
    def get_supported_metrics(self) -> list[str]:
        """获取支持的指标列表。
        
        Returns:
            支持的指标名称列表
        """
        return self.SUPPORTED_METRICS.copy()
