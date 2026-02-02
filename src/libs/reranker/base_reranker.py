"""
Reranker 基类与相关数据模型。

定义统一的重排序接口，支持多种后端实现（LLM、Cross-Encoder 等）。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


class RerankerError(Exception):
    """Reranker 相关错误的基类。"""
    pass


@dataclass
class RerankCandidate:
    """重排序候选项。
    
    Attributes:
        id: 候选项唯一标识符
        text: 候选项文本内容
        score: 初始检索分数（可选）
        metadata: 附加元数据（可选）
    """
    id: str
    text: str
    score: Optional[float] = None
    metadata: Optional[dict[str, Any]] = None


@dataclass
class RerankResult:
    """重排序结果。
    
    Attributes:
        id: 候选项 ID
        text: 候选项文本
        score: 重排序后的分数
        original_rank: 原始排名位置（从 0 开始）
        new_rank: 新排名位置（从 0 开始）
        metadata: 附加元数据（可选）
    """
    id: str
    text: str
    score: float
    original_rank: int
    new_rank: int
    metadata: Optional[dict[str, Any]] = None


class BaseReranker(ABC):
    """Reranker 抽象基类。
    
    定义统一的重排序接口，所有具体实现必须继承此类。
    
    Args:
        backend: 重排序后端类型（如 "none", "llm", "cross_encoder"）
        **kwargs: 后端特定配置参数
    """
    
    def __init__(self, backend: str, **kwargs):
        """初始化 Reranker。
        
        Args:
            backend: 重排序后端类型
            **kwargs: 后端特定配置参数
        """
        self.backend = backend
        self.config = kwargs
    
    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        top_k: Optional[int] = None,
        **kwargs
    ) -> list[RerankResult]:
        """对候选项进行重排序。
        
        Args:
            query: 查询文本
            candidates: 候选项列表
            top_k: 返回前 k 个结果（可选，默认返回全部）
            **kwargs: 额外参数（如 trace context）
            
        Returns:
            重排序后的结果列表，按新分数降序排列
            
        Raises:
            RerankerError: 重排序失败时抛出
        """
        pass
    
    def validate_candidates(self, candidates: list[RerankCandidate]) -> None:
        """验证候选项列表的有效性。
        
        Args:
            candidates: 候选项列表
            
        Raises:
            RerankerError: 候选项无效时抛出
        """
        if not candidates:
            raise RerankerError("候选项列表不能为空")
        
        if not all(isinstance(c, RerankCandidate) for c in candidates):
            raise RerankerError("所有候选项必须是 RerankCandidate 实例")
        
        # 检查 ID 唯一性
        ids = [c.id for c in candidates]
        if len(ids) != len(set(ids)):
            raise RerankerError("候选项 ID 必须唯一")
    
    def create_results(
        self,
        candidates: list[RerankCandidate],
        scores: list[float],
        original_ranks: Optional[list[int]] = None
    ) -> list[RerankResult]:
        """创建重排序结果。
        
        Args:
            candidates: 原始候选项列表
            scores: 新分数列表（与 candidates 对应）
            original_ranks: 原始排名列表（可选，默认使用输入顺序）
            
        Returns:
            重排序结果列表，按新分数降序排列
        """
        if len(candidates) != len(scores):
            raise RerankerError(
                f"候选项数量 ({len(candidates)}) 与分数数量 ({len(scores)}) 不匹配"
            )
        
        # 如果未提供原始排名，使用输入顺序
        if original_ranks is None:
            original_ranks = list(range(len(candidates)))
        
        # 创建结果并按分数排序
        results = []
        for i, (candidate, score) in enumerate(zip(candidates, scores)):
            results.append(RerankResult(
                id=candidate.id,
                text=candidate.text,
                score=score,
                original_rank=original_ranks[i],
                new_rank=-1,  # 临时值，稍后更新
                metadata=candidate.metadata
            ))
        
        # 按分数降序排序
        results.sort(key=lambda x: x.score, reverse=True)
        
        # 更新新排名
        for new_rank, result in enumerate(results):
            result.new_rank = new_rank
        
        return results
    
    def __repr__(self) -> str:
        """返回 Reranker 的字符串表示。"""
        return f"{self.__class__.__name__}(backend='{self.backend}')"
