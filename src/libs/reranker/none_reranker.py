"""
NoneReranker 实现。

不进行任何重排序，保持原始顺序和分数。
用作默认回退策略或在不需要重排序时使用。
"""

from typing import Optional

from .base_reranker import BaseReranker, RerankCandidate, RerankResult


class NoneReranker(BaseReranker):
    """不进行重排序的 Reranker。
    
    保持候选项的原始顺序和分数，不做任何修改。
    适用于：
    - 不需要重排序的场景
    - 作为其他 Reranker 失败时的回退策略
    - 测试和基准对比
    
    Args:
        **kwargs: 配置参数（会被忽略）
    """
    
    def __init__(self, **kwargs):
        """初始化 NoneReranker。
        
        Args:
            **kwargs: 配置参数（会被忽略）
        """
        super().__init__(backend="none", **kwargs)
    
    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        top_k: Optional[int] = None,
        **kwargs
    ) -> list[RerankResult]:
        """保持原始顺序，不进行重排序。
        
        Args:
            query: 查询文本（会被忽略）
            candidates: 候选项列表
            top_k: 返回前 k 个结果（可选，默认返回全部）
            **kwargs: 额外参数（会被忽略）
            
        Returns:
            原始顺序的结果列表，分数保持不变（如果无分数则设为 0.0）
        """
        # 验证输入
        self.validate_candidates(candidates)
        
        # 提取或生成分数（保持原始分数，无分数则为 0.0）
        scores = [c.score if c.score is not None else 0.0 for c in candidates]
        
        # 创建结果（使用原始顺序）
        # 注意：create_results 会按分数排序，所以我们需要手动创建结果以保持原始顺序
        results = []
        for i, (candidate, score) in enumerate(zip(candidates, scores)):
            results.append(RerankResult(
                id=candidate.id,
                text=candidate.text,
                score=score,
                original_rank=i,
                new_rank=i,  # 保持原始排名
                metadata=candidate.metadata
            ))
        
        # 应用 top_k 限制
        if top_k is not None and top_k > 0:
            results = results[:top_k]
        
        return results
