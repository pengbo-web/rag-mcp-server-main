"""
CrossEncoderReranker 实现。

使用 Cross-Encoder 模型对候选项进行重排序，支持本地/托管模型。
适用于需要精确语义匹配的重排序场景。
"""

from typing import Optional, Callable
import time

from .base_reranker import BaseReranker, RerankCandidate, RerankResult, RerankerError


class CrossEncoderReranker(BaseReranker):
    """使用 Cross-Encoder 进行重排序的 Reranker。
    
    通过 Cross-Encoder 模型对 query-candidate 对进行打分。
    支持：
    - 可插拔的 scorer 函数（本地模型或 API）
    - 超时控制和失败回退
    - Batch 处理提升效率
    
    Args:
        scorer: 打分函数，签名为 scorer(query: str, texts: list[str]) -> list[float]
        model_name: 模型名称（可选，用于日志）
        timeout: 单次打分超时时间（秒，可选，默认 30）
        batch_size: 批处理大小（可选，默认 32）
        fallback_on_error: 错误时是否回退到原始顺序（可选，默认 True）
        **kwargs: 其他配置参数
    """
    
    def __init__(
        self,
        scorer: Optional[Callable[[str, list[str]], list[float]]] = None,
        model_name: str = "cross-encoder-default",
        timeout: int = 30,
        batch_size: int = 32,
        fallback_on_error: bool = True,
        **kwargs
    ):
        """初始化 CrossEncoderReranker。
        
        Args:
            scorer: 打分函数
            model_name: 模型名称
            timeout: 超时时间（秒）
            batch_size: 批处理大小
            fallback_on_error: 错误时是否回退
            **kwargs: 其他配置参数
        """
        super().__init__(backend="cross_encoder", **kwargs)
        
        # 如果未提供 scorer，使用默认的 mock scorer
        if scorer is None:
            scorer = self._default_mock_scorer
        
        self.scorer = scorer
        self.model_name = model_name
        self.timeout = timeout
        self.batch_size = batch_size
        self.fallback_on_error = fallback_on_error
        
        # 状态跟踪
        self._last_error: Optional[Exception] = None
        self._timeout_occurred: bool = False
    
    @staticmethod
    def _default_mock_scorer(query: str, texts: list[str]) -> list[float]:
        """默认的 mock scorer，用于测试和占位。
        
        基于查询和文本的字符串相似度生成确定性分数。
        
        Args:
            query: 查询文本
            texts: 候选项文本列表
            
        Returns:
            分数列表（0-1 之间）
        """
        scores = []
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        for text in texts:
            text_lower = text.lower()
            text_words = set(text_lower.split())
            
            # 计算词重叠比例
            if not query_words or not text_words:
                overlap = 0.0
            else:
                common = query_words & text_words
                overlap = len(common) / max(len(query_words), len(text_words))
            
            # 加入文本长度因子（较短的文本略微加权）
            length_factor = 1.0 / (1.0 + len(text) / 1000.0)
            score = overlap * 0.7 + length_factor * 0.3
            
            scores.append(score)
        
        return scores
    
    def _score_with_timeout(
        self,
        query: str,
        texts: list[str]
    ) -> list[float]:
        """带超时控制的打分。
        
        Args:
            query: 查询文本
            texts: 候选项文本列表
            
        Returns:
            分数列表
            
        Raises:
            RerankerError: 打分失败或超时
        """
        try:
            start_time = time.time()
            scores = self.scorer(query, texts)
            elapsed = time.time() - start_time
            
            # 检查是否超时
            if elapsed > self.timeout:
                self._timeout_occurred = True
                raise RerankerError(
                    f"Scoring timeout: took {elapsed:.2f}s, limit {self.timeout}s"
                )
            
            # 验证分数数量
            if len(scores) != len(texts):
                raise RerankerError(
                    f"Scorer returned {len(scores)} scores for {len(texts)} texts"
                )
            
            return scores
            
        except Exception as e:
            self._last_error = e
            raise RerankerError(f"Scoring failed: {e}")
    
    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        top_k: Optional[int] = None,
        **kwargs
    ) -> list[RerankResult]:
        """使用 Cross-Encoder 对候选项进行重排序。
        
        Args:
            query: 查询文本
            candidates: 候选项列表
            top_k: 返回前 k 个结果（可选，默认返回全部）
            **kwargs: 额外参数（如 trace context）
            
        Returns:
            重排序后的结果列表，按新分数降序排列
            
        Raises:
            RerankerError: 重排序失败且 fallback_on_error=False 时抛出
        """
        # 验证输入
        self.validate_candidates(candidates)
        
        # 重置状态
        self._last_error = None
        self._timeout_occurred = False
        
        try:
            # 批处理打分
            all_scores = []
            texts = [c.text for c in candidates]
            
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                batch_scores = self._score_with_timeout(query, batch_texts)
                all_scores.extend(batch_scores)
            
            # 创建结果
            results = self.create_results(candidates, all_scores)
            
            # 应用 top_k 限制
            if top_k is not None and top_k > 0:
                results = results[:top_k]
            
            return results
            
        except RerankerError as e:
            # 如果启用回退，返回原始顺序
            if self.fallback_on_error:
                # 使用原始分数（如果有），否则使用 0.0
                fallback_scores = [
                    c.score if c.score is not None else 0.0 
                    for c in candidates
                ]
                
                # 创建回退结果（保持原始顺序）
                results = []
                for i, (candidate, score) in enumerate(zip(candidates, fallback_scores)):
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
            else:
                # 不回退，直接抛出错误
                raise e
    
    def get_last_error(self) -> Optional[Exception]:
        """获取最后一次错误。
        
        Returns:
            最后一次错误，如果没有则返回 None
        """
        return self._last_error
    
    def did_timeout_occur(self) -> bool:
        """检查是否发生超时。
        
        Returns:
            True 如果最后一次调用发生超时
        """
        return self._timeout_occurred
    
    def __repr__(self) -> str:
        """返回 Reranker 的字符串表示。"""
        return (
            f"{self.__class__.__name__}("
            f"model='{self.model_name}', "
            f"timeout={self.timeout}, "
            f"fallback={self.fallback_on_error})"
        )
