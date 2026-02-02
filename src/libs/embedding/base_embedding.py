"""
Embedding 抽象基类模块。

此模块定义了所有 Embedding 提供商必须实现的抽象接口。
支持可插拔架构，允许在不同的 Embedding 后端之间无缝切换。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class EmbeddingResponse:
    """
    Embedding 响应数据类。
    
    Attributes:
        embeddings: 嵌入向量列表
        model: 使用的模型名称
        dimensions: 向量维度
        usage: token 使用信息（可选）
    """
    embeddings: List[List[float]]
    model: str
    dimensions: int
    usage: Optional[dict] = None


class EmbeddingError(Exception):
    """Embedding 调用过程中发生的错误。"""
    pass


class BaseEmbedding(ABC):
    """
    Embedding 抽象基类。
    
    所有 Embedding 提供商实现都必须继承此类并实现 embed 方法。
    此设计支持可插拔架构，允许通过配置切换不同的 Embedding 后端。
    """
    
    def __init__(
        self,
        model: str,
        dimensions: int = 1536,
        batch_size: int = 100,
        **kwargs
    ):
        """
        初始化 Embedding 实例。
        
        Args:
            model: 模型名称（如 "text-embedding-3-small", "bge-large-zh"）
            dimensions: 嵌入向量维度
            batch_size: 批处理大小
            **kwargs: 其他提供商特定参数
        """
        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.extra_params = kwargs
    
    @abstractmethod
    def embed(
        self,
        texts: List[str],
        **kwargs
    ) -> EmbeddingResponse:
        """
        将文本列表转换为嵌入向量。
        
        Args:
            texts: 要嵌入的文本列表
            **kwargs: 可选的请求参数
            
        Returns:
            EmbeddingResponse: 包含嵌入向量的响应对象
            
        Raises:
            EmbeddingError: 当调用失败时抛出
        """
        pass
    
    def embed_single(self, text: str, **kwargs) -> List[float]:
        """
        嵌入单个文本（简化接口）。
        
        Args:
            text: 要嵌入的文本
            **kwargs: 可选的请求参数
            
        Returns:
            List[float]: 嵌入向量
            
        Raises:
            EmbeddingError: 当调用失败时抛出
        """
        response = self.embed([text], **kwargs)
        return response.embeddings[0]
    
    def embed_batch(
        self,
        texts: List[str],
        **kwargs
    ) -> List[List[float]]:
        """
        批量嵌入文本，自动分批处理（简化接口）。
        
        Args:
            texts: 要嵌入的文本列表
            **kwargs: 可选的请求参数
            
        Returns:
            List[List[float]]: 嵌入向量列表
            
        Raises:
            EmbeddingError: 当调用失败时抛出
        """
        if not texts:
            return []
        
        all_embeddings = []
        
        # 分批处理
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            response = self.embed(batch, **kwargs)
            all_embeddings.extend(response.embeddings)
        
        return all_embeddings
    
    def __repr__(self) -> str:
        """返回实例的字符串表示。"""
        return f"{self.__class__.__name__}(model={self.model}, dimensions={self.dimensions})"
