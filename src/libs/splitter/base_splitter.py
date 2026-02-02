"""
Splitter 抽象基类模块。

此模块定义了所有 Splitter 提供商必须实现的抽象接口。
支持可插拔架构，允许在不同的文本切分策略之间无缝切换。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class TextChunk:
    """
    文本块数据类。
    
    Attributes:
        text: 切分后的文本内容
        metadata: 元数据（如位置、来源等）
    """
    text: str
    metadata: Optional[Dict[str, Any]] = None


class SplitterError(Exception):
    """Splitter 调用过程中发生的错误。"""
    pass


class BaseSplitter(ABC):
    """
    Splitter 抽象基类。
    
    所有 Splitter 提供商实现都必须继承此类并实现 split_text 方法。
    此设计支持可插拔架构，允许通过配置切换不同的文本切分策略。
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        **kwargs
    ):
        """
        初始化 Splitter 实例。
        
        Args:
            chunk_size: 目标块大小（字符数）
            chunk_overlap: 块之间的重叠字符数
            **kwargs: 其他提供商特定参数
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.extra_params = kwargs
        
        # 验证参数
        if chunk_size <= 0:
            raise SplitterError(f"chunk_size must be positive, got {chunk_size}")
        if chunk_overlap < 0:
            raise SplitterError(f"chunk_overlap cannot be negative, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise SplitterError(
                f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})"
            )
    
    @abstractmethod
    def split_text(
        self,
        text: str,
        **kwargs
    ) -> List[str]:
        """
        将文本切分为块。
        
        Args:
            text: 要切分的文本
            **kwargs: 可选的切分参数
            
        Returns:
            List[str]: 切分后的文本块列表
            
        Raises:
            SplitterError: 当切分失败时抛出
        """
        pass
    
    def split_text_with_metadata(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[TextChunk]:
        """
        切分文本并附加元数据。
        
        Args:
            text: 要切分的文本
            metadata: 要附加到每个块的元数据
            **kwargs: 可选的切分参数
            
        Returns:
            List[TextChunk]: 带元数据的文本块列表
            
        Raises:
            SplitterError: 当切分失败时抛出
        """
        chunks = self.split_text(text, **kwargs)
        
        result = []
        for i, chunk_text in enumerate(chunks):
            # 为每个块创建元数据副本并添加位置信息
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata["chunk_index"] = i
            chunk_metadata["total_chunks"] = len(chunks)
            
            result.append(TextChunk(text=chunk_text, metadata=chunk_metadata))
        
        return result
    
    def get_num_chunks(self, text: str, **kwargs) -> int:
        """
        估算文本会被切分为多少块（无需实际切分）。
        
        Args:
            text: 要估算的文本
            **kwargs: 可选参数
            
        Returns:
            int: 预估的块数量
        """
        # 默认实现：实际切分后计数
        # 子类可以覆盖此方法以提供更高效的估算
        chunks = self.split_text(text, **kwargs)
        return len(chunks)
    
    def __repr__(self) -> str:
        """返回实例的字符串表示。"""
        return (
            f"{self.__class__.__name__}"
            f"(chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap})"
        )
