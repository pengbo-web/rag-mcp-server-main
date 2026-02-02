"""
VectorStore 抽象基类模块。

此模块定义了所有 VectorStore 提供商必须实现的抽象接口。
支持可插拔架构，允许在不同的向量数据库后端之间无缝切换。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class VectorRecord:
    """
    向量记录数据类。
    
    Attributes:
        id: 记录唯一标识符
        vector: 嵌入向量
        text: 原始文本内容
        metadata: 元数据字典
    """
    id: str
    vector: List[float]
    text: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class QueryResult:
    """
    查询结果数据类。
    
    Attributes:
        id: 记录标识符
        score: 相似度分数
        text: 文本内容
        metadata: 元数据
        vector: 向量（可选）
    """
    id: str
    score: float
    text: str
    metadata: Optional[Dict[str, Any]] = None
    vector: Optional[List[float]] = None


class VectorStoreError(Exception):
    """VectorStore 操作过程中发生的错误。"""
    pass


class BaseVectorStore(ABC):
    """
    VectorStore 抽象基类。
    
    所有 VectorStore 提供商实现都必须继承此类并实现核心方法。
    此设计支持可插拔架构，允许通过配置切换不同的向量数据库后端。
    """
    
    def __init__(
        self,
        collection_name: str = "default",
        **kwargs
    ):
        """
        初始化 VectorStore 实例。
        
        Args:
            collection_name: 集合/表名称
            **kwargs: 其他提供商特定参数
        """
        self.collection_name = collection_name
        self.extra_params = kwargs
    
    @abstractmethod
    def upsert(
        self,
        records: List[VectorRecord],
        **kwargs
    ) -> None:
        """
        插入或更新向量记录。
        
        Args:
            records: 要插入/更新的记录列表
            **kwargs: 可选参数
            
        Raises:
            VectorStoreError: 当操作失败时抛出
        """
        pass
    
    @abstractmethod
    def query(
        self,
        vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[QueryResult]:
        """
        查询最相似的向量。
        
        Args:
            vector: 查询向量
            top_k: 返回结果数量
            filters: 元数据过滤条件（可选）
            **kwargs: 可选参数
            
        Returns:
            List[QueryResult]: 查询结果列表，按相似度降序排列
            
        Raises:
            VectorStoreError: 当查询失败时抛出
        """
        pass
    
    @abstractmethod
    def delete(
        self,
        ids: List[str],
        **kwargs
    ) -> None:
        """
        删除指定记录。
        
        Args:
            ids: 要删除的记录 ID 列表
            **kwargs: 可选参数
            
        Raises:
            VectorStoreError: 当删除失败时抛出
        """
        pass
    
    @abstractmethod
    def get(
        self,
        ids: List[str],
        **kwargs
    ) -> List[VectorRecord]:
        """
        根据 ID 获取记录。
        
        Args:
            ids: 记录 ID 列表
            **kwargs: 可选参数
            
        Returns:
            List[VectorRecord]: 记录列表
            
        Raises:
            VectorStoreError: 当获取失败时抛出
        """
        pass
    
    def count(self, **kwargs) -> int:
        """
        获取集合中的记录数量。
        
        Args:
            **kwargs: 可选参数
            
        Returns:
            int: 记录数量
            
        Raises:
            VectorStoreError: 当操作失败时抛出
        """
        # 默认实现：子类可以覆盖以提供更高效的实现
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement count() method"
        )
    
    def clear(self, **kwargs) -> None:
        """
        清空集合中的所有记录。
        
        Args:
            **kwargs: 可选参数
            
        Raises:
            VectorStoreError: 当操作失败时抛出
        """
        # 默认实现：子类可以覆盖以提供更高效的实现
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement clear() method"
        )
    
    def __repr__(self) -> str:
        """返回实例的字符串表示。"""
        return f"{self.__class__.__name__}(collection={self.collection_name})"
