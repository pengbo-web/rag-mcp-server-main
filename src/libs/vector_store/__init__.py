"""
VectorStore 库模块。

提供统一的 VectorStore 抽象接口和工厂，支持多种向量数据库后端。
"""

from .base_vector_store import BaseVectorStore, VectorRecord, QueryResult, VectorStoreError
from .vector_store_factory import VectorStoreFactory
from .chroma_store import ChromaStore

# 注册 ChromaStore 提供商
VectorStoreFactory.register("chroma", ChromaStore)

__all__ = [
    "BaseVectorStore",
    "VectorRecord",
    "QueryResult",
    "VectorStoreError",
    "VectorStoreFactory",
    "ChromaStore",
]
