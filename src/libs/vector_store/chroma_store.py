"""
ChromaDB VectorStore 实现模块。

此模块提供 ChromaDB 的向量存储实现，支持本地持久化。
ChromaDB 是一个轻量级的向量数据库，适合原型开发和小规模生产场景。
"""

import os
from typing import List, Optional, Dict, Any

from .base_vector_store import (
    BaseVectorStore,
    VectorRecord,
    QueryResult,
    VectorStoreError
)


class ChromaStore(BaseVectorStore):
    """
    ChromaDB VectorStore 实现。
    
    提供基于 ChromaDB 的向量存储功能，支持：
    - 本地持久化存储
    - 向量相似度搜索
    - 元数据过滤
    - CRUD 操作
    
    Attributes:
        collection_name: 集合名称
        persist_directory: 持久化存储目录路径
        _client: ChromaDB 客户端实例
        _collection: ChromaDB 集合实例
    """
    
    def __init__(
        self,
        collection_name: str = "default",
        persist_directory: str = "./data/db/chroma",
        **kwargs
    ):
        """
        初始化 ChromaStore 实例。
        
        Args:
            collection_name: 集合名称
            persist_directory: 持久化目录路径
            **kwargs: 其他参数
            
        Raises:
            VectorStoreError: 当初始化失败时抛出
        """
        super().__init__(collection_name=collection_name, **kwargs)
        self.persist_directory = persist_directory
        
        try:
            # 延迟导入 chromadb，避免在未安装时阻塞其他模块
            import chromadb
            from chromadb.config import Settings as ChromaSettings
            
            # 确保持久化目录存在
            os.makedirs(self.persist_directory, exist_ok=True)
            
            # 创建持久化客户端
            self._client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # 获取或创建集合
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"description": f"Collection for {collection_name}"}
            )
            
        except ImportError as e:
            raise VectorStoreError(
                "chromadb is not installed. "
                "Install it with: pip install chromadb"
            ) from e
        except Exception as e:
            raise VectorStoreError(
                f"Failed to initialize ChromaStore: {e}"
            ) from e
    
    def upsert(
        self,
        records: List[VectorRecord],
        **kwargs
    ) -> None:
        """
        插入或更新向量记录到 ChromaDB。
        
        Args:
            records: 要插入/更新的记录列表
            **kwargs: 可选参数
            
        Raises:
            VectorStoreError: 当操作失败时抛出
        """
        if not records:
            return
        
        try:
            # 准备数据
            ids = [record.id for record in records]
            embeddings = [record.vector for record in records]
            documents = [record.text for record in records]
            metadatas = [
                record.metadata if record.metadata else {}
                for record in records
            ]
            
            # ChromaDB 的 upsert 操作
            self._collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            
        except Exception as e:
            raise VectorStoreError(
                f"Failed to upsert records to ChromaStore: {e}"
            ) from e
    
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
            filters: 元数据过滤条件（ChromaDB where 语法）
            **kwargs: 可选参数
            
        Returns:
            List[QueryResult]: 查询结果列表，按相似度降序排列
            
        Raises:
            VectorStoreError: 当查询失败时抛出
        """
        if not vector:
            raise VectorStoreError("Query vector cannot be empty")
        
        if top_k <= 0:
            raise VectorStoreError("top_k must be greater than 0")
        
        try:
            # 执行查询
            query_params = {
                "query_embeddings": [vector],
                "n_results": top_k,
            }
            
            # 添加元数据过滤（如果提供）
            if filters:
                query_params["where"] = filters
            
            results = self._collection.query(**query_params)
            
            # 转换结果格式
            query_results = []
            
            # ChromaDB 返回的结果是嵌套列表，需要展平
            ids = results["ids"][0] if results["ids"] else []
            distances = results["distances"][0] if results["distances"] else []
            documents = results["documents"][0] if results["documents"] else []
            metadatas = results["metadatas"][0] if results["metadatas"] else []
            # embeddings 可能不在结果中
            embeddings = results.get("embeddings")
            embeddings = embeddings[0] if embeddings else []
            
            for i in range(len(ids)):
                # ChromaDB 返回距离（越小越相似），转换为相似度分数
                # 使用简单的转换：score = 1 / (1 + distance)
                distance = distances[i] if i < len(distances) else 0.0
                score = 1.0 / (1.0 + distance)
                
                query_results.append(
                    QueryResult(
                        id=ids[i],
                        score=score,
                        text=documents[i] if i < len(documents) else "",
                        metadata=metadatas[i] if i < len(metadatas) else {},
                        vector=embeddings[i] if i < len(embeddings) else None
                    )
                )
            
            return query_results
            
        except Exception as e:
            raise VectorStoreError(
                f"Failed to query ChromaStore: {e}"
            ) from e
    
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
        if not ids:
            return
        
        try:
            self._collection.delete(ids=ids)
        except Exception as e:
            raise VectorStoreError(
                f"Failed to delete records from ChromaStore: {e}"
            ) from e
    
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
        if not ids:
            return []
        
        try:
            results = self._collection.get(
                ids=ids,
                include=["embeddings", "documents", "metadatas"]
            )
            
            # 转换结果格式
            records = []
            for i in range(len(results["ids"])):
                # 安全地获取 embeddings 和 documents
                embedding = []
                if results.get("embeddings") is not None and len(results["embeddings"]) > i:
                    embedding = results["embeddings"][i]
                
                document = ""
                if results.get("documents") is not None and len(results["documents"]) > i:
                    document = results["documents"][i]
                
                metadata = {}
                if results.get("metadatas") is not None and len(results["metadatas"]) > i:
                    metadata = results["metadatas"][i]
                
                records.append(
                    VectorRecord(
                        id=results["ids"][i],
                        vector=embedding,
                        text=document,
                        metadata=metadata
                    )
                )
            
            return records
            
        except Exception as e:
            raise VectorStoreError(
                f"Failed to get records from ChromaStore: {e}"
            ) from e
    
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
        try:
            return self._collection.count()
        except Exception as e:
            raise VectorStoreError(
                f"Failed to count records in ChromaStore: {e}"
            ) from e
    
    def clear(self, **kwargs) -> None:
        """
        清空集合中的所有记录。
        
        Args:
            **kwargs: 可选参数
            
        Raises:
            VectorStoreError: 当操作失败时抛出
        """
        try:
            # ChromaDB 没有直接的 clear 方法，需要删除并重新创建集合
            self._client.delete_collection(name=self.collection_name)
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": f"Collection for {self.collection_name}"}
            )
        except Exception as e:
            raise VectorStoreError(
                f"Failed to clear ChromaStore: {e}"
            ) from e
    
    def __repr__(self) -> str:
        """返回实例的字符串表示。"""
        return (
            f"ChromaStore(collection={self.collection_name}, "
            f"persist_directory={self.persist_directory})"
        )
