"""
VectorStore 工厂模块。

此模块实现工厂模式，根据配置动态创建相应的 VectorStore 实例。
支持多种后端：Chroma、Qdrant、Milvus 等。
"""

from typing import Dict, Type

from core import Settings, VectorStoreSettings
from .base_vector_store import BaseVectorStore, VectorStoreError


class VectorStoreFactory:
    """
    VectorStore 工厂类。
    
    负责根据配置创建对应的 VectorStore 提供商实例。
    采用注册表模式，支持动态扩展新的向量数据库后端。
    """
    
    # 提供商注册表：provider_name -> VectorStore 实现类
    _registry: Dict[str, Type[BaseVectorStore]] = {}
    
    @classmethod
    def register(cls, provider: str, vector_store_class: Type[BaseVectorStore]) -> None:
        """
        注册新的 VectorStore 提供商。
        
        Args:
            provider: 提供商名称（如 "chroma", "qdrant", "milvus"）
            vector_store_class: VectorStore 实现类（必须继承 BaseVectorStore）
            
        Raises:
            ValueError: 如果提供商名称已被注册
        """
        if provider in cls._registry:
            raise ValueError(f"Provider '{provider}' is already registered")
        
        if not issubclass(vector_store_class, BaseVectorStore):
            raise TypeError(f"{vector_store_class} must inherit from BaseVectorStore")
        
        cls._registry[provider] = vector_store_class
    
    @classmethod
    def create(cls, settings: Settings) -> BaseVectorStore:
        """
        根据配置创建 VectorStore 实例。
        
        Args:
            settings: 全局配置对象，包含 VectorStore 设置
            
        Returns:
            BaseVectorStore: 对应提供商的 VectorStore 实例
            
        Raises:
            VectorStoreError: 当提供商未注册或创建失败时抛出
        """
        vector_store_settings = settings.vector_store
        return cls.create_from_vector_store_settings(vector_store_settings)
    
    @classmethod
    def create_from_vector_store_settings(
        cls,
        vector_store_settings: VectorStoreSettings
    ) -> BaseVectorStore:
        """
        从 VectorStore 设置直接创建实例（用于测试和独立场景）。
        
        Args:
            vector_store_settings: VectorStore 配置对象
            
        Returns:
            BaseVectorStore: 对应提供商的 VectorStore 实例
            
        Raises:
            VectorStoreError: 当提供商未注册或创建失败时抛出
        """
        provider = vector_store_settings.provider.lower()
        
        if provider not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise VectorStoreError(
                f"Unknown VectorStore provider: '{provider}'. "
                f"Available providers: {available or 'none (no providers registered)'}"
            )
        
        vector_store_class = cls._registry[provider]
        
        try:
            # 构造参数
            init_params = {
                "collection_name": vector_store_settings.collection_name,
            }
            
            # 添加提供商特定参数
            if hasattr(vector_store_settings, 'persist_directory'):
                init_params["persist_directory"] = vector_store_settings.persist_directory
            
            return vector_store_class(**init_params)
        
        except Exception as e:
            raise VectorStoreError(
                f"Failed to create VectorStore instance for provider '{provider}': {e}"
            ) from e
    
    @classmethod
    def create_with_params(
        cls,
        provider: str,
        collection_name: str = "default",
        **kwargs
    ) -> BaseVectorStore:
        """
        使用指定参数创建 VectorStore 实例（用于独立使用场景）。
        
        Args:
            provider: 提供商名称
            collection_name: 集合名称
            **kwargs: 其他提供商特定参数
            
        Returns:
            BaseVectorStore: VectorStore 实例
            
        Raises:
            VectorStoreError: 当提供商未注册或创建失败时抛出
        """
        provider_lower = provider.lower()
        
        if provider_lower not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise VectorStoreError(
                f"Unknown VectorStore provider: '{provider}'. "
                f"Available providers: {available or 'none (no providers registered)'}"
            )
        
        vector_store_class = cls._registry[provider_lower]
        
        try:
            return vector_store_class(
                collection_name=collection_name,
                **kwargs
            )
        except Exception as e:
            raise VectorStoreError(
                f"Failed to create VectorStore instance for provider '{provider}': {e}"
            ) from e
    
    @classmethod
    def list_providers(cls) -> list[str]:
        """
        列出所有已注册的提供商。
        
        Returns:
            list[str]: 提供商名称列表
        """
        return list(cls._registry.keys())
    
    @classmethod
    def is_registered(cls, provider: str) -> bool:
        """
        检查提供商是否已注册。
        
        Args:
            provider: 提供商名称
            
        Returns:
            bool: 是否已注册
        """
        return provider.lower() in cls._registry
    
    @classmethod
    def clear_registry(cls) -> None:
        """
        清空注册表（主要用于测试）。
        """
        cls._registry.clear()
