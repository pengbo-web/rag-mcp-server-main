"""
Embedding 工厂模块。

此模块实现工厂模式，根据配置动态创建相应的 Embedding 实例。
支持多种提供商：OpenAI、Local（BGE/Sentence-Transformers）等。
"""

from typing import Dict, Type

from core import Settings, EmbeddingSettings
from .base_embedding import BaseEmbedding, EmbeddingError


class EmbeddingFactory:
    """
    Embedding 工厂类。
    
    负责根据配置创建对应的 Embedding 提供商实例。
    采用注册表模式，支持动态扩展新的提供商。
    """
    
    # 提供商注册表：provider_name -> Embedding 实现类
    _registry: Dict[str, Type[BaseEmbedding]] = {}
    
    @classmethod
    def register(cls, provider: str, embedding_class: Type[BaseEmbedding]) -> None:
        """
        注册新的 Embedding 提供商。
        
        Args:
            provider: 提供商名称（如 "openai", "local"）
            embedding_class: Embedding 实现类（必须继承 BaseEmbedding）
            
        Raises:
            ValueError: 如果提供商名称已被注册
        """
        if provider in cls._registry:
            raise ValueError(f"Provider '{provider}' is already registered")
        
        if not issubclass(embedding_class, BaseEmbedding):
            raise TypeError(f"{embedding_class} must inherit from BaseEmbedding")
        
        cls._registry[provider] = embedding_class
    
    @classmethod
    def create(cls, settings: Settings) -> BaseEmbedding:
        """
        根据配置创建 Embedding 实例。
        
        Args:
            settings: 全局配置对象，包含 Embedding 设置
            
        Returns:
            BaseEmbedding: 对应提供商的 Embedding 实例
            
        Raises:
            EmbeddingError: 当提供商未注册或创建失败时抛出
        """
        embedding_settings = settings.embedding
        return cls.create_from_embedding_settings(embedding_settings)
    
    @classmethod
    def create_from_embedding_settings(cls, embedding_settings: EmbeddingSettings) -> BaseEmbedding:
        """
        从 Embedding 设置直接创建实例（用于测试和独立场景）。
        
        Args:
            embedding_settings: Embedding 配置对象
            
        Returns:
            BaseEmbedding: 对应提供商的 Embedding 实例
            
        Raises:
            EmbeddingError: 当提供商未注册或创建失败时抛出
        """
        provider = embedding_settings.provider.lower()
        
        if provider not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise EmbeddingError(
                f"Unknown Embedding provider: '{provider}'. "
                f"Available providers: {available or 'none (no providers registered)'}"
            )
        
        embedding_class = cls._registry[provider]
        
        try:
            # 构造参数
            init_params = {
                "model": embedding_settings.model,
                "dimensions": embedding_settings.dimensions,
                "batch_size": embedding_settings.batch_size,
            }
            
            # 添加可选参数
            if embedding_settings.api_key:
                init_params["api_key"] = embedding_settings.api_key
            
            return embedding_class(**init_params)
        
        except Exception as e:
            raise EmbeddingError(
                f"Failed to create Embedding instance for provider '{provider}': {e}"
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
