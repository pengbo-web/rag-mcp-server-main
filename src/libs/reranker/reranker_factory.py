"""
Reranker 工厂类。

提供统一的 Reranker 创建接口，支持多种后端动态注册和实例化。
"""

from typing import Type

from core import Settings, RerankSettings

from .base_reranker import BaseReranker, RerankerError
from .none_reranker import NoneReranker


class RerankerFactory:
    """Reranker 工厂类。
    
    使用注册表模式管理多种 Reranker 后端实现。
    支持运行时动态注册新的后端类型。
    
    使用示例:
        # 注册后端
        RerankerFactory.register("none", NoneReranker)
        RerankerFactory.register("llm", LLMReranker)
        
        # 从配置创建
        settings = Settings(...)
        reranker = RerankerFactory.create(settings)
        
        # 直接创建
        reranker = RerankerFactory.create_with_params(backend="none")
    """
    
    # 后端注册表：backend_name -> Reranker 类
    _registry: dict[str, Type[BaseReranker]] = {}
    
    @classmethod
    def register(cls, backend: str, reranker_class: Type[BaseReranker]) -> None:
        """注册 Reranker 后端。
        
        Args:
            backend: 后端标识符（如 "none", "llm", "cross_encoder"）
            reranker_class: Reranker 类（必须继承自 BaseReranker）
            
        Raises:
            ValueError: 如果后端已注册或类型不合法
        """
        backend_lower = backend.lower()
        
        # 检查是否已注册
        if backend_lower in cls._registry:
            raise ValueError(
                f"Reranker backend '{backend}' is already registered. "
                f"Use a different name or unregister the existing one first."
            )
        
        # 检查类型
        if not issubclass(reranker_class, BaseReranker):
            raise TypeError(
                f"Reranker class must inherit from BaseReranker, "
                f"got {reranker_class.__name__}"
            )
        
        cls._registry[backend_lower] = reranker_class
    
    @classmethod
    def create_from_rerank_settings(cls, rerank_settings: RerankSettings) -> BaseReranker:
        """从 RerankSettings 创建 Reranker 实例。
        
        Args:
            rerank_settings: Rerank 配置对象
            
        Returns:
            Reranker 实例
            
        Raises:
            RerankerError: 后端未注册或创建失败
        """
        backend = rerank_settings.provider.lower()
        
        # 检查后端是否注册
        if backend not in cls._registry:
            raise RerankerError(
                f"Unknown Reranker backend: '{rerank_settings.backend}'. "
                f"Available backends: {', '.join(cls.list_backends())}"
            )
        
        reranker_class = cls._registry[backend]
        
        try:
            # 传递配置参数（移除 provider 字段）
            config = {k: v for k, v in rerank_settings.__dict__.items() if k != "provider"}
            return reranker_class(**config)
        except Exception as e:
            raise RerankerError(
                f"Failed to create Reranker with backend '{rerank_settings.provider}': {e}"
            ) from e
    
    @classmethod
    def create(cls, settings: Settings) -> BaseReranker:
        """从完整 Settings 创建 Reranker 实例。
        
        Args:
            settings: 完整配置对象
            
        Returns:
            Reranker 实例
        """
        return cls.create_from_rerank_settings(settings.rerank)
    
    @classmethod
    def create_with_params(cls, backend: str, **kwargs) -> BaseReranker:
        """使用指定参数创建 Reranker 实例。
        
        Args:
            backend: 后端标识符
            **kwargs: 后端特定配置参数
            
        Returns:
            Reranker 实例
            
        Raises:
            RerankerError: 后端未注册或创建失败
        """
        backend_lower = backend.lower()
        
        # 检查后端是否注册
        if backend_lower not in cls._registry:
            raise RerankerError(
                f"Unknown Reranker backend: '{backend}'. "
                f"Available backends: {', '.join(cls.list_backends())}"
            )
        
        reranker_class = cls._registry[backend_lower]
        
        try:
            return reranker_class(**kwargs)
        except Exception as e:
            raise RerankerError(
                f"Failed to create Reranker with backend '{backend}': {e}"
            ) from e
    
    @classmethod
    def is_registered(cls, backend: str) -> bool:
        """检查后端是否已注册。
        
        Args:
            backend: 后端标识符
            
        Returns:
            True 如果已注册，否则 False
        """
        return backend.lower() in cls._registry
    
    @classmethod
    def list_backends(cls) -> list[str]:
        """列出所有已注册的后端。
        
        Returns:
            后端标识符列表
        """
        return list(cls._registry.keys())
    
    @classmethod
    def clear_registry(cls) -> None:
        """清空注册表（主要用于测试）。"""
        cls._registry.clear()
