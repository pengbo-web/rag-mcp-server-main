"""
Splitter 工厂模块。

此模块实现工厂模式，根据配置动态创建相应的 Splitter 实例。
支持多种切分策略：Recursive、Semantic、Fixed 等。
"""

from typing import Dict, Type

from core import Settings, IngestionSettings
from .base_splitter import BaseSplitter, SplitterError


class SplitterFactory:
    """
    Splitter 工厂类。
    
    负责根据配置创建对应的 Splitter 提供商实例。
    采用注册表模式，支持动态扩展新的切分策略。
    """
    
    # 提供商注册表：strategy_name -> Splitter 实现类
    _registry: Dict[str, Type[BaseSplitter]] = {}
    
    @classmethod
    def register(cls, strategy: str, splitter_class: Type[BaseSplitter]) -> None:
        """
        注册新的 Splitter 策略。
        
        Args:
            strategy: 策略名称（如 "recursive", "semantic", "fixed"）
            splitter_class: Splitter 实现类（必须继承 BaseSplitter）
            
        Raises:
            ValueError: 如果策略名称已被注册
        """
        if strategy in cls._registry:
            raise ValueError(f"Strategy '{strategy}' is already registered")
        
        if not issubclass(splitter_class, BaseSplitter):
            raise TypeError(f"{splitter_class} must inherit from BaseSplitter")
        
        cls._registry[strategy] = splitter_class
    
    @classmethod
    def create(
        cls,
        settings: Settings,
        strategy: str = "recursive"
    ) -> BaseSplitter:
        """
        根据配置创建 Splitter 实例。
        
        Args:
            settings: 全局配置对象，包含 Ingestion 设置
            strategy: 切分策略名称（默认 "recursive"）
            
        Returns:
            BaseSplitter: 对应策略的 Splitter 实例
            
        Raises:
            SplitterError: 当策略未注册或创建失败时抛出
        """
        ingestion_settings = settings.ingestion
        return cls.create_from_ingestion_settings(ingestion_settings, strategy)
    
    @classmethod
    def create_from_ingestion_settings(
        cls,
        ingestion_settings: IngestionSettings,
        strategy: str = "recursive"
    ) -> BaseSplitter:
        """
        从 Ingestion 设置直接创建实例（用于测试和独立场景）。
        
        Args:
            ingestion_settings: Ingestion 配置对象
            strategy: 切分策略名称
            
        Returns:
            BaseSplitter: 对应策略的 Splitter 实例
            
        Raises:
            SplitterError: 当策略未注册或创建失败时抛出
        """
        strategy_lower = strategy.lower()
        
        if strategy_lower not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise SplitterError(
                f"Unknown Splitter strategy: '{strategy}'. "
                f"Available strategies: {available or 'none (no strategies registered)'}"
            )
        
        splitter_class = cls._registry[strategy_lower]
        
        try:
            # 构造参数
            init_params = {
                "chunk_size": ingestion_settings.chunk_size,
                "chunk_overlap": ingestion_settings.chunk_overlap,
            }
            
            return splitter_class(**init_params)
        
        except Exception as e:
            raise SplitterError(
                f"Failed to create Splitter instance for strategy '{strategy}': {e}"
            ) from e
    
    @classmethod
    def create_with_params(
        cls,
        strategy: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        **kwargs
    ) -> BaseSplitter:
        """
        使用指定参数创建 Splitter 实例（用于独立使用场景）。
        
        Args:
            strategy: 切分策略名称
            chunk_size: 块大小
            chunk_overlap: 块重叠
            **kwargs: 其他策略特定参数
            
        Returns:
            BaseSplitter: Splitter 实例
            
        Raises:
            SplitterError: 当策略未注册或创建失败时抛出
        """
        strategy_lower = strategy.lower()
        
        if strategy_lower not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise SplitterError(
                f"Unknown Splitter strategy: '{strategy}'. "
                f"Available strategies: {available or 'none (no strategies registered)'}"
            )
        
        splitter_class = cls._registry[strategy_lower]
        
        try:
            return splitter_class(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                **kwargs
            )
        except Exception as e:
            raise SplitterError(
                f"Failed to create Splitter instance for strategy '{strategy}': {e}"
            ) from e
    
    @classmethod
    def list_strategies(cls) -> list[str]:
        """
        列出所有已注册的策略。
        
        Returns:
            list[str]: 策略名称列表
        """
        return list(cls._registry.keys())
    
    @classmethod
    def is_registered(cls, strategy: str) -> bool:
        """
        检查策略是否已注册。
        
        Args:
            strategy: 策略名称
            
        Returns:
            bool: 是否已注册
        """
        return strategy.lower() in cls._registry
    
    @classmethod
    def clear_registry(cls) -> None:
        """
        清空注册表（主要用于测试）。
        """
        cls._registry.clear()
