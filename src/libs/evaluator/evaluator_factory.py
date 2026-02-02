"""
Evaluator 工厂类。

提供统一的 Evaluator 创建接口，支持多种评估后端动态注册和实例化。
"""

from typing import Type

from core import Settings, EvaluationSettings

from .base_evaluator import BaseEvaluator, EvaluatorError
from .custom_evaluator import CustomEvaluator


class EvaluatorFactory:
    """Evaluator 工厂类。
    
    使用注册表模式管理多种 Evaluator 后端实现。
    支持运行时动态注册新的后端类型。
    
    使用示例:
        # 注册后端
        EvaluatorFactory.register("custom", CustomEvaluator)
        EvaluatorFactory.register("ragas", RagasEvaluator)
        
        # 从配置创建
        settings = Settings(...)
        evaluator = EvaluatorFactory.create(settings)
        
        # 直接创建
        evaluator = EvaluatorFactory.create_with_params(
            provider="custom",
            metrics=["hit_rate", "mrr"]
        )
    """
    
    # 后端注册表：provider_name -> Evaluator 类
    _registry: dict[str, Type[BaseEvaluator]] = {}
    
    @classmethod
    def register(cls, provider: str, evaluator_class: Type[BaseEvaluator]) -> None:
        """注册 Evaluator 后端。
        
        Args:
            provider: 后端标识符（如 "custom", "ragas", "deepeval"）
            evaluator_class: Evaluator 类（必须继承自 BaseEvaluator）
            
        Raises:
            ValueError: 如果后端已注册或类型不合法
        """
        provider_lower = provider.lower()
        
        # 检查是否已注册
        if provider_lower in cls._registry:
            raise ValueError(
                f"Evaluator provider '{provider}' is already registered. "
                f"Use a different name or unregister the existing one first."
            )
        
        # 检查类型
        if not issubclass(evaluator_class, BaseEvaluator):
            raise TypeError(
                f"Evaluator class must inherit from BaseEvaluator, "
                f"got {evaluator_class.__name__}"
            )
        
        cls._registry[provider_lower] = evaluator_class
    
    @classmethod
    def create_from_evaluation_settings(
        cls,
        evaluation_settings: EvaluationSettings
    ) -> BaseEvaluator:
        """从 EvaluationSettings 创建 Evaluator 实例。
        
        Args:
            evaluation_settings: 评估配置对象
            
        Returns:
            Evaluator 实例
            
        Raises:
            EvaluatorError: 后端未注册或创建失败
        """
        provider = evaluation_settings.provider.lower()
        
        # 检查后端是否注册
        if provider not in cls._registry:
            raise EvaluatorError(
                f"Unknown Evaluator provider: '{evaluation_settings.provider}'. "
                f"Available providers: {', '.join(cls.list_providers())}"
            )
        
        evaluator_class = cls._registry[provider]
        
        try:
            # 传递配置参数（移除 provider 字段）
            config = {
                k: v
                for k, v in evaluation_settings.__dict__.items()
                if k != "provider"
            }
            return evaluator_class(**config)
        except Exception as e:
            raise EvaluatorError(
                f"Failed to create Evaluator with provider '{evaluation_settings.provider}': {e}"
            ) from e
    
    @classmethod
    def create(cls, settings: Settings) -> BaseEvaluator:
        """从完整 Settings 创建 Evaluator 实例。
        
        Args:
            settings: 完整配置对象
            
        Returns:
            Evaluator 实例
        """
        return cls.create_from_evaluation_settings(settings.evaluation)
    
    @classmethod
    def create_with_params(cls, provider: str, **kwargs) -> BaseEvaluator:
        """使用指定参数创建 Evaluator 实例。
        
        Args:
            provider: 后端标识符
            **kwargs: 后端特定配置参数（如 metrics, k 等）
            
        Returns:
            Evaluator 实例
            
        Raises:
            EvaluatorError: 后端未注册或创建失败
        """
        provider_lower = provider.lower()
        
        # 检查后端是否注册
        if provider_lower not in cls._registry:
            raise EvaluatorError(
                f"Unknown Evaluator provider: '{provider}'. "
                f"Available providers: {', '.join(cls.list_providers())}"
            )
        
        evaluator_class = cls._registry[provider_lower]
        
        try:
            return evaluator_class(**kwargs)
        except Exception as e:
            raise EvaluatorError(
                f"Failed to create Evaluator with provider '{provider}': {e}"
            ) from e
    
    @classmethod
    def is_registered(cls, provider: str) -> bool:
        """检查后端是否已注册。
        
        Args:
            provider: 后端标识符
            
        Returns:
            True 如果已注册，否则 False
        """
        return provider.lower() in cls._registry
    
    @classmethod
    def list_providers(cls) -> list[str]:
        """列出所有已注册的后端。
        
        Returns:
            后端标识符列表
        """
        return list(cls._registry.keys())
    
    @classmethod
    def clear_registry(cls) -> None:
        """清空注册表（主要用于测试）。"""
        cls._registry.clear()
