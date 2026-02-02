"""
LLM 工厂模块。

此模块实现工厂模式，根据配置动态创建相应的 LLM 实例。
支持多种提供商：OpenAI、Azure、Ollama、DeepSeek 等。
"""

from typing import Dict, Type

from core import Settings, LLMSettings
from .base_llm import BaseLLM, LLMError


class LLMFactory:
    """
    LLM 工厂类。
    
    负责根据配置创建对应的 LLM 提供商实例。
    采用注册表模式，支持动态扩展新的提供商。
    """
    
    # 提供商注册表：provider_name -> LLM 实现类
    _registry: Dict[str, Type[BaseLLM]] = {}
    
    @classmethod
    def register(cls, provider: str, llm_class: Type[BaseLLM]) -> None:
        """
        注册新的 LLM 提供商。
        
        Args:
            provider: 提供商名称（如 "openai", "ollama"）
            llm_class: LLM 实现类（必须继承 BaseLLM）
            
        Raises:
            ValueError: 如果提供商名称已被注册
        """
        if provider in cls._registry:
            raise ValueError(f"Provider '{provider}' is already registered")
        
        if not issubclass(llm_class, BaseLLM):
            raise TypeError(f"{llm_class} must inherit from BaseLLM")
        
        cls._registry[provider] = llm_class
    
    @classmethod
    def create(cls, settings: Settings) -> BaseLLM:
        """
        根据配置创建 LLM 实例。
        
        Args:
            settings: 全局配置对象，包含 LLM 设置
            
        Returns:
            BaseLLM: 对应提供商的 LLM 实例
            
        Raises:
            LLMError: 当提供商未注册或创建失败时抛出
        """
        llm_settings = settings.llm
        return cls.create_from_llm_settings(llm_settings)
    
    @classmethod
    def create_from_llm_settings(cls, llm_settings: LLMSettings) -> BaseLLM:
        """
        从 LLM 设置直接创建实例（用于测试和独立场景）。
        
        Args:
            llm_settings: LLM 配置对象
            
        Returns:
            BaseLLM: 对应提供商的 LLM 实例
            
        Raises:
            LLMError: 当提供商未注册或创建失败时抛出
        """
        provider = llm_settings.provider.lower()
        
        if provider not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise LLMError(
                f"Unknown LLM provider: '{provider}'. "
                f"Available providers: {available or 'none (no providers registered)'}"
            )
        
        llm_class = cls._registry[provider]
        
        try:
            # 构造参数
            init_params = {
                "model": llm_settings.model,
                "temperature": llm_settings.temperature,
                "max_tokens": llm_settings.max_tokens,
                "timeout": llm_settings.timeout,
            }
            
            # 添加可选参数
            if llm_settings.api_key:
                init_params["api_key"] = llm_settings.api_key
            if llm_settings.base_url:
                init_params["base_url"] = llm_settings.base_url
            
            return llm_class(**init_params)
        
        except Exception as e:
            raise LLMError(
                f"Failed to create LLM instance for provider '{provider}': {e}"
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
