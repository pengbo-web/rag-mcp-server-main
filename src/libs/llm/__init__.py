"""
LLM 库模块。

提供统一的 LLM 抽象接口和工厂，支持多种提供商。
"""

from .base_llm import BaseLLM, Message, ChatResponse, LLMError
from .llm_factory import LLMFactory
from .openai_llm import OpenAILLM
from .azure_llm import AzureLLM
from .deepseek_llm import DeepSeekLLM

# 注册提供商到工厂
LLMFactory.register("openai", OpenAILLM)
LLMFactory.register("azure", AzureLLM)
LLMFactory.register("deepseek", DeepSeekLLM)

__all__ = [
    "BaseLLM",
    "Message",
    "ChatResponse",
    "LLMError",
    "LLMFactory",
    "OpenAILLM",
    "AzureLLM",
    "DeepSeekLLM",
]
