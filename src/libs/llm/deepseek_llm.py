"""
DeepSeek LLM 实现模块。

此模块实现基于 DeepSeek API 的 LLM 提供商。
DeepSeek API 兼容 OpenAI 格式，可复用 OpenAI 实现。
"""

import os
from typing import List, Optional

from .openai_llm import OpenAILLM
from .base_llm import Message, ChatResponse, LLMError


class DeepSeekLLM(OpenAILLM):
    """
    DeepSeek LLM 实现类。
    
    DeepSeek API 兼容 OpenAI 格式，因此直接继承 OpenAILLM。
    主要差异在于默认的 base_url 和环境变量名称。
    """
    
    # DeepSeek API 默认端点
    DEFAULT_BASE_URL = "https://api.deepseek.com"
    
    def __init__(
        self,
        model: str = "deepseek-chat",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: int = 30,
        **kwargs
    ):
        """
        初始化 DeepSeek LLM 实例。
        
        Args:
            model: 模型名称（默认 "deepseek-chat"）
            api_key: DeepSeek API 密钥（若为 None 则从环境变量读取）
            base_url: 自定义 API 端点（默认使用 DeepSeek 官方端点）
            temperature: 生成温度（0-1）
            max_tokens: 最大生成 token 数
            timeout: 请求超时时间（秒）
            **kwargs: 其他参数
        
        Raises:
            LLMError: 当 API 密钥缺失时
        """
        # 优先使用传入的 api_key，其次尝试 DEEPSEEK_API_KEY 环境变量
        resolved_api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        
        # 如果还是没有，再尝试通用的 OPENAI_API_KEY（某些用户可能统一配置）
        if not resolved_api_key:
            resolved_api_key = os.getenv("OPENAI_API_KEY")
        
        if not resolved_api_key:
            raise LLMError(
                "DeepSeek API key is required. "
                "Set DEEPSEEK_API_KEY or OPENAI_API_KEY environment variable, "
                "or pass api_key parameter."
            )
        
        # 使用 DeepSeek 默认端点（如果没有指定）
        resolved_base_url = base_url or self.DEFAULT_BASE_URL
        
        # 调用父类构造函数
        super().__init__(
            model=model,
            api_key=resolved_api_key,
            base_url=resolved_base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            **kwargs
        )
    
    # chat 方法完全继承自 OpenAILLM，无需重写
