"""
OpenAI LLM 实现模块。

此模块实现基于 OpenAI API 的 LLM 提供商。
支持官方 OpenAI API 及兼容格式的其他提供商。
"""

import os
from typing import List, Optional

try:
    from openai import OpenAI
    from openai import OpenAIError, APIError, APIConnectionError, RateLimitError, AuthenticationError
except ImportError:
    raise ImportError(
        "OpenAI package not found. Please install: pip install openai"
    )

from .base_llm import BaseLLM, Message, ChatResponse, LLMError


class OpenAILLM(BaseLLM):
    """
    OpenAI LLM 实现类。
    
    支持官方 OpenAI API 及其他 OpenAI 兼容格式的提供商。
    使用 openai>=1.0.0 的新版 SDK。
    """
    
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: int = 30,
        **kwargs
    ):
        """
        初始化 OpenAI LLM 实例。
        
        Args:
            model: 模型名称（如 "gpt-4o-mini", "gpt-4"）
            api_key: OpenAI API 密钥（若为 None 则从环境变量读取）
            base_url: 自定义 API 端点（可选，用于代理或兼容服务）
            temperature: 生成温度（0-1）
            max_tokens: 最大生成 token 数
            timeout: 请求超时时间（秒）
            **kwargs: 其他参数
        
        Raises:
            LLMError: 当 API 密钥缺失时
        """
        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            **kwargs
        )
        
        # 处理 API 密钥：优先使用传入值，其次从环境变量读取
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise LLMError(
                "OpenAI API key is required. "
                "Set OPENAI_API_KEY environment variable or pass api_key parameter."
            )
        
        # 初始化 OpenAI 客户端
        client_kwargs = {
            "api_key": self.api_key,
            "timeout": timeout,
        }
        if base_url:
            client_kwargs["base_url"] = base_url
        
        try:
            self.client = OpenAI(**client_kwargs)
        except Exception as e:
            raise LLMError(f"Failed to initialize OpenAI client: {str(e)}")
    
    def chat(
        self,
        messages: List[Message],
        **kwargs
    ) -> ChatResponse:
        """
        执行聊天补全。
        
        Args:
            messages: 对话消息列表
            **kwargs: 可选的请求参数（如 temperature 覆盖）
            
        Returns:
            ChatResponse: LLM 响应对象
            
        Raises:
            LLMError: 当 API 调用失败时抛出，包含详细错误信息
        """
        # 验证消息列表
        if not messages:
            raise LLMError("Messages list cannot be empty")
        
        # 验证消息格式
        for i, msg in enumerate(messages):
            if not isinstance(msg, Message):
                raise LLMError(
                    f"Message at index {i} must be a Message object, got {type(msg)}"
                )
            if msg.role not in ["system", "user", "assistant"]:
                raise LLMError(
                    f"Message at index {i} has invalid role: '{msg.role}'. "
                    "Must be 'system', 'user', or 'assistant'."
                )
        
        # 准备请求参数
        request_params = {
            "model": self.model,
            "messages": [msg.to_dict() for msg in messages],
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        
        # 调用 OpenAI API
        try:
            response = self.client.chat.completions.create(**request_params)
            
            # 提取响应内容
            content = response.choices[0].message.content
            if content is None:
                content = ""
            
            # 提取使用信息
            usage = None
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
            
            return ChatResponse(
                content=content,
                model=response.model,
                usage=usage,
                raw_response=response
            )
        
        except AuthenticationError as e:
            raise LLMError(
                f"OpenAI authentication failed. Please check your API key. "
                f"Error: {str(e)}"
            )
        except RateLimitError as e:
            raise LLMError(
                f"OpenAI rate limit exceeded. Please try again later. "
                f"Error: {str(e)}"
            )
        except APIConnectionError as e:
            raise LLMError(
                f"Failed to connect to OpenAI API. Please check your network. "
                f"Error: {str(e)}"
            )
        except APIError as e:
            raise LLMError(
                f"OpenAI API error: {str(e)}"
            )
        except OpenAIError as e:
            raise LLMError(
                f"OpenAI SDK error: {str(e)}"
            )
        except Exception as e:
            raise LLMError(
                f"Unexpected error during OpenAI API call: {str(e)}"
            )
