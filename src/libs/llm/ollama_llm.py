"""
Ollama LLM 实现模块。

此模块实现基于 Ollama 本地服务的 LLM 提供商。
Ollama 是一个运行大型语言模型的本地服务，支持完全离线部署。
"""

import os
from typing import List, Optional

try:
    from openai import OpenAI
    from openai import OpenAIError, APIError, APIConnectionError, APITimeoutError
except ImportError:
    raise ImportError(
        "OpenAI package not found. Please install: pip install openai"
    )

from .base_llm import BaseLLM, Message, ChatResponse, LLMError


class OllamaLLM(BaseLLM):
    """
    Ollama LLM 实现类。
    
    Ollama 是本地运行的 LLM 服务，提供 OpenAI 兼容的 API 接口。
    适用于完全离线、隐私敏感或零 API 成本的场景。
    """
    
    # Ollama 默认本地端点
    DEFAULT_BASE_URL = "http://localhost:11434/v1"
    
    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = "ollama",  # Ollama 不需要真实 API key
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: int = 30,
        **kwargs
    ):
        """
        初始化 Ollama LLM 实例。
        
        Args:
            model: 模型名称（如 "llama2", "mistral", "qwen"）
            base_url: Ollama 服务地址（默认 http://localhost:11434/v1）
            api_key: API 密钥（Ollama 不需要，设置占位值即可）
            temperature: 生成温度（0-1）
            max_tokens: 最大生成 token 数
            timeout: 请求超时时间（秒）
            **kwargs: 其他参数
        
        Raises:
            LLMError: 当初始化失败时
        """
        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            **kwargs
        )
        
        # 使用默认本地端点（如果没有指定）
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL") or self.DEFAULT_BASE_URL
        
        # Ollama 不需要真实 API key，但 OpenAI SDK 需要一个非空值
        self.api_key = api_key or "ollama"
        
        # 初始化 OpenAI 客户端（Ollama 提供 OpenAI 兼容接口）
        try:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=timeout,
            )
        except Exception as e:
            raise LLMError(
                f"Failed to initialize Ollama client. "
                f"Please ensure Ollama is running at {self.base_url}. "
                f"Error: {str(e)}"
            )
    
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
        
        # 调用 Ollama API
        try:
            response = self.client.chat.completions.create(**request_params)
            
            # 提取响应内容
            content = response.choices[0].message.content
            if content is None:
                content = ""
            
            # 提取使用信息（Ollama 可能不返回此信息）
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
        
        except APITimeoutError as e:
            raise LLMError(
                f"Ollama API request timed out after {self.timeout}s. "
                f"The model '{self.model}' may be too large or the service is overloaded. "
                f"Error: {str(e)}"
            )
        except APIConnectionError as e:
            raise LLMError(
                f"Failed to connect to Ollama service at {self.base_url}. "
                f"Please ensure Ollama is running. "
                f"You can start it with: ollama serve. "
                f"Error: {str(e)}"
            )
        except APIError as e:
            # 检查是否是模型不存在的错误
            error_msg = str(e).lower()
            if "not found" in error_msg or "model" in error_msg:
                raise LLMError(
                    f"Model '{self.model}' not found in Ollama. "
                    f"Please pull the model first: ollama pull {self.model}. "
                    f"Error: {str(e)}"
                )
            else:
                raise LLMError(f"Ollama API error: {str(e)}")
        except OpenAIError as e:
            raise LLMError(f"Ollama SDK error: {str(e)}")
        except Exception as e:
            raise LLMError(
                f"Unexpected error during Ollama API call: {str(e)}"
            )
