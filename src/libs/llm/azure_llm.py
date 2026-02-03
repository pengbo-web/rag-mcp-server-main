"""
Azure OpenAI LLM 实现模块。

此模块实现基于 Azure OpenAI Service 的 LLM 提供商。
支持企业级 Azure 云端部署，符合合规与安全要求。
"""

import os
from typing import List, Optional

try:
    from openai import AzureOpenAI
    from openai import OpenAIError, APIError, APIConnectionError, RateLimitError, AuthenticationError
except ImportError:
    raise ImportError(
        "OpenAI package not found. Please install: pip install openai"
    )

from .base_llm import BaseLLM, Message, ChatResponse, LLMError


class AzureLLM(BaseLLM):
    """
    Azure OpenAI LLM 实现类。
    
    支持 Microsoft Azure OpenAI Service，提供企业级合规性。
    需要配置 Azure 特定的 endpoint、api_key 和 deployment_name。
    """
    
    def __init__(
        self,
        model: str,  # 在 Azure 中这通常是 deployment_name
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,  # Azure endpoint
        api_version: str = "2024-02-15-preview",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: int = 30,
        **kwargs
    ):
        """
        初始化 Azure OpenAI LLM 实例。
        
        Args:
            model: Azure 部署名称（deployment_name）
            api_key: Azure OpenAI API 密钥（若为 None 则从环境变量读取）
            base_url: Azure OpenAI endpoint（如 "https://xxx.openai.azure.com"）
            api_version: API 版本
            temperature: 生成温度（0-1）
            max_tokens: 最大生成 token 数
            timeout: 请求超时时间（秒）
            **kwargs: 其他参数
        
        Raises:
            LLMError: 当必需配置缺失时
        """
        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            **kwargs
        )
        
        # 处理 API 密钥
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        if not self.api_key:
            raise LLMError(
                "Azure OpenAI API key is required. "
                "Set AZURE_OPENAI_API_KEY environment variable or pass api_key parameter."
            )
        
        # 处理 Azure endpoint
        self.azure_endpoint = base_url or os.getenv("AZURE_OPENAI_ENDPOINT")
        if not self.azure_endpoint:
            raise LLMError(
                "Azure OpenAI endpoint is required. "
                "Set AZURE_OPENAI_ENDPOINT environment variable or pass base_url parameter."
            )
        
        self.api_version = api_version
        
        # 初始化 Azure OpenAI 客户端
        try:
            self.client = AzureOpenAI(
                api_key=self.api_key,
                api_version=self.api_version,
                azure_endpoint=self.azure_endpoint,
                timeout=timeout,
            )
        except Exception as e:
            raise LLMError(f"Failed to initialize Azure OpenAI client: {str(e)}")
    
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
            "model": self.model,  # 在 Azure 中这是 deployment_name
            "messages": [msg.to_dict() for msg in messages],
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        
        # 调用 Azure OpenAI API
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
                f"Azure OpenAI authentication failed. Please check your API key and endpoint. "
                f"Error: {str(e)}"
            )
        except RateLimitError as e:
            raise LLMError(
                f"Azure OpenAI rate limit exceeded. Please try again later. "
                f"Error: {str(e)}"
            )
        except APIConnectionError as e:
            raise LLMError(
                f"Failed to connect to Azure OpenAI API. Please check your network and endpoint. "
                f"Error: {str(e)}"
            )
        except APIError as e:
            raise LLMError(
                f"Azure OpenAI API error: {str(e)}"
            )
        except OpenAIError as e:
            raise LLMError(
                f"Azure OpenAI SDK error: {str(e)}"
            )
        except Exception as e:
            raise LLMError(
                f"Unexpected error during Azure OpenAI API call: {str(e)}"
            )
