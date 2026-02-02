"""
LLM 抽象基类模块。

此模块定义了所有 LLM 提供商必须实现的抽象接口。
支持可插拔架构，允许在不同的 LLM 后端之间无缝切换。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Message:
    """
    对话消息数据类。
    
    Attributes:
        role: 消息角色（system/user/assistant）
        content: 消息内容
    """
    role: str  # "system" | "user" | "assistant"
    content: str
    
    def to_dict(self) -> Dict[str, str]:
        """转换为字典格式。"""
        return {"role": self.role, "content": self.content}


@dataclass
class ChatResponse:
    """
    LLM 响应数据类。
    
    Attributes:
        content: 生成的文本内容
        model: 使用的模型名称
        usage: token 使用信息（可选）
        raw_response: 原始响应对象（用于调试）
    """
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    raw_response: Optional[Any] = None


class LLMError(Exception):
    """LLM 调用过程中发生的错误。"""
    pass


class BaseLLM(ABC):
    """
    LLM 抽象基类。
    
    所有 LLM 提供商实现都必须继承此类并实现 chat 方法。
    此设计支持可插拔架构，允许通过配置切换不同的 LLM 后端。
    """
    
    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: int = 30,
        **kwargs
    ):
        """
        初始化 LLM 实例。
        
        Args:
            model: 模型名称（如 "gpt-4", "deepseek-chat"）
            temperature: 生成温度（0-1），控制随机性
            max_tokens: 最大生成 token 数
            timeout: 请求超时时间（秒）
            **kwargs: 其他提供商特定参数
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.extra_params = kwargs
    
    @abstractmethod
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
            LLMError: 当调用失败时抛出
        """
        pass
    
    def chat_simple(self, messages: List[Message], **kwargs) -> str:
        """
        简化的聊天接口，直接返回文本内容。
        
        Args:
            messages: 对话消息列表
            **kwargs: 可选的请求参数
            
        Returns:
            str: 生成的文本内容
            
        Raises:
            LLMError: 当调用失败时抛出
        """
        response = self.chat(messages, **kwargs)
        return response.content
    
    def __repr__(self) -> str:
        """返回实例的字符串表示。"""
        return f"{self.__class__.__name__}(model={self.model})"
