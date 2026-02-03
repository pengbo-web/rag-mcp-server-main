"""
OpenAI Embedding 实现模块。

此模块实现基于 OpenAI Embedding API 的向量化提供商。
支持 text-embedding-3-small/large、text-embedding-ada-002 等模型。
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

from .base_embedding import BaseEmbedding, EmbeddingResponse, EmbeddingError


class OpenAIEmbedding(BaseEmbedding):
    """
    OpenAI Embedding 实现类。
    
    支持 OpenAI 官方的文本嵌入模型：
    - text-embedding-3-small (1536 维，性能优化)
    - text-embedding-3-large (3072 维，最高质量)
    - text-embedding-ada-002 (1536 维，传统模型)
    """
    
    # 模型默认维度映射
    DEFAULT_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }
    
    # OpenAI API 输入限制
    MAX_INPUT_LENGTH = 8191  # tokens
    MAX_BATCH_SIZE = 2048    # 每批最大文本数
    
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        dimensions: Optional[int] = None,
        batch_size: int = 100,
        timeout: int = 30,
        **kwargs
    ):
        """
        初始化 OpenAI Embedding 实例。
        
        Args:
            model: 模型名称（默认 text-embedding-3-small）
            api_key: OpenAI API 密钥（若为 None 则从环境变量读取）
            base_url: 自定义 API 端点（可选，用于代理或兼容服务）
            dimensions: 向量维度（若为 None 则使用模型默认维度）
            batch_size: 批处理大小（每次 API 调用的文本数）
            timeout: 请求超时时间（秒）
            **kwargs: 其他参数
        
        Raises:
            EmbeddingError: 当 API 密钥缺失或配置无效时
        """
        # 确定向量维度
        if dimensions is None:
            dimensions = self.DEFAULT_DIMENSIONS.get(model, 1536)
        
        super().__init__(
            model=model,
            dimensions=dimensions,
            batch_size=min(batch_size, self.MAX_BATCH_SIZE),
            **kwargs
        )
        
        # 处理 API 密钥
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise EmbeddingError(
                "OpenAI API key is required. "
                "Set OPENAI_API_KEY environment variable or pass api_key parameter."
            )
        
        self.timeout = timeout
        
        # 初始化 OpenAI 客户端
        client_kwargs = {
            "api_key": self.api_key,
            "timeout": timeout,
        }
        if base_url:
            client_kwargs["base_url"] = base_url
            self.base_url = base_url
        else:
            self.base_url = "https://api.openai.com/v1"
        
        try:
            self.client = OpenAI(**client_kwargs)
        except Exception as e:
            raise EmbeddingError(
                f"Failed to initialize OpenAI client: {str(e)}"
            )
    
    def embed(
        self,
        texts: List[str],
        **kwargs
    ) -> EmbeddingResponse:
        """
        将文本列表转换为嵌入向量。
        
        Args:
            texts: 要嵌入的文本列表
            **kwargs: 可选的请求参数（如 dimensions 覆盖）
            
        Returns:
            EmbeddingResponse: 包含嵌入向量的响应对象
            
        Raises:
            EmbeddingError: 当调用失败时抛出
        """
        # 验证输入
        if not texts:
            raise EmbeddingError("Text list cannot be empty")
        
        if not isinstance(texts, list):
            raise EmbeddingError(
                f"Texts must be a list, got {type(texts)}"
            )
        
        # 验证文本类型
        for i, text in enumerate(texts):
            if not isinstance(text, str):
                raise EmbeddingError(
                    f"Text at index {i} must be a string, got {type(text)}"
                )
            if not text.strip():
                raise EmbeddingError(
                    f"Text at index {i} is empty or whitespace-only"
                )
        
        # 检查批量大小
        if len(texts) > self.MAX_BATCH_SIZE:
            raise EmbeddingError(
                f"Batch size {len(texts)} exceeds maximum {self.MAX_BATCH_SIZE}. "
                f"Please split into smaller batches."
            )
        
        # 准备请求参数
        request_params = {
            "input": texts,
            "model": self.model,
        }
        
        # 添加可选的维度参数（仅 v3 模型支持）
        dimensions = kwargs.get("dimensions", self.dimensions)
        if "text-embedding-3" in self.model:
            request_params["dimensions"] = dimensions
        
        # 调用 OpenAI Embedding API
        try:
            response = self.client.embeddings.create(**request_params)
            
            # 提取嵌入向量
            embeddings = [item.embedding for item in response.data]
            
            # 提取使用信息
            usage = None
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
            
            return EmbeddingResponse(
                embeddings=embeddings,
                model=response.model,
                dimensions=len(embeddings[0]) if embeddings else dimensions,
                usage=usage
            )
        
        except AuthenticationError as e:
            raise EmbeddingError(
                f"OpenAI authentication failed. Please check your API key. "
                f"Error: {str(e)}"
            )
        except RateLimitError as e:
            raise EmbeddingError(
                f"OpenAI rate limit exceeded. Please retry later or upgrade your plan. "
                f"Error: {str(e)}"
            )
        except APIConnectionError as e:
            raise EmbeddingError(
                f"Failed to connect to OpenAI API at {self.base_url}. "
                f"Please check your network connection. "
                f"Error: {str(e)}"
            )
        except APIError as e:
            # 检查是否是输入过长错误
            error_msg = str(e).lower()
            if "maximum context length" in error_msg or "too long" in error_msg:
                raise EmbeddingError(
                    f"Input text exceeds model's maximum context length ({self.MAX_INPUT_LENGTH} tokens). "
                    f"Please split the text into smaller chunks. "
                    f"Error: {str(e)}"
                )
            else:
                raise EmbeddingError(f"OpenAI API error: {str(e)}")
        except OpenAIError as e:
            raise EmbeddingError(f"OpenAI SDK error: {str(e)}")
        except Exception as e:
            raise EmbeddingError(
                f"Unexpected error during embedding: {str(e)}"
            )
