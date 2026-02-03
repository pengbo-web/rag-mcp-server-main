"""
Embedding 库模块。

提供统一的 Embedding 抽象接口和工厂，支持多种提供商。
"""

from .base_embedding import BaseEmbedding, EmbeddingResponse, EmbeddingError
from .embedding_factory import EmbeddingFactory
from .openai_embedding import OpenAIEmbedding

# 注册提供商到工厂
EmbeddingFactory.register("openai", OpenAIEmbedding)

__all__ = [
    "BaseEmbedding",
    "EmbeddingResponse",
    "EmbeddingError",
    "EmbeddingFactory",
    "OpenAIEmbedding",
]
