"""
Local Embedding 实现模块。

此模块实现本地 Embedding 提供商（占位实现）。
当前使用确定性的假向量生成，保证链路可跑。
未来可扩展为真实的本地模型（如 BGE、Sentence-Transformers、Ollama）。
"""

import hashlib
from typing import List, Optional

from .base_embedding import BaseEmbedding, EmbeddingResponse, EmbeddingError


class LocalEmbedding(BaseEmbedding):
    """
    Local Embedding 实现类（占位实现）。
    
    当前使用确定性的假向量生成策略：
    - 基于文本内容的哈希生成稳定向量
    - 保证相同输入产生相同输出
    - 满足 BaseEmbedding 接口契约
    
    未来扩展方向：
    - 集成 Sentence-Transformers (BGE, MiniLM, etc.)
    - 支持 Ollama embedding 端点
    - 支持 ONNX 模型加载
    """
    
    def __init__(
        self,
        model: str = "fake-local-embedding",
        dimensions: int = 768,
        batch_size: int = 100,
        **kwargs
    ):
        """
        初始化 Local Embedding 实例。
        
        Args:
            model: 模型名称（当前为占位名称）
            dimensions: 向量维度（默认 768，常见本地模型维度）
            batch_size: 批处理大小
            **kwargs: 其他参数（预留用于真实模型配置）
        """
        super().__init__(
            model=model,
            dimensions=dimensions,
            batch_size=batch_size,
            **kwargs
        )
        
        # 验证维度必须是正整数
        if dimensions <= 0:
            raise EmbeddingError(
                f"Dimensions must be positive, got {dimensions}"
            )
    
    def embed(
        self,
        texts: List[str],
        **kwargs
    ) -> EmbeddingResponse:
        """
        将文本列表转换为嵌入向量（使用确定性假向量）。
        
        当前实现策略：
        - 使用文本内容的 SHA-256 哈希生成种子
        - 基于种子生成确定性的归一化向量
        - 保证相同文本产生相同向量
        
        Args:
            texts: 要嵌入的文本列表
            **kwargs: 可选的请求参数
            
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
        if len(texts) > self.batch_size:
            raise EmbeddingError(
                f"Batch size {len(texts)} exceeds maximum {self.batch_size}. "
                f"Please split into smaller batches."
            )
        
        # 生成确定性假向量
        try:
            embeddings = []
            for text in texts:
                vector = self._generate_deterministic_vector(text)
                embeddings.append(vector)
            
            return EmbeddingResponse(
                embeddings=embeddings,
                model=self.model,
                dimensions=self.dimensions,
                usage=None  # 本地模型不产生 token 使用信息
            )
        
        except Exception as e:
            raise EmbeddingError(
                f"Unexpected error during local embedding: {str(e)}"
            )
    
    def _generate_deterministic_vector(self, text: str) -> List[float]:
        """
        基于文本内容生成确定性的假向量。
        
        使用 SHA-256 哈希生成种子，然后生成归一化向量。
        保证相同文本产生相同向量，不同文本产生不同向量。
        
        Args:
            text: 输入文本
            
        Returns:
            List[float]: 归一化的向量（长度为 self.dimensions）
        """
        # 使用 SHA-256 哈希作为种子
        hash_bytes = hashlib.sha256(text.encode('utf-8')).digest()
        
        # 将哈希字节转换为浮点数向量
        vector = []
        for i in range(self.dimensions):
            # 循环使用哈希字节
            byte_index = i % len(hash_bytes)
            # 将字节值 (0-255) 转换为 (-1, 1) 范围的浮点数
            value = (hash_bytes[byte_index] / 255.0) * 2 - 1
            vector.append(value)
        
        # L2 归一化（使向量长度为 1）
        magnitude = sum(x * x for x in vector) ** 0.5
        if magnitude > 0:
            vector = [x / magnitude for x in vector]
        
        return vector
