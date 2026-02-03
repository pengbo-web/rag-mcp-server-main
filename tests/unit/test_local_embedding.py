"""
Local Embedding 单元测试模块。

此模块测试 Local Embedding 提供商的核心功能，包括：
- 基本初始化
- 确定性向量生成
- 批量处理
- 错误处理
- 向量归一化验证
- 工厂集成
"""

import pytest

from src.libs.embedding.local_embedding import LocalEmbedding
from src.libs.embedding.base_embedding import EmbeddingError
from src.libs.embedding.embedding_factory import EmbeddingFactory


class TestLocalEmbeddingInitialization:
    """测试 Local Embedding 初始化行为。"""
    
    def test_initialization_with_defaults(self):
        """测试使用默认参数初始化。"""
        embedding = LocalEmbedding()
        
        assert embedding.model == "fake-local-embedding"
        assert embedding.dimensions == 768
        assert embedding.batch_size == 100
    
    def test_initialization_with_custom_dimensions(self):
        """测试自定义向量维度。"""
        embedding = LocalEmbedding(
            model="custom-local",
            dimensions=384,
            batch_size=50
        )
        
        assert embedding.model == "custom-local"
        assert embedding.dimensions == 384
        assert embedding.batch_size == 50
    
    def test_initialization_with_invalid_dimensions_raises_error(self):
        """测试无效维度抛出错误。"""
        with pytest.raises(EmbeddingError, match="must be positive"):
            LocalEmbedding(dimensions=0)
        
        with pytest.raises(EmbeddingError, match="must be positive"):
            LocalEmbedding(dimensions=-10)


class TestLocalEmbeddingEmbed:
    """测试 Local Embedding 嵌入功能。"""
    
    def test_embed_success(self):
        """测试成功的嵌入调用。"""
        embedding = LocalEmbedding(dimensions=128)
        texts = ["hello world", "test text"]
        response = embedding.embed(texts)
        
        # 验证响应结构
        assert len(response.embeddings) == 2
        assert response.model == "fake-local-embedding"
        assert response.dimensions == 128
        assert response.usage is None  # 本地模型不返回 usage
        
        # 验证向量维度
        for vector in response.embeddings:
            assert len(vector) == 128
            assert all(isinstance(v, float) for v in vector)
    
    def test_embed_single(self):
        """测试单个文本嵌入。"""
        embedding = LocalEmbedding(dimensions=64)
        vector = embedding.embed_single("hello")
        
        assert len(vector) == 64
        assert all(isinstance(v, float) for v in vector)
    
    def test_embed_deterministic(self):
        """测试嵌入的确定性（相同输入产生相同输出）。"""
        embedding = LocalEmbedding(dimensions=256)
        text = "deterministic test"
        
        # 多次嵌入相同文本
        vector1 = embedding.embed_single(text)
        vector2 = embedding.embed_single(text)
        vector3 = embedding.embed_single(text)
        
        # 验证向量完全相同
        assert vector1 == vector2
        assert vector2 == vector3
    
    def test_embed_different_texts_produce_different_vectors(self):
        """测试不同文本产生不同向量。"""
        embedding = LocalEmbedding(dimensions=128)
        
        vector1 = embedding.embed_single("hello")
        vector2 = embedding.embed_single("world")
        vector3 = embedding.embed_single("test")
        
        # 验证向量不同
        assert vector1 != vector2
        assert vector2 != vector3
        assert vector1 != vector3
    
    def test_embed_vectors_are_normalized(self):
        """测试向量已归一化（L2 范数为 1）。"""
        embedding = LocalEmbedding(dimensions=100)
        texts = ["hello", "world", "test", "normalized vectors"]
        response = embedding.embed(texts)
        
        # 验证每个向量的 L2 范数接近 1
        for vector in response.embeddings:
            magnitude = sum(x * x for x in vector) ** 0.5
            assert abs(magnitude - 1.0) < 1e-6  # 允许浮点误差
    
    def test_embed_batch_processing(self):
        """测试批量处理。"""
        embedding = LocalEmbedding(dimensions=64, batch_size=50)
        
        # 创建批量文本
        texts = [f"text {i}" for i in range(20)]
        response = embedding.embed(texts)
        
        assert len(response.embeddings) == 20
        assert all(len(v) == 64 for v in response.embeddings)
    
    def test_embed_consistent_across_instances(self):
        """测试不同实例产生相同结果。"""
        text = "consistency test"
        
        emb1 = LocalEmbedding(dimensions=128)
        emb2 = LocalEmbedding(dimensions=128)
        
        vector1 = emb1.embed_single(text)
        vector2 = emb2.embed_single(text)
        
        # 相同配置的不同实例应产生相同向量
        assert vector1 == vector2
    
    def test_embed_various_dimensions(self):
        """测试不同维度配置。"""
        text = "dimension test"
        
        for dim in [64, 128, 256, 384, 512, 768, 1024, 1536]:
            embedding = LocalEmbedding(dimensions=dim)
            vector = embedding.embed_single(text)
            assert len(vector) == dim


class TestLocalEmbeddingErrorHandling:
    """测试 Local Embedding 错误处理。"""
    
    def test_embed_empty_list_raises_error(self):
        """测试空文本列表的错误处理。"""
        embedding = LocalEmbedding()
        
        with pytest.raises(EmbeddingError, match="cannot be empty"):
            embedding.embed([])
    
    def test_embed_non_list_raises_error(self):
        """测试非列表输入的错误处理。"""
        embedding = LocalEmbedding()
        
        with pytest.raises(EmbeddingError, match="must be a list"):
            embedding.embed("not a list")
    
    def test_embed_non_string_item_raises_error(self):
        """测试非字符串元素的错误处理。"""
        embedding = LocalEmbedding()
        
        with pytest.raises(EmbeddingError, match="must be a string"):
            embedding.embed(["hello", 123])
    
    def test_embed_empty_string_raises_error(self):
        """测试空字符串的错误处理。"""
        embedding = LocalEmbedding()
        
        with pytest.raises(EmbeddingError, match="empty or whitespace-only"):
            embedding.embed(["hello", "  ", "world"])
    
    def test_embed_exceeds_batch_size_raises_error(self):
        """测试超过批量大小限制。"""
        embedding = LocalEmbedding(batch_size=10)
        
        # 创建超过限制的批量
        large_batch = ["text"] * 15
        
        with pytest.raises(EmbeddingError, match="exceeds maximum"):
            embedding.embed(large_batch)


class TestLocalEmbeddingVectorProperties:
    """测试 Local Embedding 向量属性。"""
    
    def test_vector_values_in_valid_range(self):
        """测试向量值在有效范围内。"""
        embedding = LocalEmbedding(dimensions=100)
        texts = ["test1", "test2", "test3"]
        response = embedding.embed(texts)
        
        # 归一化向量的值应该在 [-1, 1] 范围内
        for vector in response.embeddings:
            for value in vector:
                assert -1.0 <= value <= 1.0
    
    def test_vector_not_all_zeros(self):
        """测试向量不是全零。"""
        embedding = LocalEmbedding(dimensions=128)
        vector = embedding.embed_single("test")
        
        # 至少有一些非零值
        non_zero_count = sum(1 for v in vector if abs(v) > 1e-6)
        assert non_zero_count > 0
    
    def test_vector_diversity(self):
        """测试向量具有多样性（不是所有值都相同）。"""
        embedding = LocalEmbedding(dimensions=100)
        vector = embedding.embed_single("diversity test")
        
        # 向量应该有多个不同的值
        unique_values = len(set(round(v, 6) for v in vector))
        assert unique_values > 10  # 至少有10个不同的值
    
    def test_similar_texts_produce_different_vectors(self):
        """测试相似文本产生不同向量（哈希敏感性）。"""
        embedding = LocalEmbedding(dimensions=128)
        
        vector1 = embedding.embed_single("hello world")
        vector2 = embedding.embed_single("hello world!")
        vector3 = embedding.embed_single("Hello world")
        
        # 即使文本相似，由于哈希敏感性，向量应该不同
        assert vector1 != vector2
        assert vector1 != vector3
        assert vector2 != vector3


class TestFactoryIntegration:
    """测试 Local Embedding 与工厂的集成。"""
    
    def setup_method(self):
        """每个测试方法前注册 Local 提供商。"""
        # 确保 Local 已注册
        if "local" not in EmbeddingFactory._registry:
            from src.libs.embedding import local_embedding
            EmbeddingFactory.register("local", local_embedding.LocalEmbedding)
    
    def test_factory_creates_local_embedding(self):
        """测试工厂能正确创建 Local Embedding 实例。"""
        from core import EmbeddingSettings
        
        settings = EmbeddingSettings(
            provider="local",
            model="fake-local-embedding",
            dimensions=768,
            batch_size=100
        )
        
        embedding = EmbeddingFactory.create_from_embedding_settings(settings)
        
        assert isinstance(embedding, LocalEmbedding)
        assert embedding.model == "fake-local-embedding"
        assert embedding.dimensions == 768
    
    def test_factory_creates_with_custom_config(self):
        """测试工厂使用自定义配置创建实例。"""
        from core import EmbeddingSettings
        
        settings = EmbeddingSettings(
            provider="local",
            model="custom-local",
            dimensions=384,
            batch_size=50
        )
        
        embedding = EmbeddingFactory.create_from_embedding_settings(settings)
        
        assert isinstance(embedding, LocalEmbedding)
        assert embedding.model == "custom-local"
        assert embedding.dimensions == 384
        assert embedding.batch_size == 50
    
    def test_factory_local_embedding_can_embed(self):
        """测试通过工厂创建的实例可以正常嵌入。"""
        from core import EmbeddingSettings
        
        settings = EmbeddingSettings(
            provider="local",
            model="test-local",
            dimensions=128,
            batch_size=100
        )
        
        embedding = EmbeddingFactory.create_from_embedding_settings(settings)
        vector = embedding.embed_single("factory test")
        
        assert len(vector) == 128
        # 验证归一化
        magnitude = sum(x * x for x in vector) ** 0.5
        assert abs(magnitude - 1.0) < 1e-6
