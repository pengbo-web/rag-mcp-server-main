"""
Embedding 工厂测试。

测试 EmbeddingFactory 的注册、创建和路由逻辑。
使用 Fake 实现避免真实 API 调用。
"""

import pytest

from core import Settings, EmbeddingSettings
from libs.embedding import BaseEmbedding, EmbeddingResponse, EmbeddingError, EmbeddingFactory


class FakeEmbedding(BaseEmbedding):
    """用于测试的假 Embedding 实现。"""
    
    def __init__(self, model: str, dimensions: int = 1536, **kwargs):
        """初始化假 Embedding。"""
        super().__init__(model, dimensions, **kwargs)
        self.call_count = 0
        self.last_texts = None
    
    def embed(self, texts: list[str], **kwargs) -> EmbeddingResponse:
        """模拟嵌入。"""
        self.call_count += 1
        self.last_texts = texts
        
        # 返回固定的假向量（全为 0.1）
        embeddings = [[0.1] * self.dimensions for _ in texts]
        
        return EmbeddingResponse(
            embeddings=embeddings,
            model=self.model,
            dimensions=self.dimensions,
            usage={"total_tokens": len(texts) * 10}
        )


class AnotherFakeEmbedding(BaseEmbedding):
    """另一个假 Embedding 实现（用于测试多提供商）。"""
    
    def embed(self, texts: list[str], **kwargs) -> EmbeddingResponse:
        """模拟嵌入。"""
        # 返回不同的假向量（全为 0.5）
        embeddings = [[0.5] * self.dimensions for _ in texts]
        
        return EmbeddingResponse(
            embeddings=embeddings,
            model=self.model,
            dimensions=self.dimensions
        )


class TestEmbeddingFactory:
    """测试 Embedding 工厂类。"""
    
    def setup_method(self):
        """每个测试前清空注册表。"""
        EmbeddingFactory.clear_registry()
    
    def teardown_method(self):
        """每个测试后清空注册表。"""
        EmbeddingFactory.clear_registry()
    
    @pytest.mark.unit
    def test_register_provider(self):
        """测试注册提供商。"""
        EmbeddingFactory.register("fake", FakeEmbedding)
        
        assert EmbeddingFactory.is_registered("fake")
        assert "fake" in EmbeddingFactory.list_providers()
    
    @pytest.mark.unit
    def test_register_duplicate_provider_raises_error(self):
        """测试重复注册提供商会抛出错误。"""
        EmbeddingFactory.register("fake", FakeEmbedding)
        
        with pytest.raises(ValueError, match="already registered"):
            EmbeddingFactory.register("fake", AnotherFakeEmbedding)
    
    @pytest.mark.unit
    def test_register_non_baseembedding_raises_error(self):
        """测试注册非 BaseEmbedding 子类会抛出错误。"""
        class NotAnEmbedding:
            pass
        
        with pytest.raises(TypeError, match="must inherit from BaseEmbedding"):
            EmbeddingFactory.register("invalid", NotAnEmbedding)
    
    @pytest.mark.unit
    def test_create_from_embedding_settings(self):
        """测试从 Embedding 设置创建实例。"""
        EmbeddingFactory.register("fake", FakeEmbedding)
        
        embedding_settings = EmbeddingSettings(
            provider="fake",
            model="fake-embedding",
            dimensions=768,
            batch_size=50
        )
        
        embedding = EmbeddingFactory.create_from_embedding_settings(embedding_settings)
        
        assert isinstance(embedding, FakeEmbedding)
        assert embedding.model == "fake-embedding"
        assert embedding.dimensions == 768
        assert embedding.batch_size == 50
    
    @pytest.mark.unit
    def test_create_from_settings(self, sample_settings_dict):
        """测试从完整 Settings 创建实例。"""
        EmbeddingFactory.register("openai", FakeEmbedding)
        
        # 修改设置使用 fake provider
        sample_settings_dict["embedding"]["provider"] = "openai"
        
        from core.settings import (
            Settings, _parse_section, LLMSettings, EmbeddingSettings,
            VectorStoreSettings, RetrievalSettings, RerankSettings,
            EvaluationSettings, ObservabilitySettings, IngestionSettings
        )
        settings = Settings(
            llm=_parse_section(sample_settings_dict, "llm", LLMSettings),
            embedding=_parse_section(sample_settings_dict, "embedding", EmbeddingSettings),
            vector_store=_parse_section(sample_settings_dict, "vector_store", VectorStoreSettings),
            retrieval=_parse_section(sample_settings_dict, "retrieval", RetrievalSettings),
            rerank=_parse_section(sample_settings_dict, "rerank", RerankSettings),
            evaluation=_parse_section(sample_settings_dict, "evaluation", EvaluationSettings),
            observability=_parse_section(sample_settings_dict, "observability", ObservabilitySettings),
            ingestion=_parse_section(sample_settings_dict, "ingestion", IngestionSettings)
        )
        
        embedding = EmbeddingFactory.create(settings)
        
        assert isinstance(embedding, FakeEmbedding)
        assert embedding.model == sample_settings_dict["embedding"]["model"]
    
    @pytest.mark.unit
    def test_create_with_unknown_provider_raises_error(self):
        """测试使用未注册的提供商会抛出错误。"""
        embedding_settings = EmbeddingSettings(
            provider="unknown",
            model="test-model"
        )
        
        with pytest.raises(EmbeddingError, match="Unknown Embedding provider"):
            EmbeddingFactory.create_from_embedding_settings(embedding_settings)
    
    @pytest.mark.unit
    def test_provider_case_insensitive(self):
        """测试提供商名称不区分大小写。"""
        EmbeddingFactory.register("fake", FakeEmbedding)
        
        embedding_settings = EmbeddingSettings(
            provider="FAKE",  # 大写
            model="test-model"
        )
        
        embedding = EmbeddingFactory.create_from_embedding_settings(embedding_settings)
        assert isinstance(embedding, FakeEmbedding)
    
    @pytest.mark.unit
    def test_multiple_providers(self):
        """测试注册和使用多个提供商。"""
        EmbeddingFactory.register("fake1", FakeEmbedding)
        EmbeddingFactory.register("fake2", AnotherFakeEmbedding)
        
        emb1 = EmbeddingFactory.create_from_embedding_settings(
            EmbeddingSettings(provider="fake1", model="model1")
        )
        emb2 = EmbeddingFactory.create_from_embedding_settings(
            EmbeddingSettings(provider="fake2", model="model2")
        )
        
        assert isinstance(emb1, FakeEmbedding)
        assert isinstance(emb2, AnotherFakeEmbedding)
        assert emb1.model == "model1"
        assert emb2.model == "model2"
    
    @pytest.mark.unit
    def test_embedding_can_be_called(self):
        """测试创建的 Embedding 实例可以正常调用。"""
        EmbeddingFactory.register("fake", FakeEmbedding)
        
        embedding = EmbeddingFactory.create_from_embedding_settings(
            EmbeddingSettings(provider="fake", model="test-model", dimensions=128)
        )
        
        texts = ["hello", "world"]
        response = embedding.embed(texts)
        
        assert len(response.embeddings) == 2
        assert len(response.embeddings[0]) == 128
        assert response.model == "test-model"
        assert isinstance(embedding, FakeEmbedding)
        assert embedding.call_count == 1
    
    @pytest.mark.unit
    def test_embed_single_returns_single_vector(self):
        """测试 embed_single 方法返回单个向量。"""
        EmbeddingFactory.register("fake", FakeEmbedding)
        
        embedding = EmbeddingFactory.create_from_embedding_settings(
            EmbeddingSettings(provider="fake", model="test-model", dimensions=64)
        )
        
        vector = embedding.embed_single("hello")
        
        assert isinstance(vector, list)
        assert len(vector) == 64
        assert all(isinstance(x, float) for x in vector)
    
    @pytest.mark.unit
    def test_embed_batch_handles_large_input(self):
        """测试 embed_batch 方法处理大批量输入（自动分批）。"""
        EmbeddingFactory.register("fake", FakeEmbedding)
        
        embedding = EmbeddingFactory.create_from_embedding_settings(
            EmbeddingSettings(provider="fake", model="test-model", batch_size=10)
        )
        
        # 25 个文本，batch_size=10，应该分 3 批
        texts = [f"text_{i}" for i in range(25)]
        vectors = embedding.embed_batch(texts)
        
        assert len(vectors) == 25
        assert isinstance(embedding, FakeEmbedding)
        assert embedding.call_count == 3  # 10 + 10 + 5
    
    @pytest.mark.unit
    def test_embed_returns_stable_vectors(self):
        """测试 Fake embedding 返回稳定的向量（用于验收）。"""
        EmbeddingFactory.register("fake", FakeEmbedding)
        
        embedding = EmbeddingFactory.create_from_embedding_settings(
            EmbeddingSettings(provider="fake", model="test-model", dimensions=4)
        )
        
        response = embedding.embed(["test"])
        
        # 验证向量稳定（全为 0.1）
        assert response.embeddings == [[0.1, 0.1, 0.1, 0.1]]
    
    @pytest.mark.unit
    def test_list_providers_empty_when_none_registered(self):
        """测试未注册任何提供商时列表为空。"""
        assert EmbeddingFactory.list_providers() == []
    
    @pytest.mark.unit
    def test_list_providers_returns_all_registered(self):
        """测试列出所有已注册的提供商。"""
        EmbeddingFactory.register("fake1", FakeEmbedding)
        EmbeddingFactory.register("fake2", AnotherFakeEmbedding)
        
        providers = EmbeddingFactory.list_providers()
        
        assert len(providers) == 2
        assert "fake1" in providers
        assert "fake2" in providers
