"""
Reranker 工厂测试。

测试 RerankerFactory 和 BaseReranker 的接口契约。
使用 NoneReranker 和 Fake 实现验证接口设计的正确性。
"""

import pytest

from core import Settings, RerankSettings
from libs.reranker import (
    BaseReranker,
    RerankCandidate,
    RerankResult,
    RerankerError,
    RerankerFactory,
    NoneReranker,
)


class FakeReranker(BaseReranker):
    """用于测试的假 Reranker 实现。"""
    
    def __init__(self, **kwargs):
        """初始化假 Reranker。"""
        super().__init__(backend="fake", **kwargs)
        self.rerank_count = 0
    
    def rerank(self, query, candidates, top_k=None, **kwargs):
        """模拟重排序（反转顺序）。"""
        self.rerank_count += 1
        self.validate_candidates(candidates)
        
        # 反转顺序作为"重排序"
        reversed_candidates = list(reversed(candidates))
        scores = [1.0 - (i * 0.1) for i in range(len(reversed_candidates))]
        
        results = self.create_results(reversed_candidates, scores)
        
        if top_k is not None and top_k > 0:
            results = results[:top_k]
        
        return results


class AnotherFakeReranker(BaseReranker):
    """另一个假 Reranker 实现（用于测试多后端）。"""
    
    def __init__(self, **kwargs):
        super().__init__(backend="another_fake", **kwargs)
    
    def rerank(self, query, candidates, top_k=None, **kwargs):
        self.validate_candidates(candidates)
        scores = [0.5] * len(candidates)
        return self.create_results(candidates, scores)


class TestRerankerFactory:
    """测试 Reranker 工厂。"""
    
    def setup_method(self):
        """每个测试前清空注册表。"""
        RerankerFactory.clear_registry()
    
    def teardown_method(self):
        """每个测试后清空注册表。"""
        RerankerFactory.clear_registry()
    
    @pytest.mark.unit
    def test_register_backend(self):
        """测试注册后端。"""
        RerankerFactory.register("fake", FakeReranker)
        
        assert RerankerFactory.is_registered("fake")
        assert "fake" in RerankerFactory.list_backends()
    
    @pytest.mark.unit
    def test_register_duplicate_backend_raises_error(self):
        """测试重复注册后端会抛出错误。"""
        RerankerFactory.register("fake", FakeReranker)
        
        with pytest.raises(ValueError, match="already registered"):
            RerankerFactory.register("fake", AnotherFakeReranker)
    
    @pytest.mark.unit
    def test_register_non_basereranker_raises_error(self):
        """测试注册非 BaseReranker 子类会抛出错误。"""
        class NotAReranker:
            pass
        
        with pytest.raises(TypeError, match="must inherit from BaseReranker"):
            RerankerFactory.register("invalid", NotAReranker)
    
    @pytest.mark.unit
    def test_create_from_rerank_settings(self):
        """测试从 RerankSettings 创建实例。"""
        RerankerFactory.register("fake", FakeReranker)
        
        rerank_settings = RerankSettings(provider="fake")
        
        reranker = RerankerFactory.create_from_rerank_settings(rerank_settings)
        
        assert isinstance(reranker, FakeReranker)
        assert reranker.backend == "fake"
    
    @pytest.mark.unit
    def test_create_from_settings(self, sample_settings_dict):
        """测试从完整 Settings 创建实例。"""
        RerankerFactory.register("none", NoneReranker)
        
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
        
        reranker = RerankerFactory.create(settings)
        
        assert isinstance(reranker, NoneReranker)
    
    @pytest.mark.unit
    def test_create_with_params(self):
        """测试使用指定参数创建实例。"""
        RerankerFactory.register("fake", FakeReranker)
        
        reranker = RerankerFactory.create_with_params(backend="fake")
        
        assert isinstance(reranker, FakeReranker)
        assert reranker.backend == "fake"
    
    @pytest.mark.unit
    def test_create_with_unknown_backend_raises_error(self):
        """测试使用未注册的后端会抛出错误。"""
        with pytest.raises(RerankerError, match="Unknown Reranker backend"):
            RerankerFactory.create_with_params(backend="unknown")
    
    @pytest.mark.unit
    def test_backend_case_insensitive(self):
        """测试后端名称不区分大小写。"""
        RerankerFactory.register("fake", FakeReranker)
        
        reranker = RerankerFactory.create_with_params(backend="FAKE")
        assert isinstance(reranker, FakeReranker)
    
    @pytest.mark.unit
    def test_multiple_backends(self):
        """测试注册和使用多个后端。"""
        RerankerFactory.register("fake1", FakeReranker)
        RerankerFactory.register("fake2", AnotherFakeReranker)
        
        reranker1 = RerankerFactory.create_with_params(backend="fake1")
        reranker2 = RerankerFactory.create_with_params(backend="fake2")
        
        assert isinstance(reranker1, FakeReranker)
        assert isinstance(reranker2, AnotherFakeReranker)
    
    @pytest.mark.unit
    def test_list_backends_returns_all_registered(self):
        """测试列出所有已注册的后端。"""
        RerankerFactory.register("fake1", FakeReranker)
        RerankerFactory.register("fake2", AnotherFakeReranker)
        
        backends = RerankerFactory.list_backends()
        
        assert len(backends) == 2
        assert "fake1" in backends
        assert "fake2" in backends


class TestNoneReranker:
    """测试 NoneReranker 实现。"""
    
    @pytest.mark.unit
    def test_none_reranker_preserves_order(self):
        """测试 NoneReranker 保持原始顺序。"""
        reranker = NoneReranker()
        
        candidates = [
            RerankCandidate(id="1", text="first", score=0.9),
            RerankCandidate(id="2", text="second", score=0.8),
            RerankCandidate(id="3", text="third", score=0.7),
        ]
        
        results = reranker.rerank(query="test", candidates=candidates)
        
        # 验证顺序保持不变
        assert len(results) == 3
        assert results[0].id == "1"
        assert results[1].id == "2"
        assert results[2].id == "3"
        
        # 验证分数保持不变
        assert results[0].score == 0.9
        assert results[1].score == 0.8
        assert results[2].score == 0.7
    
    @pytest.mark.unit
    def test_none_reranker_handles_missing_scores(self):
        """测试 NoneReranker 处理缺失分数的候选项。"""
        reranker = NoneReranker()
        
        candidates = [
            RerankCandidate(id="1", text="first"),  # 无分数
            RerankCandidate(id="2", text="second", score=0.5),
            RerankCandidate(id="3", text="third"),  # 无分数
        ]
        
        results = reranker.rerank(query="test", candidates=candidates)
        
        # 无分数的候选项应该被赋予 0.0
        assert results[0].score == 0.0
        assert results[1].score == 0.5
        assert results[2].score == 0.0
    
    @pytest.mark.unit
    def test_none_reranker_respects_top_k(self):
        """测试 NoneReranker 遵守 top_k 限制。"""
        reranker = NoneReranker()
        
        candidates = [
            RerankCandidate(id=f"id_{i}", text=f"text {i}", score=1.0 - i * 0.1)
            for i in range(10)
        ]
        
        results = reranker.rerank(query="test", candidates=candidates, top_k=3)
        
        assert len(results) == 3
        assert results[0].id == "id_0"
        assert results[1].id == "id_1"
        assert results[2].id == "id_2"


class TestRerankerContract:
    """测试 Reranker 接口契约。"""
    
    def setup_method(self):
        """每个测试前清空注册表。"""
        RerankerFactory.clear_registry()
    
    def teardown_method(self):
        """每个测试后清空注册表。"""
        RerankerFactory.clear_registry()
    
    @pytest.mark.unit
    def test_rerank_validates_empty_candidates(self):
        """测试空候选项列表会抛出错误。"""
        RerankerFactory.register("fake", FakeReranker)
        reranker = RerankerFactory.create_with_params(backend="fake")
        
        with pytest.raises(RerankerError, match="候选项列表不能为空"):
            reranker.rerank(query="test", candidates=[])
    
    @pytest.mark.unit
    def test_rerank_validates_candidate_types(self):
        """测试非 RerankCandidate 类型会抛出错误。"""
        RerankerFactory.register("fake", FakeReranker)
        reranker = RerankerFactory.create_with_params(backend="fake")
        
        # 传入普通 dict 而不是 RerankCandidate
        with pytest.raises(RerankerError, match="必须是 RerankCandidate 实例"):
            reranker.rerank(query="test", candidates=[{"id": "1", "text": "test"}])
    
    @pytest.mark.unit
    def test_rerank_validates_unique_ids(self):
        """测试重复 ID 会抛出错误。"""
        RerankerFactory.register("fake", FakeReranker)
        reranker = RerankerFactory.create_with_params(backend="fake")
        
        candidates = [
            RerankCandidate(id="duplicate", text="first"),
            RerankCandidate(id="duplicate", text="second"),  # 重复 ID
        ]
        
        with pytest.raises(RerankerError, match="ID 必须唯一"):
            reranker.rerank(query="test", candidates=candidates)
    
    @pytest.mark.unit
    def test_rerank_result_structure(self):
        """测试重排序结果的结构。"""
        RerankerFactory.register("fake", FakeReranker)
        reranker = RerankerFactory.create_with_params(backend="fake")
        
        candidates = [
            RerankCandidate(id="1", text="first", score=0.5, metadata={"source": "doc1"}),
            RerankCandidate(id="2", text="second", score=0.6),
        ]
        
        results = reranker.rerank(query="test", candidates=candidates)
        
        # 验证结果结构
        assert all(isinstance(r, RerankResult) for r in results)
        assert all(hasattr(r, 'id') for r in results)
        assert all(hasattr(r, 'text') for r in results)
        assert all(hasattr(r, 'score') for r in results)
        assert all(hasattr(r, 'original_rank') for r in results)
        assert all(hasattr(r, 'new_rank') for r in results)
        
        # 验证元数据保留
        result_with_metadata = [r for r in results if r.id == "1"][0]
        assert result_with_metadata.metadata == {"source": "doc1"}
    
    @pytest.mark.unit
    def test_rerank_preserves_text_and_metadata(self):
        """测试重排序保留原始文本和元数据。"""
        RerankerFactory.register("fake", FakeReranker)
        reranker = RerankerFactory.create_with_params(backend="fake")
        
        candidates = [
            RerankCandidate(
                id="1", 
                text="original text", 
                metadata={"key": "value"}
            ),
        ]
        
        results = reranker.rerank(query="test", candidates=candidates)
        
        assert results[0].text == "original text"
        assert results[0].metadata == {"key": "value"}
    
    @pytest.mark.unit
    def test_default_none_backend_registered(self):
        """测试可以注册和使用 none 后端。"""
        # 注册 none 后端
        RerankerFactory.register("none", NoneReranker)
        assert RerankerFactory.is_registered("none")
        
        reranker = RerankerFactory.create_with_params(backend="none")
        assert isinstance(reranker, NoneReranker)
