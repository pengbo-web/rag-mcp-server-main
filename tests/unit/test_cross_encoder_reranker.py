"""
CrossEncoderReranker 单元测试。

测试 Cross-Encoder 重排序器的各项功能，包括打分、超时控制、回退机制等。
"""

import pytest
import time
from unittest.mock import Mock

from src.libs.reranker.base_reranker import RerankCandidate, RerankerError
from src.libs.reranker.cross_encoder_reranker import CrossEncoderReranker


def mock_scorer_simple(query: str, texts: list[str]) -> list[float]:
    """简单的 mock scorer：按文本长度打分。"""
    return [1.0 / (len(text) + 1) for text in texts]


def mock_scorer_query_aware(query: str, texts: list[str]) -> list[float]:
    """查询感知的 mock scorer：基于词重叠。"""
    query_words = set(query.lower().split())
    scores = []
    for text in texts:
        text_words = set(text.lower().split())
        overlap = len(query_words & text_words)
        scores.append(float(overlap) / max(len(query_words), 1))
    return scores


def mock_scorer_slow(query: str, texts: list[str]) -> list[float]:
    """慢速 mock scorer：模拟超时。"""
    time.sleep(2)  # 模拟慢速处理
    return [0.5] * len(texts)


def mock_scorer_error(query: str, texts: list[str]) -> list[float]:
    """错误 mock scorer：总是抛出异常。"""
    raise ValueError("Mock scorer error")


@pytest.fixture
def sample_candidates():
    """创建测试候选项。"""
    return [
        RerankCandidate(id="1", text="Python is a programming language", score=0.9),
        RerankCandidate(id="2", text="Java is also a programming language", score=0.8),
        RerankCandidate(id="3", text="Machine learning uses algorithms", score=0.7),
    ]


class TestCrossEncoderRerankerInitialization:
    """测试 CrossEncoderReranker 初始化。"""
    
    def test_init_with_default_scorer(self):
        """测试使用默认 scorer 初始化。"""
        reranker = CrossEncoderReranker()
        
        assert reranker.backend == "cross_encoder"
        assert reranker.model_name == "cross-encoder-default"
        assert reranker.timeout == 30
        assert reranker.batch_size == 32
        assert reranker.fallback_on_error is True
        assert reranker.scorer is not None
    
    def test_init_with_custom_scorer(self):
        """测试使用自定义 scorer 初始化。"""
        reranker = CrossEncoderReranker(
            scorer=mock_scorer_simple,
            model_name="custom-model",
            timeout=60,
            batch_size=16,
            fallback_on_error=False
        )
        
        assert reranker.scorer is mock_scorer_simple
        assert reranker.model_name == "custom-model"
        assert reranker.timeout == 60
        assert reranker.batch_size == 16
        assert reranker.fallback_on_error is False
    
    def test_repr(self):
        """测试字符串表示。"""
        reranker = CrossEncoderReranker(model_name="test-model", timeout=45)
        repr_str = repr(reranker)
        
        assert "CrossEncoderReranker" in repr_str
        assert "test-model" in repr_str
        assert "45" in repr_str


class TestDefaultMockScorer:
    """测试默认 mock scorer。"""
    
    def test_default_scorer_basic(self):
        """测试默认 scorer 基本功能。"""
        reranker = CrossEncoderReranker()
        
        scores = reranker.scorer("python programming", ["python", "java", "machine learning"])
        
        assert len(scores) == 3
        assert all(0 <= s <= 1 for s in scores)
        # "python" 应该有最高分（与查询词重叠）
        assert scores[0] > scores[1]
    
    def test_default_scorer_deterministic(self):
        """测试默认 scorer 确定性。"""
        reranker = CrossEncoderReranker()
        
        scores1 = reranker.scorer("test query", ["text1", "text2"])
        scores2 = reranker.scorer("test query", ["text1", "text2"])
        
        assert scores1 == scores2
    
    def test_default_scorer_empty_query(self):
        """测试默认 scorer 处理空查询。"""
        reranker = CrossEncoderReranker()
        
        scores = reranker.scorer("", ["text1", "text2"])
        
        assert len(scores) == 2
        assert all(isinstance(s, float) for s in scores)


class TestScoring:
    """测试打分功能。"""
    
    def test_score_with_custom_scorer(self, sample_candidates):
        """测试使用自定义 scorer 打分。"""
        reranker = CrossEncoderReranker(scorer=mock_scorer_simple)
        
        results = reranker.rerank("test query", sample_candidates)
        
        assert len(results) == 3
        # 验证分数已应用（按文本长度排序）
        assert all(r.score > 0 for r in results)
    
    def test_score_query_aware(self, sample_candidates):
        """测试查询感知的打分。"""
        reranker = CrossEncoderReranker(scorer=mock_scorer_query_aware)
        
        results = reranker.rerank("programming language", sample_candidates)
        
        assert len(results) == 3
        # "programming language" 应该给前两个候选项更高分
        assert results[0].text in [
            "Python is a programming language",
            "Java is also a programming language"
        ]


class TestReranking:
    """测试重排序功能。"""
    
    def test_rerank_basic(self, sample_candidates):
        """测试基本重排序功能。"""
        reranker = CrossEncoderReranker(scorer=mock_scorer_simple)
        
        results = reranker.rerank("test query", sample_candidates)
        
        assert len(results) == 3
        # 验证结果结构
        assert all(hasattr(r, 'id') for r in results)
        assert all(hasattr(r, 'score') for r in results)
        assert all(hasattr(r, 'new_rank') for r in results)
        # 验证按分数降序排列
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score
    
    def test_rerank_with_top_k(self, sample_candidates):
        """测试 top_k 限制。"""
        reranker = CrossEncoderReranker(scorer=mock_scorer_simple)
        
        results = reranker.rerank("test query", sample_candidates, top_k=2)
        
        assert len(results) == 2
    
    def test_rerank_preserves_metadata(self, sample_candidates):
        """测试重排序保留元数据。"""
        candidates_with_metadata = [
            RerankCandidate(id="1", text="text1", metadata={"key": "value1"}),
            RerankCandidate(id="2", text="text2", metadata={"key": "value2"}),
        ]
        
        reranker = CrossEncoderReranker(scorer=mock_scorer_simple)
        results = reranker.rerank("query", candidates_with_metadata)
        
        assert all(r.metadata is not None for r in results)
        assert any(r.metadata.get("key") == "value1" for r in results)
    
    def test_rerank_empty_candidates_raises_error(self):
        """测试空候选项列表抛出错误。"""
        reranker = CrossEncoderReranker(scorer=mock_scorer_simple)
        
        with pytest.raises(RerankerError, match="候选项列表不能为空"):
            reranker.rerank("query", [])


class TestBatchProcessing:
    """测试批处理功能。"""
    
    def test_batch_processing_small_batch(self):
        """测试小批量处理。"""
        candidates = [
            RerankCandidate(id=str(i), text=f"text{i}")
            for i in range(10)
        ]
        
        reranker = CrossEncoderReranker(
            scorer=mock_scorer_simple,
            batch_size=3
        )
        
        results = reranker.rerank("query", candidates)
        
        assert len(results) == 10
    
    def test_batch_processing_large_batch(self):
        """测试大批量处理。"""
        candidates = [
            RerankCandidate(id=str(i), text=f"text{i}")
            for i in range(100)
        ]
        
        reranker = CrossEncoderReranker(
            scorer=mock_scorer_simple,
            batch_size=32
        )
        
        results = reranker.rerank("query", candidates)
        
        assert len(results) == 100


class TestTimeoutHandling:
    """测试超时处理。"""
    
    def test_timeout_with_fallback(self, sample_candidates):
        """测试超时时启用回退。"""
        reranker = CrossEncoderReranker(
            scorer=mock_scorer_slow,
            timeout=1,  # 1 秒超时，但 scorer 需要 2 秒
            fallback_on_error=True
        )
        
        # 不应抛出错误，而是回退到原始顺序
        results = reranker.rerank("query", sample_candidates)
        
        assert len(results) == 3
        assert reranker.did_timeout_occur() is True
        assert reranker.get_last_error() is not None
    
    def test_timeout_without_fallback(self, sample_candidates):
        """测试超时时禁用回退。"""
        reranker = CrossEncoderReranker(
            scorer=mock_scorer_slow,
            timeout=1,
            fallback_on_error=False
        )
        
        # 应该抛出错误
        with pytest.raises(RerankerError, match="timeout"):
            reranker.rerank("query", sample_candidates)


class TestErrorHandling:
    """测试错误处理。"""
    
    def test_scorer_error_with_fallback(self, sample_candidates):
        """测试 scorer 错误时启用回退。"""
        reranker = CrossEncoderReranker(
            scorer=mock_scorer_error,
            fallback_on_error=True
        )
        
        # 不应抛出错误，而是回退到原始顺序
        results = reranker.rerank("query", sample_candidates)
        
        assert len(results) == 3
        assert reranker.get_last_error() is not None
        # 验证回退到原始分数
        assert results[0].score == 0.9  # 原始分数
        assert results[1].score == 0.8
        assert results[2].score == 0.7
    
    def test_scorer_error_without_fallback(self, sample_candidates):
        """测试 scorer 错误时禁用回退。"""
        reranker = CrossEncoderReranker(
            scorer=mock_scorer_error,
            fallback_on_error=False
        )
        
        # 应该抛出错误
        with pytest.raises(RerankerError, match="Scoring failed"):
            reranker.rerank("query", sample_candidates)
    
    def test_scorer_wrong_score_count(self, sample_candidates):
        """测试 scorer 返回错误数量的分数。"""
        def bad_scorer(query: str, texts: list[str]) -> list[float]:
            return [0.5]  # 总是返回 1 个分数，不管输入多少
        
        reranker = CrossEncoderReranker(
            scorer=bad_scorer,
            fallback_on_error=False
        )
        
        with pytest.raises(RerankerError, match="returned.*scores for"):
            reranker.rerank("query", sample_candidates)


class TestFallbackBehavior:
    """测试回退行为。"""
    
    def test_fallback_preserves_original_order(self):
        """测试回退保持原始顺序。"""
        candidates = [
            RerankCandidate(id="1", text="text1", score=0.9),
            RerankCandidate(id="2", text="text2", score=0.8),
            RerankCandidate(id="3", text="text3", score=0.7),
        ]
        
        reranker = CrossEncoderReranker(
            scorer=mock_scorer_error,
            fallback_on_error=True
        )
        
        results = reranker.rerank("query", candidates)
        
        # 验证保持原始顺序
        assert [r.id for r in results] == ["1", "2", "3"]
        assert [r.new_rank for r in results] == [0, 1, 2]
    
    def test_fallback_with_missing_scores(self):
        """测试回退时处理缺失的原始分数。"""
        candidates = [
            RerankCandidate(id="1", text="text1"),  # 无分数
            RerankCandidate(id="2", text="text2", score=0.8),
        ]
        
        reranker = CrossEncoderReranker(
            scorer=mock_scorer_error,
            fallback_on_error=True
        )
        
        results = reranker.rerank("query", candidates)
        
        # 缺失的分数应该用 0.0 填充
        assert results[0].score == 0.0
        assert results[1].score == 0.8


class TestStateTracking:
    """测试状态跟踪。"""
    
    def test_get_last_error_after_success(self, sample_candidates):
        """测试成功后获取错误。"""
        reranker = CrossEncoderReranker(scorer=mock_scorer_simple)
        
        reranker.rerank("query", sample_candidates)
        
        assert reranker.get_last_error() is None
        assert reranker.did_timeout_occur() is False
    
    def test_get_last_error_after_failure(self, sample_candidates):
        """测试失败后获取错误。"""
        reranker = CrossEncoderReranker(
            scorer=mock_scorer_error,
            fallback_on_error=True
        )
        
        reranker.rerank("query", sample_candidates)
        
        assert reranker.get_last_error() is not None
        assert isinstance(reranker.get_last_error(), Exception)
    
    def test_state_reset_on_new_call(self, sample_candidates):
        """测试新调用重置状态。"""
        reranker = CrossEncoderReranker(
            scorer=mock_scorer_error,
            fallback_on_error=True
        )
        
        # 第一次调用导致错误
        reranker.rerank("query", sample_candidates)
        assert reranker.get_last_error() is not None
        
        # 更换为正常 scorer
        reranker.scorer = mock_scorer_simple
        
        # 第二次调用成功
        reranker.rerank("query", sample_candidates)
        assert reranker.get_last_error() is None


class TestFactoryIntegration:
    """测试工厂集成。"""
    
    def test_factory_creates_cross_encoder(self, sample_candidates):
        """测试工厂可创建 CrossEncoderReranker。"""
        from src.libs.reranker import RerankerFactory
        
        reranker = RerankerFactory.create_with_params(
            backend="cross_encoder",
            scorer=mock_scorer_simple
        )
        
        assert isinstance(reranker, CrossEncoderReranker)
        
        # 验证可以正常使用
        results = reranker.rerank("query", sample_candidates)
        assert len(results) == 3
