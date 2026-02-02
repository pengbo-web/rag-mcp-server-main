"""
CustomEvaluator 测试。

测试 EvaluatorFactory 和 CustomEvaluator 的功能。
验证各种评估指标的计算正确性。
"""

import pytest

from core import Settings, EvaluationSettings
from libs.evaluator import (
    BaseEvaluator,
    EvaluationQuery,
    EvaluationResult,
    EvaluationMetrics,
    EvaluatorError,
    CustomEvaluator,
    EvaluatorFactory,
)


class FakeEvaluator(BaseEvaluator):
    """用于测试的假 Evaluator 实现。"""
    
    def __init__(self, metrics: list[str], **kwargs):
        super().__init__(provider="fake", metrics=metrics, **kwargs)
        self.evaluate_count = 0
    
    def evaluate(self, queries, results, **kwargs):
        self.evaluate_count += 1
        self.validate_inputs(queries, results)
        
        # 返回固定指标
        return EvaluationMetrics(
            metrics={metric: 0.5 for metric in self.metrics}
        )


class TestEvaluatorFactory:
    """测试 Evaluator 工厂。"""
    
    def setup_method(self):
        """每个测试前清空注册表。"""
        EvaluatorFactory.clear_registry()
    
    def teardown_method(self):
        """每个测试后清空注册表。"""
        EvaluatorFactory.clear_registry()
    
    @pytest.mark.unit
    def test_register_provider(self):
        """测试注册提供商。"""
        EvaluatorFactory.register("fake", FakeEvaluator)
        
        assert EvaluatorFactory.is_registered("fake")
        assert "fake" in EvaluatorFactory.list_providers()
    
    @pytest.mark.unit
    def test_register_duplicate_provider_raises_error(self):
        """测试重复注册提供商会抛出错误。"""
        EvaluatorFactory.register("fake", FakeEvaluator)
        
        with pytest.raises(ValueError, match="already registered"):
            EvaluatorFactory.register("fake", CustomEvaluator)
    
    @pytest.mark.unit
    def test_register_non_baseevaluator_raises_error(self):
        """测试注册非 BaseEvaluator 子类会抛出错误。"""
        class NotAnEvaluator:
            pass
        
        with pytest.raises(TypeError, match="must inherit from BaseEvaluator"):
            EvaluatorFactory.register("invalid", NotAnEvaluator)
    
    @pytest.mark.unit
    def test_create_from_evaluation_settings(self):
        """测试从 EvaluationSettings 创建实例。"""
        EvaluatorFactory.register("fake", FakeEvaluator)
        
        evaluation_settings = EvaluationSettings(
            provider="fake",
            metrics=["hit_rate", "mrr"]
        )
        
        evaluator = EvaluatorFactory.create_from_evaluation_settings(evaluation_settings)
        
        assert isinstance(evaluator, FakeEvaluator)
        assert evaluator.provider == "fake"
        assert evaluator.metrics == ["hit_rate", "mrr"]
    
    @pytest.mark.unit
    def test_create_from_settings(self, sample_settings_dict):
        """测试从完整 Settings 创建实例。"""
        EvaluatorFactory.register("custom", CustomEvaluator)
        
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
        
        evaluator = EvaluatorFactory.create(settings)
        
        assert isinstance(evaluator, CustomEvaluator)
    
    @pytest.mark.unit
    def test_create_with_params(self):
        """测试使用指定参数创建实例。"""
        EvaluatorFactory.register("fake", FakeEvaluator)
        
        evaluator = EvaluatorFactory.create_with_params(
            provider="fake",
            metrics=["hit_rate"]
        )
        
        assert isinstance(evaluator, FakeEvaluator)
        assert evaluator.metrics == ["hit_rate"]
    
    @pytest.mark.unit
    def test_create_with_unknown_provider_raises_error(self):
        """测试使用未注册的提供商会抛出错误。"""
        with pytest.raises(EvaluatorError, match="Unknown Evaluator provider"):
            EvaluatorFactory.create_with_params(provider="unknown", metrics=["hit_rate"])
    
    @pytest.mark.unit
    def test_provider_case_insensitive(self):
        """测试提供商名称不区分大小写。"""
        EvaluatorFactory.register("fake", FakeEvaluator)
        
        evaluator = EvaluatorFactory.create_with_params(provider="FAKE", metrics=["hit_rate"])
        assert isinstance(evaluator, FakeEvaluator)


class TestCustomEvaluator:
    """测试 CustomEvaluator 实现。"""
    
    @pytest.mark.unit
    def test_hit_rate_calculation(self):
        """测试命中率计算。"""
        evaluator = CustomEvaluator(metrics=["hit_rate"])
        
        queries = [
            EvaluationQuery("q1", "query 1", ["doc1", "doc2"]),
            EvaluationQuery("q2", "query 2", ["doc3"]),
            EvaluationQuery("q3", "query 3", ["doc5"]),
        ]
        
        results = [
            EvaluationResult("q1", ["doc1", "doc4"]),  # 命中
            EvaluationResult("q2", ["doc4", "doc5"]),  # 未命中
            EvaluationResult("q3", ["doc5", "doc6"]),  # 命中
        ]
        
        metrics = evaluator.evaluate(queries, results)
        
        # 3个查询中有2个命中，hit_rate = 2/3 ≈ 0.667
        assert abs(metrics.get_metric("hit_rate") - 0.667) < 0.01
    
    @pytest.mark.unit
    def test_mrr_calculation(self):
        """测试 MRR 计算。"""
        evaluator = CustomEvaluator(metrics=["mrr"])
        
        queries = [
            EvaluationQuery("q1", "query 1", ["doc2"]),
            EvaluationQuery("q2", "query 2", ["doc1"]),
            EvaluationQuery("q3", "query 3", ["doc5"]),
        ]
        
        results = [
            EvaluationResult("q1", ["doc1", "doc2", "doc3"]),  # 第2位，1/2=0.5
            EvaluationResult("q2", ["doc1", "doc2"]),           # 第1位，1/1=1.0
            EvaluationResult("q3", ["doc1", "doc2", "doc3"]),   # 未命中，0.0
        ]
        
        metrics = evaluator.evaluate(queries, results)
        
        # MRR = (0.5 + 1.0 + 0.0) / 3 = 0.5
        assert abs(metrics.get_metric("mrr") - 0.5) < 0.01
    
    @pytest.mark.unit
    def test_precision_at_k_calculation(self):
        """测试 Precision@k 计算。"""
        evaluator = CustomEvaluator(metrics=["precision@k"], k=3)
        
        queries = [
            EvaluationQuery("q1", "query 1", ["doc1", "doc2", "doc3"]),
        ]
        
        results = [
            EvaluationResult("q1", ["doc1", "doc4", "doc2", "doc5"]),
        ]
        
        metrics = evaluator.evaluate(queries, results)
        
        # 前3个中有2个相关，precision@3 = 2/3 ≈ 0.667
        assert abs(metrics.get_metric("precision@k") - 0.667) < 0.01
    
    @pytest.mark.unit
    def test_recall_at_k_calculation(self):
        """测试 Recall@k 计算。"""
        evaluator = CustomEvaluator(metrics=["recall@k"], k=3)
        
        queries = [
            EvaluationQuery("q1", "query 1", ["doc1", "doc2", "doc3", "doc4"]),
        ]
        
        results = [
            EvaluationResult("q1", ["doc1", "doc5", "doc2", "doc6"]),
        ]
        
        metrics = evaluator.evaluate(queries, results)
        
        # 前3个中命中2个，总共4个相关，recall@3 = 2/4 = 0.5
        assert abs(metrics.get_metric("recall@k") - 0.5) < 0.01
    
    @pytest.mark.unit
    def test_ndcg_at_k_calculation(self):
        """测试 NDCG@k 计算。"""
        evaluator = CustomEvaluator(metrics=["ndcg@k"], k=3)
        
        queries = [
            EvaluationQuery("q1", "query 1", ["doc1", "doc2"]),
        ]
        
        results = [
            EvaluationResult("q1", ["doc1", "doc3", "doc2"]),
        ]
        
        metrics = evaluator.evaluate(queries, results)
        
        # 检查 NDCG 在 0 到 1 之间
        assert 0.0 <= metrics.get_metric("ndcg@k") <= 1.0
    
    @pytest.mark.unit
    def test_multiple_metrics(self):
        """测试同时计算多个指标。"""
        evaluator = CustomEvaluator(
            metrics=["hit_rate", "mrr", "precision@k"],
            k=5
        )
        
        queries = [
            EvaluationQuery("q1", "query 1", ["doc1", "doc2"]),
        ]
        
        results = [
            EvaluationResult("q1", ["doc1", "doc3", "doc4"]),
        ]
        
        metrics = evaluator.evaluate(queries, results)
        
        assert "hit_rate" in metrics.metrics
        assert "mrr" in metrics.metrics
        assert "precision@k" in metrics.metrics
    
    @pytest.mark.unit
    def test_empty_queries_raises_error(self):
        """测试空查询列表会抛出错误。"""
        evaluator = CustomEvaluator(metrics=["hit_rate"])
        
        with pytest.raises(EvaluatorError, match="查询列表不能为空"):
            evaluator.evaluate([], [])
    
    @pytest.mark.unit
    def test_mismatched_query_result_count_raises_error(self):
        """测试查询和结果数量不匹配会抛出错误。"""
        evaluator = CustomEvaluator(metrics=["hit_rate"])
        
        queries = [
            EvaluationQuery("q1", "query 1", ["doc1"]),
        ]
        
        results = [
            EvaluationResult("q1", ["doc1"]),
            EvaluationResult("q2", ["doc2"]),
        ]
        
        with pytest.raises(EvaluatorError, match="查询数量.*与结果数量.*不匹配"):
            evaluator.evaluate(queries, results)
    
    @pytest.mark.unit
    def test_mismatched_query_ids_raises_error(self):
        """测试查询和结果 ID 不匹配会抛出错误。"""
        evaluator = CustomEvaluator(metrics=["hit_rate"])
        
        queries = [
            EvaluationQuery("q1", "query 1", ["doc1"]),
        ]
        
        results = [
            EvaluationResult("q2", ["doc1"]),  # ID 不匹配
        ]
        
        with pytest.raises(EvaluatorError, match="查询 ID 与结果 ID 不匹配"):
            evaluator.evaluate(queries, results)
    
    @pytest.mark.unit
    def test_empty_metrics_raises_error(self):
        """测试空指标列表会抛出错误。"""
        with pytest.raises(EvaluatorError, match="指标列表不能为空"):
            CustomEvaluator(metrics=[])
    
    @pytest.mark.unit
    def test_unsupported_metric_raises_error(self):
        """测试不支持的指标会抛出错误。"""
        with pytest.raises(EvaluatorError, match="不支持的指标"):
            CustomEvaluator(metrics=["unknown_metric"])
    
    @pytest.mark.unit
    def test_duplicate_metrics_raises_error(self):
        """测试重复指标会抛出错误。"""
        with pytest.raises(EvaluatorError, match="指标列表包含重复项"):
            CustomEvaluator(metrics=["hit_rate", "hit_rate"])
