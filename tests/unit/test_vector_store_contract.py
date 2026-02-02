"""
VectorStore 契约测试。

测试 VectorStoreFactory 和 BaseVectorStore 的接口契约。
使用 Fake 实现验证接口设计的正确性，不涉及真实数据库。
"""

import pytest

from core import Settings, VectorStoreSettings
from libs.vector_store import (
    BaseVectorStore,
    VectorRecord,
    QueryResult,
    VectorStoreError,
    VectorStoreFactory,
)


class FakeVectorStore(BaseVectorStore):
    """用于测试的假 VectorStore 实现。"""
    
    def __init__(self, collection_name: str = "default", **kwargs):
        """初始化假 VectorStore。"""
        super().__init__(collection_name, **kwargs)
        self.records = {}  # id -> VectorRecord
        self.upsert_count = 0
        self.query_count = 0
    
    def upsert(self, records: list[VectorRecord], **kwargs) -> None:
        """模拟插入/更新。"""
        self.upsert_count += 1
        for record in records:
            self.records[record.id] = record
    
    def query(
        self,
        vector: list[float],
        top_k: int = 10,
        filters: dict = None,
        **kwargs
    ) -> list[QueryResult]:
        """模拟查询（返回固定结果）。"""
        self.query_count += 1
        
        # 简单实现：返回前 top_k 个记录
        results = []
        for i, (record_id, record) in enumerate(list(self.records.items())[:top_k]):
            results.append(QueryResult(
                id=record.id,
                score=1.0 - (i * 0.1),  # 降序分数
                text=record.text,
                metadata=record.metadata,
                vector=record.vector
            ))
        
        return results
    
    def delete(self, ids: list[str], **kwargs) -> None:
        """模拟删除。"""
        for record_id in ids:
            self.records.pop(record_id, None)
    
    def get(self, ids: list[str], **kwargs) -> list[VectorRecord]:
        """模拟获取。"""
        results = []
        for record_id in ids:
            if record_id in self.records:
                results.append(self.records[record_id])
        return results
    
    def count(self, **kwargs) -> int:
        """返回记录数量。"""
        return len(self.records)
    
    def clear(self, **kwargs) -> None:
        """清空所有记录。"""
        self.records.clear()


class AnotherFakeVectorStore(BaseVectorStore):
    """另一个假 VectorStore 实现（用于测试多提供商）。"""
    
    def __init__(self, collection_name: str = "default", **kwargs):
        super().__init__(collection_name, **kwargs)
        self.storage = []
    
    def upsert(self, records: list[VectorRecord], **kwargs) -> None:
        self.storage.extend(records)
    
    def query(self, vector: list[float], top_k: int = 10, filters: dict = None, **kwargs) -> list[QueryResult]:
        return []
    
    def delete(self, ids: list[str], **kwargs) -> None:
        pass
    
    def get(self, ids: list[str], **kwargs) -> list[VectorRecord]:
        return []


class TestVectorStoreContract:
    """测试 VectorStore 契约。"""
    
    def setup_method(self):
        """每个测试前清空注册表。"""
        VectorStoreFactory.clear_registry()
    
    def teardown_method(self):
        """每个测试后清空注册表。"""
        VectorStoreFactory.clear_registry()
    
    @pytest.mark.unit
    def test_register_provider(self):
        """测试注册提供商。"""
        VectorStoreFactory.register("fake", FakeVectorStore)
        
        assert VectorStoreFactory.is_registered("fake")
        assert "fake" in VectorStoreFactory.list_providers()
    
    @pytest.mark.unit
    def test_register_duplicate_provider_raises_error(self):
        """测试重复注册提供商会抛出错误。"""
        VectorStoreFactory.register("fake", FakeVectorStore)
        
        with pytest.raises(ValueError, match="already registered"):
            VectorStoreFactory.register("fake", AnotherFakeVectorStore)
    
    @pytest.mark.unit
    def test_register_non_basevectorstore_raises_error(self):
        """测试注册非 BaseVectorStore 子类会抛出错误。"""
        class NotAVectorStore:
            pass
        
        with pytest.raises(TypeError, match="must inherit from BaseVectorStore"):
            VectorStoreFactory.register("invalid", NotAVectorStore)
    
    @pytest.mark.unit
    def test_create_from_vector_store_settings(self):
        """测试从 VectorStore 设置创建实例。"""
        VectorStoreFactory.register("fake", FakeVectorStore)
        
        vector_store_settings = VectorStoreSettings(
            provider="fake",
            collection_name="test_collection"
        )
        
        store = VectorStoreFactory.create_from_vector_store_settings(vector_store_settings)
        
        assert isinstance(store, FakeVectorStore)
        assert store.collection_name == "test_collection"
    
    @pytest.mark.unit
    def test_create_from_settings(self, sample_settings_dict):
        """测试从完整 Settings 创建实例。"""
        VectorStoreFactory.register("chroma", FakeVectorStore)
        
        # 修改设置使用 fake provider
        sample_settings_dict["vector_store"]["provider"] = "chroma"
        
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
        
        store = VectorStoreFactory.create(settings)
        
        assert isinstance(store, FakeVectorStore)
    
    @pytest.mark.unit
    def test_create_with_params(self):
        """测试使用指定参数创建实例。"""
        VectorStoreFactory.register("fake", FakeVectorStore)
        
        store = VectorStoreFactory.create_with_params(
            provider="fake",
            collection_name="my_collection"
        )
        
        assert isinstance(store, FakeVectorStore)
        assert store.collection_name == "my_collection"
    
    @pytest.mark.unit
    def test_create_with_unknown_provider_raises_error(self):
        """测试使用未注册的提供商会抛出错误。"""
        with pytest.raises(VectorStoreError, match="Unknown VectorStore provider"):
            VectorStoreFactory.create_with_params(provider="unknown")
    
    @pytest.mark.unit
    def test_provider_case_insensitive(self):
        """测试提供商名称不区分大小写。"""
        VectorStoreFactory.register("fake", FakeVectorStore)
        
        store = VectorStoreFactory.create_with_params(provider="FAKE")
        assert isinstance(store, FakeVectorStore)
    
    @pytest.mark.unit
    def test_upsert_records(self):
        """测试插入记录（契约测试：输入输出格式）。"""
        VectorStoreFactory.register("fake", FakeVectorStore)
        
        store = VectorStoreFactory.create_with_params(provider="fake")
        
        records = [
            VectorRecord(
                id="1",
                vector=[0.1, 0.2, 0.3],
                text="test text 1",
                metadata={"source": "test"}
            ),
            VectorRecord(
                id="2",
                vector=[0.4, 0.5, 0.6],
                text="test text 2",
                metadata={"source": "test"}
            )
        ]
        
        # 验证方法存在且可调用（不抛出异常）
        store.upsert(records)
        
        # 验证假实现确实存储了数据
        assert isinstance(store, FakeVectorStore)
        assert store.upsert_count == 1
        assert len(store.records) == 2
    
    @pytest.mark.unit
    def test_query_returns_correct_shape(self):
        """测试查询返回正确的数据格式（契约测试）。"""
        VectorStoreFactory.register("fake", FakeVectorStore)
        
        store = VectorStoreFactory.create_with_params(provider="fake")
        
        # 先插入一些数据
        records = [
            VectorRecord(id=f"id_{i}", vector=[0.1] * 3, text=f"text {i}", metadata={"index": i})
            for i in range(5)
        ]
        store.upsert(records)
        
        # 执行查询
        query_vector = [0.1, 0.2, 0.3]
        results = store.query(vector=query_vector, top_k=3)
        
        # 验证返回格式
        assert isinstance(results, list)
        assert len(results) <= 3
        
        for result in results:
            assert isinstance(result, QueryResult)
            assert hasattr(result, 'id')
            assert hasattr(result, 'score')
            assert hasattr(result, 'text')
            assert hasattr(result, 'metadata')
    
    @pytest.mark.unit
    def test_delete_records(self):
        """测试删除记录（契约测试）。"""
        VectorStoreFactory.register("fake", FakeVectorStore)
        
        store = VectorStoreFactory.create_with_params(provider="fake")
        
        # 插入记录
        records = [
            VectorRecord(id="1", vector=[0.1] * 3, text="text 1"),
            VectorRecord(id="2", vector=[0.2] * 3, text="text 2"),
        ]
        store.upsert(records)
        
        # 删除记录
        store.delete(["1"])
        
        # 验证删除成功
        assert isinstance(store, FakeVectorStore)
        assert "1" not in store.records
        assert "2" in store.records
    
    @pytest.mark.unit
    def test_get_records_by_ids(self):
        """测试根据 ID 获取记录（契约测试）。"""
        VectorStoreFactory.register("fake", FakeVectorStore)
        
        store = VectorStoreFactory.create_with_params(provider="fake")
        
        # 插入记录
        records = [
            VectorRecord(id="1", vector=[0.1] * 3, text="text 1"),
            VectorRecord(id="2", vector=[0.2] * 3, text="text 2"),
        ]
        store.upsert(records)
        
        # 获取记录
        retrieved = store.get(["1", "2"])
        
        # 验证返回格式
        assert isinstance(retrieved, list)
        assert len(retrieved) == 2
        assert all(isinstance(r, VectorRecord) for r in retrieved)
    
    @pytest.mark.unit
    def test_count_records(self):
        """测试计数功能（契约测试）。"""
        VectorStoreFactory.register("fake", FakeVectorStore)
        
        store = VectorStoreFactory.create_with_params(provider="fake")
        
        # 初始为空
        assert store.count() == 0
        
        # 插入记录
        records = [
            VectorRecord(id=f"id_{i}", vector=[0.1] * 3, text=f"text {i}")
            for i in range(5)
        ]
        store.upsert(records)
        
        # 验证计数
        assert store.count() == 5
    
    @pytest.mark.unit
    def test_clear_collection(self):
        """测试清空集合（契约测试）。"""
        VectorStoreFactory.register("fake", FakeVectorStore)
        
        store = VectorStoreFactory.create_with_params(provider="fake")
        
        # 插入记录
        records = [
            VectorRecord(id=f"id_{i}", vector=[0.1] * 3, text=f"text {i}")
            for i in range(3)
        ]
        store.upsert(records)
        
        assert store.count() == 3
        
        # 清空
        store.clear()
        
        # 验证已清空
        assert store.count() == 0
    
    @pytest.mark.unit
    def test_multiple_providers(self):
        """测试注册和使用多个提供商。"""
        VectorStoreFactory.register("fake1", FakeVectorStore)
        VectorStoreFactory.register("fake2", AnotherFakeVectorStore)
        
        store1 = VectorStoreFactory.create_with_params(provider="fake1")
        store2 = VectorStoreFactory.create_with_params(provider="fake2")
        
        assert isinstance(store1, FakeVectorStore)
        assert isinstance(store2, AnotherFakeVectorStore)
    
    @pytest.mark.unit
    def test_list_providers_returns_all_registered(self):
        """测试列出所有已注册的提供商。"""
        VectorStoreFactory.register("fake1", FakeVectorStore)
        VectorStoreFactory.register("fake2", AnotherFakeVectorStore)
        
        providers = VectorStoreFactory.list_providers()
        
        assert len(providers) == 2
        assert "fake1" in providers
        assert "fake2" in providers
