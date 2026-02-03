"""
ChromaStore 集成测试。

此测试文件验证 ChromaStore 的实际功能，包括：
- 基本的 upsert 和 query 操作
- 元数据过滤
- CRUD 完整流程
- 持久化功能

注意：这些是集成测试，需要 chromadb 依赖。
如果环境中未安装 chromadb，测试将被跳过。
"""

import os
import shutil
import time
import pytest
from typing import List

from src.libs.vector_store import (
    ChromaStore,
    VectorRecord,
    QueryResult,
    VectorStoreError,
    VectorStoreFactory
)
from core import VectorStoreSettings


# 检查是否安装了 chromadb
try:
    import chromadb
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False


pytestmark = pytest.mark.skipif(
    not HAS_CHROMADB,
    reason="chromadb not installed"
)


@pytest.fixture
def test_persist_dir(tmp_path):
    """创建临时持久化目录。"""
    persist_dir = tmp_path / "test_chroma_db"
    yield str(persist_dir)
    # 清理：尝试删除，失败则忽略（Windows 文件锁定问题）
    if persist_dir.exists():
        try:
            # 给一点时间让文件句柄释放
            time.sleep(0.1)
            shutil.rmtree(persist_dir)
        except PermissionError:
            # Windows 上 ChromaDB 可能保持文件锁定，忽略清理错误
            pass


@pytest.fixture
def chroma_store(test_persist_dir):
    """创建 ChromaStore 实例用于测试。"""
    store = ChromaStore(
        collection_name="test_collection",
        persist_directory=test_persist_dir
    )
    yield store
    # 清理：关闭客户端连接
    try:
        if hasattr(store, '_client'):
            # ChromaDB 客户端没有显式的 close 方法，删除引用即可
            del store._client
            del store._collection
    except:
        pass


@pytest.fixture
def sample_records() -> List[VectorRecord]:
    """创建示例记录用于测试。"""
    return [
        VectorRecord(
            id="doc1",
            vector=[0.1, 0.2, 0.3],
            text="This is the first document about AI.",
            metadata={"topic": "ai", "category": "tech"}
        ),
        VectorRecord(
            id="doc2",
            vector=[0.2, 0.3, 0.4],
            text="This is the second document about machine learning.",
            metadata={"topic": "ml", "category": "tech"}
        ),
        VectorRecord(
            id="doc3",
            vector=[0.9, 0.8, 0.7],
            text="This is a document about cooking recipes.",
            metadata={"topic": "cooking", "category": "food"}
        ),
    ]


class TestChromaStoreInitialization:
    """测试 ChromaStore 初始化。"""
    
    def test_initialization_with_defaults(self, test_persist_dir):
        """测试使用默认参数初始化。"""
        store = ChromaStore(persist_directory=test_persist_dir)
        
        assert store.collection_name == "default"
        assert store.persist_directory == test_persist_dir
        assert store._client is not None
        assert store._collection is not None
    
    def test_initialization_with_custom_params(self, test_persist_dir):
        """测试使用自定义参数初始化。"""
        store = ChromaStore(
            collection_name="custom_collection",
            persist_directory=test_persist_dir
        )
        
        assert store.collection_name == "custom_collection"
        assert store.persist_directory == test_persist_dir
    
    def test_persist_directory_created(self, tmp_path):
        """测试持久化目录自动创建。"""
        persist_dir = tmp_path / "new_chroma_db"
        assert not persist_dir.exists()
        
        store = ChromaStore(persist_directory=str(persist_dir))
        
        assert persist_dir.exists()
        store.clear()


class TestChromaStoreUpsert:
    """测试 ChromaStore upsert 操作。"""
    
    def test_upsert_single_record(self, chroma_store):
        """测试插入单条记录。"""
        record = VectorRecord(
            id="test1",
            vector=[0.1, 0.2, 0.3],
            text="Test document",
            metadata={"key": "value"}
        )
        
        chroma_store.upsert([record])
        
        # 验证记录已插入
        assert chroma_store.count() == 1
    
    def test_upsert_multiple_records(self, chroma_store, sample_records):
        """测试插入多条记录。"""
        chroma_store.upsert(sample_records)
        
        assert chroma_store.count() == 3
    
    def test_upsert_empty_list(self, chroma_store):
        """测试插入空列表不报错。"""
        chroma_store.upsert([])
        
        assert chroma_store.count() == 0
    
    def test_upsert_update_existing_record(self, chroma_store):
        """测试更新已存在的记录。"""
        # 插入初始记录
        record1 = VectorRecord(
            id="update_test",
            vector=[0.1, 0.2, 0.3],
            text="Original text",
            metadata={"version": "1"}
        )
        chroma_store.upsert([record1])
        
        # 更新记录
        record2 = VectorRecord(
            id="update_test",
            vector=[0.4, 0.5, 0.6],
            text="Updated text",
            metadata={"version": "2"}
        )
        chroma_store.upsert([record2])
        
        # 验证仅有一条记录
        assert chroma_store.count() == 1
        
        # 验证内容已更新
        results = chroma_store.get(["update_test"])
        assert len(results) == 1
        assert results[0].text == "Updated text"
        assert results[0].metadata["version"] == "2"


class TestChromaStoreQuery:
    """测试 ChromaStore query 操作。"""
    
    def test_query_returns_results(self, chroma_store, sample_records):
        """测试查询返回结果。"""
        chroma_store.upsert(sample_records)
        
        # 查询与 doc1 相似的向量
        query_vector = [0.1, 0.2, 0.3]
        results = chroma_store.query(query_vector, top_k=2)
        
        assert len(results) == 2
        assert all(isinstance(r, QueryResult) for r in results)
        # 第一个结果应该是 doc1（完全匹配）
        assert results[0].id == "doc1"
    
    def test_query_with_top_k(self, chroma_store, sample_records):
        """测试 top_k 参数。"""
        chroma_store.upsert(sample_records)
        
        query_vector = [0.5, 0.5, 0.5]
        results = chroma_store.query(query_vector, top_k=1)
        
        assert len(results) == 1
    
    def test_query_with_metadata_filter(self, chroma_store, sample_records):
        """测试元数据过滤。"""
        chroma_store.upsert(sample_records)
        
        query_vector = [0.5, 0.5, 0.5]
        # 过滤只返回 tech 类别的文档
        results = chroma_store.query(
            query_vector,
            top_k=10,
            filters={"category": "tech"}
        )
        
        # 应该只返回 doc1 和 doc2
        assert len(results) == 2
        assert all(r.metadata["category"] == "tech" for r in results)
    
    def test_query_empty_vector_raises_error(self, chroma_store):
        """测试空向量查询抛出错误。"""
        with pytest.raises(VectorStoreError, match="cannot be empty"):
            chroma_store.query([], top_k=1)
    
    def test_query_invalid_top_k_raises_error(self, chroma_store):
        """测试无效 top_k 抛出错误。"""
        with pytest.raises(VectorStoreError, match="must be greater than 0"):
            chroma_store.query([0.1, 0.2, 0.3], top_k=0)
    
    def test_query_result_has_score(self, chroma_store, sample_records):
        """测试查询结果包含相似度分数。"""
        chroma_store.upsert(sample_records)
        
        query_vector = [0.1, 0.2, 0.3]
        results = chroma_store.query(query_vector, top_k=1)
        
        assert results[0].score > 0
        assert results[0].score <= 1.0


class TestChromaStoreDelete:
    """测试 ChromaStore delete 操作。"""
    
    def test_delete_single_record(self, chroma_store, sample_records):
        """测试删除单条记录。"""
        chroma_store.upsert(sample_records)
        assert chroma_store.count() == 3
        
        chroma_store.delete(["doc1"])
        
        assert chroma_store.count() == 2
    
    def test_delete_multiple_records(self, chroma_store, sample_records):
        """测试删除多条记录。"""
        chroma_store.upsert(sample_records)
        
        chroma_store.delete(["doc1", "doc2"])
        
        assert chroma_store.count() == 1
    
    def test_delete_empty_list(self, chroma_store, sample_records):
        """测试删除空列表不报错。"""
        chroma_store.upsert(sample_records)
        initial_count = chroma_store.count()
        
        chroma_store.delete([])
        
        assert chroma_store.count() == initial_count
    
    def test_delete_non_existent_id(self, chroma_store, sample_records):
        """测试删除不存在的 ID 不报错。"""
        chroma_store.upsert(sample_records)
        
        # ChromaDB 对不存在的 ID 不报错
        chroma_store.delete(["non_existent_id"])
        
        assert chroma_store.count() == 3


class TestChromaStoreGet:
    """测试 ChromaStore get 操作。"""
    
    def test_get_single_record(self, chroma_store, sample_records):
        """测试获取单条记录。"""
        chroma_store.upsert(sample_records)
        
        results = chroma_store.get(["doc1"])
        
        assert len(results) == 1
        assert results[0].id == "doc1"
        assert results[0].text == "This is the first document about AI."
    
    def test_get_multiple_records(self, chroma_store, sample_records):
        """测试获取多条记录。"""
        chroma_store.upsert(sample_records)
        
        results = chroma_store.get(["doc1", "doc3"])
        
        assert len(results) == 2
        ids = [r.id for r in results]
        assert "doc1" in ids
        assert "doc3" in ids
    
    def test_get_empty_list_returns_empty(self, chroma_store):
        """测试获取空列表返回空结果。"""
        results = chroma_store.get([])
        
        assert len(results) == 0
    
    def test_get_non_existent_id(self, chroma_store, sample_records):
        """测试获取不存在的 ID 返回空结果。"""
        chroma_store.upsert(sample_records)
        
        results = chroma_store.get(["non_existent_id"])
        
        # ChromaDB 对不存在的 ID 返回空结果
        assert len(results) == 0


class TestChromaStoreCount:
    """测试 ChromaStore count 操作。"""
    
    def test_count_empty_collection(self, chroma_store):
        """测试空集合计数。"""
        assert chroma_store.count() == 0
    
    def test_count_after_upsert(self, chroma_store, sample_records):
        """测试插入后计数。"""
        chroma_store.upsert(sample_records)
        
        assert chroma_store.count() == 3
    
    def test_count_after_delete(self, chroma_store, sample_records):
        """测试删除后计数。"""
        chroma_store.upsert(sample_records)
        chroma_store.delete(["doc1"])
        
        assert chroma_store.count() == 2


class TestChromaStoreClear:
    """测试 ChromaStore clear 操作。"""
    
    def test_clear_collection(self, chroma_store, sample_records):
        """测试清空集合。"""
        chroma_store.upsert(sample_records)
        assert chroma_store.count() == 3
        
        chroma_store.clear()
        
        assert chroma_store.count() == 0
    
    def test_clear_empty_collection(self, chroma_store):
        """测试清空空集合不报错。"""
        chroma_store.clear()
        
        assert chroma_store.count() == 0


class TestChromaStoreRoundtrip:
    """测试 ChromaStore 完整的 roundtrip 流程。"""
    
    def test_upsert_query_roundtrip(self, chroma_store):
        """测试插入-查询完整流程。"""
        # 1. 插入记录
        records = [
            VectorRecord(
                id="rt1",
                vector=[1.0, 0.0, 0.0],
                text="Document about topic A",
                metadata={"topic": "A"}
            ),
            VectorRecord(
                id="rt2",
                vector=[0.0, 1.0, 0.0],
                text="Document about topic B",
                metadata={"topic": "B"}
            ),
            VectorRecord(
                id="rt3",
                vector=[0.0, 0.0, 1.0],
                text="Document about topic C",
                metadata={"topic": "C"}
            ),
        ]
        chroma_store.upsert(records)
        
        # 2. 查询
        query_vector = [1.0, 0.0, 0.0]  # 应该最匹配 rt1
        results = chroma_store.query(query_vector, top_k=3)
        
        # 3. 验证结果
        assert len(results) == 3
        assert results[0].id == "rt1"  # 最相似
        assert results[0].text == "Document about topic A"
        assert results[0].metadata["topic"] == "A"
        
        # 4. 获取记录
        fetched = chroma_store.get(["rt1", "rt2"])
        assert len(fetched) == 2
        
        # 5. 删除记录
        chroma_store.delete(["rt1"])
        assert chroma_store.count() == 2
        
        # 6. 再次查询
        results_after_delete = chroma_store.query(query_vector, top_k=3)
        assert len(results_after_delete) == 2
        assert all(r.id != "rt1" for r in results_after_delete)


class TestChromaStorePersistence:
    """测试 ChromaStore 持久化功能。"""
    
    def test_persistence_across_instances(self, test_persist_dir, sample_records):
        """测试数据在实例间持久化。"""
        # 第一个实例：插入数据
        store1 = ChromaStore(
            collection_name="persist_test",
            persist_directory=test_persist_dir
        )
        store1.upsert(sample_records)
        initial_count = store1.count()
        assert initial_count == 3
        
        # 销毁第一个实例（模拟进程重启）
        del store1
        
        # 第二个实例：验证数据仍存在
        store2 = ChromaStore(
            collection_name="persist_test",
            persist_directory=test_persist_dir
        )
        assert store2.count() == initial_count
        
        # 验证可以查询到数据
        results = store2.query([0.1, 0.2, 0.3], top_k=1)
        assert len(results) > 0
        
        # 清理
        store2.clear()


class TestFactoryIntegration:
    """测试 ChromaStore 与工厂的集成。"""
    
    def test_factory_creates_chroma_store(self, test_persist_dir):
        """测试工厂能正确创建 ChromaStore 实例。"""
        settings = VectorStoreSettings(
            provider="chroma",
            collection_name="factory_test",
            persist_directory=test_persist_dir
        )
        
        store = VectorStoreFactory.create_from_vector_store_settings(settings)
        
        assert isinstance(store, ChromaStore)
        assert store.collection_name == "factory_test"
        assert store.persist_directory == test_persist_dir
        
        # 清理
        store.clear()
    
    def test_factory_chroma_store_functional(self, test_persist_dir):
        """测试通过工厂创建的实例功能正常。"""
        settings = VectorStoreSettings(
            provider="chroma",
            collection_name="functional_test",
            persist_directory=test_persist_dir
        )
        
        store = VectorStoreFactory.create_from_vector_store_settings(settings)
        
        # 插入记录
        record = VectorRecord(
            id="factory1",
            vector=[0.1, 0.2, 0.3],
            text="Factory test document",
            metadata={"source": "factory"}
        )
        store.upsert([record])
        
        # 查询验证
        results = store.query([0.1, 0.2, 0.3], top_k=1)
        assert len(results) == 1
        assert results[0].id == "factory1"
        
        # 清理
        store.clear()
