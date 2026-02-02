"""
Splitter 工厂测试。

测试 SplitterFactory 的注册、创建和路由逻辑。
使用 Fake 实现避免复杂的文本处理逻辑。
"""

import pytest

from core import Settings, IngestionSettings
from libs.splitter import BaseSplitter, TextChunk, SplitterError, SplitterFactory


class FakeSplitter(BaseSplitter):
    """用于测试的假 Splitter 实现。"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, **kwargs):
        """初始化假 Splitter。"""
        super().__init__(chunk_size, chunk_overlap, **kwargs)
        self.call_count = 0
        self.last_text = None
    
    def split_text(self, text: str, **kwargs) -> list[str]:
        """模拟切分（简单按空格分割）。"""
        self.call_count += 1
        self.last_text = text
        
        if not text:
            return []
        
        # 简单实现：按空格分词，然后按 chunk_size 分组
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            word_size = len(word) + 1  # +1 for space
            if current_size + word_size > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                # 保留重叠
                overlap_words = int(len(current_chunk) * self.chunk_overlap / self.chunk_size)
                current_chunk = current_chunk[-overlap_words:] if overlap_words > 0 else []
                current_size = sum(len(w) + 1 for w in current_chunk)
            
            current_chunk.append(word)
            current_size += word_size
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks


class AnotherFakeSplitter(BaseSplitter):
    """另一个假 Splitter 实现（用于测试多策略）。"""
    
    def split_text(self, text: str, **kwargs) -> list[str]:
        """模拟切分（固定长度）。"""
        if not text:
            return []
        
        chunks = []
        for i in range(0, len(text), self.chunk_size):
            chunks.append(text[i:i + self.chunk_size])
        
        return chunks


class TestSplitterFactory:
    """测试 Splitter 工厂类。"""
    
    def setup_method(self):
        """每个测试前清空注册表。"""
        SplitterFactory.clear_registry()
    
    def teardown_method(self):
        """每个测试后清空注册表。"""
        SplitterFactory.clear_registry()
    
    @pytest.mark.unit
    def test_register_strategy(self):
        """测试注册策略。"""
        SplitterFactory.register("fake", FakeSplitter)
        
        assert SplitterFactory.is_registered("fake")
        assert "fake" in SplitterFactory.list_strategies()
    
    @pytest.mark.unit
    def test_register_duplicate_strategy_raises_error(self):
        """测试重复注册策略会抛出错误。"""
        SplitterFactory.register("fake", FakeSplitter)
        
        with pytest.raises(ValueError, match="already registered"):
            SplitterFactory.register("fake", AnotherFakeSplitter)
    
    @pytest.mark.unit
    def test_register_non_basesplitter_raises_error(self):
        """测试注册非 BaseSplitter 子类会抛出错误。"""
        class NotASplitter:
            pass
        
        with pytest.raises(TypeError, match="must inherit from BaseSplitter"):
            SplitterFactory.register("invalid", NotASplitter)
    
    @pytest.mark.unit
    def test_create_from_ingestion_settings(self):
        """测试从 Ingestion 设置创建实例。"""
        SplitterFactory.register("fake", FakeSplitter)
        
        ingestion_settings = IngestionSettings(
            chunk_size=500,
            chunk_overlap=100,
            batch_size=50
        )
        
        splitter = SplitterFactory.create_from_ingestion_settings(
            ingestion_settings, 
            strategy="fake"
        )
        
        assert isinstance(splitter, FakeSplitter)
        assert splitter.chunk_size == 500
        assert splitter.chunk_overlap == 100
    
    @pytest.mark.unit
    def test_create_from_settings(self, sample_settings_dict):
        """测试从完整 Settings 创建实例。"""
        SplitterFactory.register("recursive", FakeSplitter)
        
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
        
        splitter = SplitterFactory.create(settings, strategy="recursive")
        
        assert isinstance(splitter, FakeSplitter)
        assert splitter.chunk_size == sample_settings_dict["ingestion"]["chunk_size"]
    
    @pytest.mark.unit
    def test_create_with_params(self):
        """测试使用指定参数创建实例。"""
        SplitterFactory.register("fake", FakeSplitter)
        
        splitter = SplitterFactory.create_with_params(
            strategy="fake",
            chunk_size=800,
            chunk_overlap=150
        )
        
        assert isinstance(splitter, FakeSplitter)
        assert splitter.chunk_size == 800
        assert splitter.chunk_overlap == 150
    
    @pytest.mark.unit
    def test_create_with_unknown_strategy_raises_error(self):
        """测试使用未注册的策略会抛出错误。"""
        with pytest.raises(SplitterError, match="Unknown Splitter strategy"):
            SplitterFactory.create_with_params(strategy="unknown", chunk_size=1000)
    
    @pytest.mark.unit
    def test_strategy_case_insensitive(self):
        """测试策略名称不区分大小写。"""
        SplitterFactory.register("fake", FakeSplitter)
        
        splitter = SplitterFactory.create_with_params(strategy="FAKE", chunk_size=1000)
        assert isinstance(splitter, FakeSplitter)
    
    @pytest.mark.unit
    def test_multiple_strategies(self):
        """测试注册和使用多个策略。"""
        SplitterFactory.register("fake1", FakeSplitter)
        SplitterFactory.register("fake2", AnotherFakeSplitter)
        
        spl1 = SplitterFactory.create_with_params(strategy="fake1", chunk_size=500)
        spl2 = SplitterFactory.create_with_params(strategy="fake2", chunk_size=500)
        
        assert isinstance(spl1, FakeSplitter)
        assert isinstance(spl2, AnotherFakeSplitter)
    
    @pytest.mark.unit
    def test_splitter_can_be_called(self):
        """测试创建的 Splitter 实例可以正常调用。"""
        SplitterFactory.register("fake", FakeSplitter)
        
        splitter = SplitterFactory.create_with_params(strategy="fake", chunk_size=50, chunk_overlap=10)
        
        text = "This is a test text that will be split into chunks"
        chunks = splitter.split_text(text)
        
        assert len(chunks) > 0
        assert all(isinstance(c, str) for c in chunks)
        assert isinstance(splitter, FakeSplitter)
        assert splitter.call_count == 1
    
    @pytest.mark.unit
    def test_split_text_with_metadata(self):
        """测试带元数据的切分。"""
        SplitterFactory.register("fake", FakeSplitter)
        
        splitter = SplitterFactory.create_with_params(strategy="fake", chunk_size=50, chunk_overlap=10)
        
        text = "This is a test text"
        metadata = {"source": "test.txt", "page": 1}
        chunks = splitter.split_text_with_metadata(text, metadata)
        
        assert all(isinstance(c, TextChunk) for c in chunks)
        assert all(c.metadata["source"] == "test.txt" for c in chunks)
        assert all("chunk_index" in c.metadata for c in chunks)
        assert all("total_chunks" in c.metadata for c in chunks)
    
    @pytest.mark.unit
    def test_get_num_chunks(self):
        """测试估算块数量。"""
        SplitterFactory.register("fake", FakeSplitter)
        
        splitter = SplitterFactory.create_with_params(strategy="fake", chunk_size=50, chunk_overlap=10)
        
        text = "This is a test text that will be split"
        num_chunks = splitter.get_num_chunks(text)
        
        assert isinstance(num_chunks, int)
        assert num_chunks > 0
    
    @pytest.mark.unit
    def test_invalid_chunk_size_raises_error(self):
        """测试无效的 chunk_size 会抛出错误。"""
        SplitterFactory.register("fake", FakeSplitter)
        
        with pytest.raises(SplitterError, match="chunk_size must be positive"):
            SplitterFactory.create_with_params(strategy="fake", chunk_size=0)
    
    @pytest.mark.unit
    def test_invalid_chunk_overlap_raises_error(self):
        """测试无效的 chunk_overlap 会抛出错误。"""
        SplitterFactory.register("fake", FakeSplitter)
        
        with pytest.raises(SplitterError, match="chunk_overlap cannot be negative"):
            SplitterFactory.create_with_params(strategy="fake", chunk_size=1000, chunk_overlap=-1)
    
    @pytest.mark.unit
    def test_overlap_larger_than_size_raises_error(self):
        """测试 overlap 大于等于 size 会抛出错误。"""
        SplitterFactory.register("fake", FakeSplitter)
        
        with pytest.raises(SplitterError, match="must be less than chunk_size"):
            SplitterFactory.create_with_params(strategy="fake", chunk_size=100, chunk_overlap=100)
    
    @pytest.mark.unit
    def test_list_strategies_empty_when_none_registered(self):
        """测试未注册任何策略时列表为空。"""
        assert SplitterFactory.list_strategies() == []
    
    @pytest.mark.unit
    def test_list_strategies_returns_all_registered(self):
        """测试列出所有已注册的策略。"""
        SplitterFactory.register("fake1", FakeSplitter)
        SplitterFactory.register("fake2", AnotherFakeSplitter)
        
        strategies = SplitterFactory.list_strategies()
        
        assert len(strategies) == 2
        assert "fake1" in strategies
        assert "fake2" in strategies
