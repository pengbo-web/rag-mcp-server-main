"""
LLM 工厂测试。

测试 LLMFactory 的注册、创建和路由逻辑。
使用 Fake 实现避免真实 API 调用。
"""

import pytest

from core import Settings, LLMSettings
from libs.llm import BaseLLM, Message, ChatResponse, LLMError, LLMFactory


class FakeLLM(BaseLLM):
    """用于测试的假 LLM 实现。"""
    
    def __init__(self, model: str, **kwargs):
        """初始化假 LLM。"""
        super().__init__(model, **kwargs)
        self.call_count = 0
        self.last_messages = None
    
    def chat(self, messages: list[Message], **kwargs) -> ChatResponse:
        """模拟聊天补全。"""
        self.call_count += 1
        self.last_messages = messages
        
        # 返回固定响应
        return ChatResponse(
            content="This is a fake response",
            model=self.model,
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        )


class AnotherFakeLLM(BaseLLM):
    """另一个假 LLM 实现（用于测试多提供商）。"""
    
    def chat(self, messages: list[Message], **kwargs) -> ChatResponse:
        """模拟聊天补全。"""
        return ChatResponse(
            content="Another fake response",
            model=self.model
        )


class TestLLMFactory:
    """测试 LLM 工厂类。"""
    
    def setup_method(self):
        """每个测试前清空注册表。"""
        LLMFactory.clear_registry()
    
    def teardown_method(self):
        """每个测试后清空注册表。"""
        LLMFactory.clear_registry()
    
    @pytest.mark.unit
    def test_register_provider(self):
        """测试注册提供商。"""
        LLMFactory.register("fake", FakeLLM)
        
        assert LLMFactory.is_registered("fake")
        assert "fake" in LLMFactory.list_providers()
    
    @pytest.mark.unit
    def test_register_duplicate_provider_raises_error(self):
        """测试重复注册提供商会抛出错误。"""
        LLMFactory.register("fake", FakeLLM)
        
        with pytest.raises(ValueError, match="already registered"):
            LLMFactory.register("fake", AnotherFakeLLM)
    
    @pytest.mark.unit
    def test_register_non_basellm_raises_error(self):
        """测试注册非 BaseLLM 子类会抛出错误。"""
        class NotAnLLM:
            pass
        
        with pytest.raises(TypeError, match="must inherit from BaseLLM"):
            LLMFactory.register("invalid", NotAnLLM)
    
    @pytest.mark.unit
    def test_create_from_llm_settings(self):
        """测试从 LLM 设置创建实例。"""
        LLMFactory.register("fake", FakeLLM)
        
        llm_settings = LLMSettings(
            provider="fake",
            model="fake-model",
            temperature=0.5,
            max_tokens=1024
        )
        
        llm = LLMFactory.create_from_llm_settings(llm_settings)
        
        assert isinstance(llm, FakeLLM)
        assert llm.model == "fake-model"
        assert llm.temperature == 0.5
        assert llm.max_tokens == 1024
    
    @pytest.mark.unit
    def test_create_from_settings(self, sample_settings_dict):
        """测试从完整 Settings 创建实例。"""
        LLMFactory.register("openai", FakeLLM)
        
        # 修改设置使用 fake provider
        sample_settings_dict["llm"]["provider"] = "openai"
        
        from core.settings import Settings, _parse_section, EmbeddingSettings, VectorStoreSettings, RetrievalSettings, RerankSettings, EvaluationSettings, ObservabilitySettings, IngestionSettings
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
        
        llm = LLMFactory.create(settings)
        
        assert isinstance(llm, FakeLLM)
        # 检查模型名称是从配置中正确读取的
        assert llm.model == sample_settings_dict["llm"]["model"]
    
    @pytest.mark.unit
    def test_create_with_unknown_provider_raises_error(self):
        """测试使用未注册的提供商会抛出错误。"""
        llm_settings = LLMSettings(
            provider="unknown",
            model="test-model"
        )
        
        with pytest.raises(LLMError, match="Unknown LLM provider"):
            LLMFactory.create_from_llm_settings(llm_settings)
    
    @pytest.mark.unit
    def test_provider_case_insensitive(self):
        """测试提供商名称不区分大小写。"""
        LLMFactory.register("fake", FakeLLM)
        
        llm_settings = LLMSettings(
            provider="FAKE",  # 大写
            model="test-model"
        )
        
        llm = LLMFactory.create_from_llm_settings(llm_settings)
        assert isinstance(llm, FakeLLM)
    
    @pytest.mark.unit
    def test_multiple_providers(self):
        """测试注册和使用多个提供商。"""
        LLMFactory.register("fake1", FakeLLM)
        LLMFactory.register("fake2", AnotherFakeLLM)
        
        llm1 = LLMFactory.create_from_llm_settings(
            LLMSettings(provider="fake1", model="model1")
        )
        llm2 = LLMFactory.create_from_llm_settings(
            LLMSettings(provider="fake2", model="model2")
        )
        
        assert isinstance(llm1, FakeLLM)
        assert isinstance(llm2, AnotherFakeLLM)
        assert llm1.model == "model1"
        assert llm2.model == "model2"
    
    @pytest.mark.unit
    def test_llm_can_be_called(self):
        """测试创建的 LLM 实例可以正常调用。"""
        LLMFactory.register("fake", FakeLLM)
        
        llm = LLMFactory.create_from_llm_settings(
            LLMSettings(provider="fake", model="test-model")
        )
        
        messages = [Message(role="user", content="Hello")]
        response = llm.chat(messages)
        
        assert response.content == "This is a fake response"
        assert response.model == "test-model"
        assert isinstance(llm, FakeLLM)
        assert llm.call_count == 1
    
    @pytest.mark.unit
    def test_chat_simple_returns_string(self):
        """测试 chat_simple 方法返回字符串。"""
        LLMFactory.register("fake", FakeLLM)
        
        llm = LLMFactory.create_from_llm_settings(
            LLMSettings(provider="fake", model="test-model")
        )
        
        messages = [Message(role="user", content="Hello")]
        content = llm.chat_simple(messages)
        
        assert isinstance(content, str)
        assert content == "This is a fake response"
    
    @pytest.mark.unit
    def test_list_providers_empty_when_none_registered(self):
        """测试未注册任何提供商时列表为空。"""
        assert LLMFactory.list_providers() == []
    
    @pytest.mark.unit
    def test_list_providers_returns_all_registered(self):
        """测试列出所有已注册的提供商。"""
        LLMFactory.register("fake1", FakeLLM)
        LLMFactory.register("fake2", AnotherFakeLLM)
        
        providers = LLMFactory.list_providers()
        
        assert len(providers) == 2
        assert "fake1" in providers
        assert "fake2" in providers
