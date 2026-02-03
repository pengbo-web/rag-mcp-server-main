"""
OpenAI-Compatible LLM 提供商测试。

测试 OpenAI、Azure、DeepSeek LLM 实现的基本功能。
使用 Mock 避免真实 API 调用。
"""

import os
from unittest.mock import Mock, patch, MagicMock
import pytest

from libs.llm import (
    Message,
    ChatResponse,
    LLMError,
    LLMFactory,
    OpenAILLM,
    AzureLLM,
    DeepSeekLLM,
)


class TestOpenAILLM:
    """测试 OpenAI LLM 实现。"""
    
    @pytest.mark.unit
    def test_initialization_with_api_key(self):
        """测试使用显式 API 密钥初始化。"""
        with patch("libs.llm.openai_llm.OpenAI"):
            llm = OpenAILLM(
                model="gpt-4o-mini",
                api_key="test-api-key"
            )
            
            assert llm.model == "gpt-4o-mini"
            assert llm.api_key == "test-api-key"
    
    @pytest.mark.unit
    def test_initialization_from_env(self, monkeypatch):
        """测试从环境变量读取 API 密钥。"""
        monkeypatch.setenv("OPENAI_API_KEY", "env-api-key")
        
        with patch("libs.llm.openai_llm.OpenAI"):
            llm = OpenAILLM(model="gpt-4o-mini")
            
            assert llm.api_key == "env-api-key"
    
    @pytest.mark.unit
    def test_initialization_without_api_key_raises_error(self, monkeypatch):
        """测试缺少 API 密钥时抛出错误。"""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        
        with pytest.raises(LLMError, match="API key is required"):
            OpenAILLM(model="gpt-4o-mini")
    
    @pytest.mark.unit
    def test_chat_success(self):
        """测试成功的聊天补全。"""
        # Mock OpenAI 客户端
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Hello, world!"
        mock_response.model = "gpt-4o-mini"
        mock_response.usage = Mock(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15
        )
        mock_client.chat.completions.create.return_value = mock_response
        
        with patch("libs.llm.openai_llm.OpenAI", return_value=mock_client):
            llm = OpenAILLM(model="gpt-4o-mini", api_key="test-key")
            
            messages = [
                Message(role="system", content="You are a helpful assistant."),
                Message(role="user", content="Hello!")
            ]
            
            response = llm.chat(messages)
            
            assert response.content == "Hello, world!"
            assert response.model == "gpt-4o-mini"
            assert response.usage == {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
    
    @pytest.mark.unit
    def test_chat_empty_messages_raises_error(self):
        """测试空消息列表抛出错误。"""
        with patch("libs.llm.openai_llm.OpenAI"):
            llm = OpenAILLM(model="gpt-4o-mini", api_key="test-key")
            
            with pytest.raises(LLMError, match="cannot be empty"):
                llm.chat([])
    
    @pytest.mark.unit
    def test_chat_invalid_message_type_raises_error(self):
        """测试无效消息类型抛出错误。"""
        with patch("libs.llm.openai_llm.OpenAI"):
            llm = OpenAILLM(model="gpt-4o-mini", api_key="test-key")
            
            with pytest.raises(LLMError, match="must be a Message object"):
                llm.chat([{"role": "user", "content": "test"}])
    
    @pytest.mark.unit
    def test_chat_invalid_role_raises_error(self):
        """测试无效消息角色抛出错误。"""
        with patch("libs.llm.openai_llm.OpenAI"):
            llm = OpenAILLM(model="gpt-4o-mini", api_key="test-key")
            
            invalid_message = Message(role="invalid", content="test")
            
            with pytest.raises(LLMError, match="invalid role"):
                llm.chat([invalid_message])
    
    @pytest.mark.unit
    def test_chat_authentication_error(self):
        """测试认证失败场景。"""
        from openai import AuthenticationError
        
        # Mock response 对象
        mock_response = Mock()
        mock_response.status_code = 401
        
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = AuthenticationError(
            "Invalid API key",
            response=mock_response,
            body={"error": {"message": "Invalid API key"}}
        )
        
        with patch("libs.llm.openai_llm.OpenAI", return_value=mock_client):
            llm = OpenAILLM(model="gpt-4o-mini", api_key="invalid-key")
            
            messages = [Message(role="user", content="test")]
            
            with pytest.raises(LLMError, match="authentication failed"):
                llm.chat(messages)
    
    @pytest.mark.unit
    def test_chat_rate_limit_error(self):
        """测试速率限制场景。"""
        from openai import RateLimitError
        
        # Mock response 对象
        mock_response = Mock()
        mock_response.status_code = 429
        
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = RateLimitError(
            "Rate limit exceeded",
            response=mock_response,
            body={"error": {"message": "Rate limit exceeded"}}
        )
        
        with patch("libs.llm.openai_llm.OpenAI", return_value=mock_client):
            llm = OpenAILLM(model="gpt-4o-mini", api_key="test-key")
            
            messages = [Message(role="user", content="test")]
            
            with pytest.raises(LLMError, match="rate limit exceeded"):
                llm.chat(messages)


class TestAzureLLM:
    """测试 Azure LLM 实现。"""
    
    @pytest.mark.unit
    def test_initialization_with_required_params(self):
        """测试使用必需参数初始化。"""
        with patch("libs.llm.azure_llm.AzureOpenAI"):
            llm = AzureLLM(
                model="gpt-4",
                api_key="test-key",
                base_url="https://test.openai.azure.com"
            )
            
            assert llm.model == "gpt-4"
            assert llm.api_key == "test-key"
            assert llm.azure_endpoint == "https://test.openai.azure.com"
    
    @pytest.mark.unit
    def test_initialization_from_env(self, monkeypatch):
        """测试从环境变量读取配置。"""
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "env-key")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://env.openai.azure.com")
        
        with patch("libs.llm.azure_llm.AzureOpenAI"):
            llm = AzureLLM(model="gpt-4")
            
            assert llm.api_key == "env-key"
            assert llm.azure_endpoint == "https://env.openai.azure.com"
    
    @pytest.mark.unit
    def test_initialization_without_api_key_raises_error(self, monkeypatch):
        """测试缺少 API 密钥时抛出错误。"""
        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com")
        
        with pytest.raises(LLMError, match="API key is required"):
            AzureLLM(model="gpt-4")
    
    @pytest.mark.unit
    def test_initialization_without_endpoint_raises_error(self, monkeypatch):
        """测试缺少 endpoint 时抛出错误。"""
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
        monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
        
        with pytest.raises(LLMError, match="endpoint is required"):
            AzureLLM(model="gpt-4")
    
    @pytest.mark.unit
    def test_chat_success(self):
        """测试成功的聊天补全。"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Azure response"
        mock_response.model = "gpt-4"
        mock_response.usage = None
        mock_client.chat.completions.create.return_value = mock_response
        
        with patch("libs.llm.azure_llm.AzureOpenAI", return_value=mock_client):
            llm = AzureLLM(
                model="gpt-4",
                api_key="test-key",
                base_url="https://test.openai.azure.com"
            )
            
            messages = [Message(role="user", content="Hello")]
            response = llm.chat(messages)
            
            assert response.content == "Azure response"
            assert response.model == "gpt-4"


class TestDeepSeekLLM:
    """测试 DeepSeek LLM 实现。"""
    
    @pytest.mark.unit
    def test_initialization_with_api_key(self):
        """测试使用显式 API 密钥初始化。"""
        with patch("libs.llm.openai_llm.OpenAI"):
            llm = DeepSeekLLM(api_key="test-key")
            
            assert llm.model == "deepseek-chat"
            assert llm.api_key == "test-key"
    
    @pytest.mark.unit
    def test_initialization_from_deepseek_env(self, monkeypatch):
        """测试从 DEEPSEEK_API_KEY 环境变量读取。"""
        monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        
        with patch("libs.llm.openai_llm.OpenAI"):
            llm = DeepSeekLLM()
            
            assert llm.api_key == "deepseek-key"
    
    @pytest.mark.unit
    def test_initialization_fallback_to_openai_env(self, monkeypatch):
        """测试回退到 OPENAI_API_KEY 环境变量。"""
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
        
        with patch("libs.llm.openai_llm.OpenAI"):
            llm = DeepSeekLLM()
            
            assert llm.api_key == "openai-key"
    
    @pytest.mark.unit
    def test_default_base_url(self):
        """测试默认使用 DeepSeek API 端点。"""
        with patch("libs.llm.openai_llm.OpenAI") as mock_openai:
            llm = DeepSeekLLM(api_key="test-key")
            
            # 验证调用时使用了正确的 base_url
            mock_openai.assert_called_once()
            call_kwargs = mock_openai.call_args[1]
            assert call_kwargs["base_url"] == "https://api.deepseek.com"


class TestFactoryIntegration:
    """测试工厂模式集成。"""
    
    def setup_method(self):
        """每个测试前确保提供商已注册。"""
        # 如果未注册，则注册（避免重复注册错误）
        if not LLMFactory.is_registered("openai"):
            LLMFactory.register("openai", OpenAILLM)
        if not LLMFactory.is_registered("azure"):
            LLMFactory.register("azure", AzureLLM)
        if not LLMFactory.is_registered("deepseek"):
            LLMFactory.register("deepseek", DeepSeekLLM)
    
    @pytest.mark.unit
    def test_openai_provider_registered(self):
        """测试 OpenAI 提供商已注册。"""
        assert LLMFactory.is_registered("openai")
    
    @pytest.mark.unit
    def test_azure_provider_registered(self):
        """测试 Azure 提供商已注册。"""
        assert LLMFactory.is_registered("azure")
    
    @pytest.mark.unit
    def test_deepseek_provider_registered(self):
        """测试 DeepSeek 提供商已注册。"""
        assert LLMFactory.is_registered("deepseek")
    
    @pytest.mark.unit
    @patch("libs.llm.openai_llm.OpenAI")
    def test_create_openai_from_factory(self, mock_openai):
        """测试通过工厂创建 OpenAI LLM。"""
        from core import LLMSettings
        
        settings = LLMSettings(
            provider="openai",
            model="gpt-4o-mini",
            api_key="test-key"
        )
        
        llm = LLMFactory.create_from_llm_settings(settings)
        
        assert isinstance(llm, OpenAILLM)
        assert llm.model == "gpt-4o-mini"
    
    @pytest.mark.unit
    @patch("libs.llm.azure_llm.AzureOpenAI")
    def test_create_azure_from_factory(self, mock_azure):
        """测试通过工厂创建 Azure LLM。"""
        from core import LLMSettings
        
        settings = LLMSettings(
            provider="azure",
            model="gpt-4",
            api_key="test-key",
            base_url="https://test.openai.azure.com"
        )
        
        llm = LLMFactory.create_from_llm_settings(settings)
        
        assert isinstance(llm, AzureLLM)
        assert llm.model == "gpt-4"
    
    @pytest.mark.unit
    @patch("libs.llm.openai_llm.OpenAI")
    def test_create_deepseek_from_factory(self, mock_openai):
        """测试通过工厂创建 DeepSeek LLM。"""
        from core import LLMSettings
        
        settings = LLMSettings(
            provider="deepseek",
            model="deepseek-chat",
            api_key="test-key"
        )
        
        llm = LLMFactory.create_from_llm_settings(settings)
        
        assert isinstance(llm, DeepSeekLLM)
        assert llm.model == "deepseek-chat"
