"""
Ollama LLM 单元测试模块。

此模块测试 Ollama LLM 提供商的核心功能，包括：
- 基本初始化
- 聊天补全
- 错误处理（连接失败、超时、模型不存在）
- 配置管理
- 工厂集成
"""

import os
from unittest.mock import Mock, patch, MagicMock
import pytest

from src.libs.llm.ollama_llm import OllamaLLM
from src.libs.llm.base_llm import Message, LLMError
from src.libs.llm.llm_factory import LLMFactory


class TestOllamaLLMInitialization:
    """测试 Ollama LLM 初始化行为。"""
    
    @patch("src.libs.llm.ollama_llm.OpenAI")
    def test_initialization_with_defaults(self, mock_openai_cls):
        """测试使用默认参数初始化。"""
        mock_client = Mock()
        mock_openai_cls.return_value = mock_client
        
        llm = OllamaLLM(model="llama2")
        
        assert llm.model == "llama2"
        assert llm.base_url == "http://localhost:11434/v1"
        assert llm.api_key == "ollama"
        assert llm.temperature == 0.7
        assert llm.max_tokens == 2048
        assert llm.timeout == 30
        assert llm.client == mock_client
        
        # 验证 OpenAI 客户端初始化参数
        mock_openai_cls.assert_called_once_with(
            api_key="ollama",
            base_url="http://localhost:11434/v1",
            timeout=30,
        )
    
    @patch("src.libs.llm.ollama_llm.OpenAI")
    def test_initialization_with_custom_base_url(self, mock_openai_cls):
        """测试使用自定义服务地址初始化。"""
        mock_client = Mock()
        mock_openai_cls.return_value = mock_client
        
        custom_url = "http://192.168.1.100:11434/v1"
        llm = OllamaLLM(model="mistral", base_url=custom_url)
        
        assert llm.base_url == custom_url
        mock_openai_cls.assert_called_once_with(
            api_key="ollama",
            base_url=custom_url,
            timeout=30,
        )
    
    @patch.dict(os.environ, {"OLLAMA_BASE_URL": "http://ollama-server:11434/v1"})
    @patch("src.libs.llm.ollama_llm.OpenAI")
    def test_initialization_with_env_base_url(self, mock_openai_cls):
        """测试从环境变量读取服务地址。"""
        mock_client = Mock()
        mock_openai_cls.return_value = mock_client
        
        llm = OllamaLLM(model="qwen")
        
        assert llm.base_url == "http://ollama-server:11434/v1"
    
    @patch("src.libs.llm.ollama_llm.OpenAI")
    def test_initialization_with_custom_parameters(self, mock_openai_cls):
        """测试使用自定义参数初始化。"""
        mock_client = Mock()
        mock_openai_cls.return_value = mock_client
        
        llm = OllamaLLM(
            model="llama2",
            base_url="http://custom:11434/v1",
            temperature=0.5,
            max_tokens=1024,
            timeout=60
        )
        
        assert llm.temperature == 0.5
        assert llm.max_tokens == 1024
        assert llm.timeout == 60
        
        mock_openai_cls.assert_called_once_with(
            api_key="ollama",
            base_url="http://custom:11434/v1",
            timeout=60,
        )
    
    @patch("src.libs.llm.ollama_llm.OpenAI")
    def test_initialization_failure(self, mock_openai_cls):
        """测试初始化失败的错误处理。"""
        mock_openai_cls.side_effect = Exception("Connection refused")
        
        with pytest.raises(LLMError) as exc_info:
            OllamaLLM(model="llama2")
        
        error_msg = str(exc_info.value)
        assert "Failed to initialize Ollama client" in error_msg
        assert "Please ensure Ollama is running" in error_msg
        assert "Connection refused" in error_msg


class TestOllamaLLMChat:
    """测试 Ollama LLM 聊天功能。"""
    
    @patch("src.libs.llm.ollama_llm.OpenAI")
    def test_chat_success(self, mock_openai_cls):
        """测试成功的聊天调用。"""
        # Mock OpenAI 客户端和响应
        mock_client = Mock()
        mock_openai_cls.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Hello! How can I help you?"
        mock_response.model = "llama2"
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30
        
        mock_client.chat.completions.create.return_value = mock_response
        
        # 执行测试
        llm = OllamaLLM(model="llama2")
        messages = [Message(role="user", content="Hello")]
        response = llm.chat(messages)
        
        # 验证响应
        assert response.content == "Hello! How can I help you?"
        assert response.model == "llama2"
        assert response.usage["prompt_tokens"] == 10
        assert response.usage["completion_tokens"] == 20
        assert response.usage["total_tokens"] == 30
        
        # 验证 API 调用
        mock_client.chat.completions.create.assert_called_once_with(
            model="llama2",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.7,
            max_tokens=2048,
        )
    
    @patch("src.libs.llm.ollama_llm.OpenAI")
    def test_chat_with_multiple_messages(self, mock_openai_cls):
        """测试多轮对话。"""
        mock_client = Mock()
        mock_openai_cls.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Python is great!"
        mock_response.model = "llama2"
        mock_response.usage = None  # Ollama 可能不返回 usage
        
        mock_client.chat.completions.create.return_value = mock_response
        
        llm = OllamaLLM(model="llama2")
        messages = [
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="What do you think about Python?"),
        ]
        response = llm.chat(messages)
        
        assert response.content == "Python is great!"
        assert response.usage is None
        
        # 验证发送的消息格式
        call_args = mock_client.chat.completions.create.call_args
        assert call_args[1]["messages"] == [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What do you think about Python?"},
        ]
    
    @patch("src.libs.llm.ollama_llm.OpenAI")
    def test_chat_with_custom_parameters(self, mock_openai_cls):
        """测试使用自定义参数的聊天调用。"""
        mock_client = Mock()
        mock_openai_cls.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.model = "llama2"
        mock_response.usage = None
        
        mock_client.chat.completions.create.return_value = mock_response
        
        llm = OllamaLLM(model="llama2", temperature=0.5, max_tokens=512)
        messages = [Message(role="user", content="Test")]
        response = llm.chat(messages, temperature=0.3, max_tokens=256)
        
        # 验证参数覆盖
        call_args = mock_client.chat.completions.create.call_args
        assert call_args[1]["temperature"] == 0.3
        assert call_args[1]["max_tokens"] == 256
    
    @patch("src.libs.llm.ollama_llm.OpenAI")
    def test_chat_with_empty_response(self, mock_openai_cls):
        """测试空响应内容的处理。"""
        mock_client = Mock()
        mock_openai_cls.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = None  # 空响应
        mock_response.model = "llama2"
        mock_response.usage = None
        
        mock_client.chat.completions.create.return_value = mock_response
        
        llm = OllamaLLM(model="llama2")
        messages = [Message(role="user", content="Test")]
        response = llm.chat(messages)
        
        # 空响应应该转换为空字符串
        assert response.content == ""


class TestOllamaLLMErrorHandling:
    """测试 Ollama LLM 错误处理。"""
    
    @patch("src.libs.llm.ollama_llm.OpenAI")
    def test_chat_with_empty_messages(self, mock_openai_cls):
        """测试空消息列表的错误处理。"""
        mock_client = Mock()
        mock_openai_cls.return_value = mock_client
        
        llm = OllamaLLM(model="llama2")
        
        with pytest.raises(LLMError) as exc_info:
            llm.chat([])
        
        assert "Messages list cannot be empty" in str(exc_info.value)
    
    @patch("src.libs.llm.ollama_llm.OpenAI")
    def test_chat_with_invalid_message_type(self, mock_openai_cls):
        """测试无效消息类型的错误处理。"""
        mock_client = Mock()
        mock_openai_cls.return_value = mock_client
        
        llm = OllamaLLM(model="llama2")
        
        with pytest.raises(LLMError) as exc_info:
            llm.chat([{"role": "user", "content": "test"}])  # 应该是 Message 对象
        
        error_msg = str(exc_info.value)
        assert "must be a Message object" in error_msg
    
    @patch("src.libs.llm.ollama_llm.OpenAI")
    def test_chat_with_invalid_role(self, mock_openai_cls):
        """测试无效角色的错误处理。"""
        mock_client = Mock()
        mock_openai_cls.return_value = mock_client
        
        llm = OllamaLLM(model="llama2")
        
        with pytest.raises(LLMError) as exc_info:
            llm.chat([Message(role="invalid", content="test")])
        
        error_msg = str(exc_info.value)
        assert "invalid role" in error_msg
        assert "Must be 'system', 'user', or 'assistant'" in error_msg
    
    @patch("src.libs.llm.ollama_llm.OpenAI")
    def test_chat_connection_error(self, mock_openai_cls):
        """测试连接失败的错误处理。"""
        from openai import APIConnectionError
        
        mock_client = Mock()
        mock_openai_cls.return_value = mock_client
        
        # Mock 连接错误
        mock_request = Mock()
        mock_client.chat.completions.create.side_effect = APIConnectionError(
            request=mock_request,
        )
        
        llm = OllamaLLM(model="llama2")
        messages = [Message(role="user", content="test")]
        
        with pytest.raises(LLMError) as exc_info:
            llm.chat(messages)
        
        error_msg = str(exc_info.value)
        assert "Failed to connect to Ollama service" in error_msg
        assert "http://localhost:11434/v1" in error_msg
        assert "Please ensure Ollama is running" in error_msg
        assert "ollama serve" in error_msg
    
    @patch("src.libs.llm.ollama_llm.OpenAI")
    def test_chat_timeout_error(self, mock_openai_cls):
        """测试超时的错误处理。"""
        from openai import APITimeoutError
        
        mock_client = Mock()
        mock_openai_cls.return_value = mock_client
        
        # Mock 超时错误
        mock_request = Mock()
        mock_client.chat.completions.create.side_effect = APITimeoutError(
            request=mock_request,
        )
        
        llm = OllamaLLM(model="llama2", timeout=30)
        messages = [Message(role="user", content="test")]
        
        with pytest.raises(LLMError) as exc_info:
            llm.chat(messages)
        
        error_msg = str(exc_info.value)
        assert "timed out" in error_msg
        assert "30s" in error_msg
        assert "llama2" in error_msg
    
    @patch("src.libs.llm.ollama_llm.OpenAI")
    def test_chat_model_not_found_error(self, mock_openai_cls):
        """测试模型不存在的错误处理。"""
        from openai import APIError
        
        mock_client = Mock()
        mock_openai_cls.return_value = mock_client
        
        # Mock 模型不存在错误
        mock_response = Mock()
        mock_response.status_code = 404
        mock_client.chat.completions.create.side_effect = APIError(
            "model not found",
            body={"error": "model not found"},
            request=Mock()
        )
        
        llm = OllamaLLM(model="nonexistent")
        messages = [Message(role="user", content="test")]
        
        with pytest.raises(LLMError) as exc_info:
            llm.chat(messages)
        
        error_msg = str(exc_info.value)
        assert "Model 'nonexistent' not found" in error_msg
        assert "ollama pull nonexistent" in error_msg
    
    @patch("src.libs.llm.ollama_llm.OpenAI")
    def test_chat_api_error(self, mock_openai_cls):
        """测试一般 API 错误的处理。"""
        from openai import APIError
        
        mock_client = Mock()
        mock_openai_cls.return_value = mock_client
        
        # Mock 一般 API 错误
        mock_response = Mock()
        mock_response.status_code = 500
        mock_client.chat.completions.create.side_effect = APIError(
            "Internal server error",
            body={"error": "internal error"},
            request=Mock()
        )
        
        llm = OllamaLLM(model="llama2")
        messages = [Message(role="user", content="test")]
        
        with pytest.raises(LLMError) as exc_info:
            llm.chat(messages)
        
        error_msg = str(exc_info.value)
        assert "Ollama API error" in error_msg
    
    @patch("src.libs.llm.ollama_llm.OpenAI")
    def test_chat_unexpected_error(self, mock_openai_cls):
        """测试意外错误的处理。"""
        mock_client = Mock()
        mock_openai_cls.return_value = mock_client
        
        # Mock 意外错误
        mock_client.chat.completions.create.side_effect = ValueError("Unexpected issue")
        
        llm = OllamaLLM(model="llama2")
        messages = [Message(role="user", content="test")]
        
        with pytest.raises(LLMError) as exc_info:
            llm.chat(messages)
        
        error_msg = str(exc_info.value)
        assert "Unexpected error" in error_msg
        assert "Unexpected issue" in error_msg


class TestFactoryIntegration:
    """测试 Ollama LLM 与工厂的集成。"""
    
    def setup_method(self):
        """每个测试方法前注册 Ollama 提供商。"""
        # 确保 Ollama 已注册
        if "ollama" not in LLMFactory._registry:
            from src.libs.llm import ollama_llm
            LLMFactory.register("ollama", ollama_llm.OllamaLLM)
    
    @patch("src.libs.llm.ollama_llm.OpenAI")
    def test_factory_creates_ollama_llm(self, mock_openai_cls):
        """测试工厂能正确创建 Ollama LLM 实例。"""
        from core import LLMSettings
        
        mock_client = Mock()
        mock_openai_cls.return_value = mock_client
        
        llm_settings = LLMSettings(
            provider="ollama",
            model="llama2",
            base_url="http://localhost:11434/v1",
            temperature=0.7,
            max_tokens=2048,
            timeout=30
        )
        
        llm = LLMFactory.create_from_llm_settings(llm_settings)
        
        assert isinstance(llm, OllamaLLM)
        assert llm.model == "llama2"
        assert llm.base_url == "http://localhost:11434/v1"
    
    @patch("src.libs.llm.ollama_llm.OpenAI")
    def test_factory_creates_with_config_dict(self, mock_openai_cls):
        """测试工厂使用配置字典创建实例。"""
        from core import LLMSettings
        
        mock_client = Mock()
        mock_openai_cls.return_value = mock_client
        
        llm_settings = LLMSettings(
            provider="ollama",
            model="mistral",
            base_url="http://192.168.1.100:11434/v1",
            temperature=0.5,
            max_tokens=1024,
            timeout=30
        )
        
        llm = LLMFactory.create_from_llm_settings(llm_settings)
        
        assert isinstance(llm, OllamaLLM)
        assert llm.model == "mistral"
        assert llm.base_url == "http://192.168.1.100:11434/v1"
        assert llm.temperature == 0.5
        assert llm.max_tokens == 1024
