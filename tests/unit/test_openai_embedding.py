"""
OpenAI Embedding 单元测试模块。

此模块测试 OpenAI Embedding 提供商的核心功能，包括：
- 基本初始化
- 嵌入生成
- 批量处理
- 错误处理（认证失败、速率限制、输入过长）
- 配置管理
- 工厂集成
"""

import os
from unittest.mock import Mock, patch
import pytest

from src.libs.embedding.openai_embedding import OpenAIEmbedding
from src.libs.embedding.base_embedding import EmbeddingError
from src.libs.embedding.embedding_factory import EmbeddingFactory


class TestOpenAIEmbeddingInitialization:
    """测试 OpenAI Embedding 初始化行为。"""
    
    @patch("src.libs.embedding.openai_embedding.OpenAI")
    def test_initialization_with_api_key(self, mock_openai_cls):
        """测试使用 API 密钥初始化。"""
        mock_client = Mock()
        mock_openai_cls.return_value = mock_client
        
        embedding = OpenAIEmbedding(
            model="text-embedding-3-small",
            api_key="test-key"
        )
        
        assert embedding.model == "text-embedding-3-small"
        assert embedding.api_key == "test-key"
        assert embedding.dimensions == 1536
        assert embedding.batch_size == 100
        assert embedding.client == mock_client
        
        # 验证 OpenAI 客户端初始化
        mock_openai_cls.assert_called_once_with(
            api_key="test-key",
            timeout=30,
        )
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"})
    @patch("src.libs.embedding.openai_embedding.OpenAI")
    def test_initialization_from_env(self, mock_openai_cls):
        """测试从环境变量读取 API 密钥。"""
        mock_client = Mock()
        mock_openai_cls.return_value = mock_client
        
        embedding = OpenAIEmbedding(model="text-embedding-3-small")
        
        assert embedding.api_key == "env-key"
    
    @patch.dict(os.environ, {}, clear=True)
    @patch("src.libs.embedding.openai_embedding.OpenAI")
    def test_initialization_without_api_key_raises_error(self, mock_openai_cls):
        """测试缺少 API 密钥时抛出错误。"""
        with pytest.raises(EmbeddingError, match="API key is required"):
            OpenAIEmbedding(model="text-embedding-3-small")
    
    @patch("src.libs.embedding.openai_embedding.OpenAI")
    def test_initialization_with_custom_dimensions(self, mock_openai_cls):
        """测试自定义向量维度。"""
        mock_client = Mock()
        mock_openai_cls.return_value = mock_client
        
        embedding = OpenAIEmbedding(
            model="text-embedding-3-large",
            api_key="test-key",
            dimensions=1024
        )
        
        assert embedding.dimensions == 1024
    
    @patch("src.libs.embedding.openai_embedding.OpenAI")
    def test_initialization_with_default_dimensions(self, mock_openai_cls):
        """测试模型默认维度。"""
        mock_client = Mock()
        mock_openai_cls.return_value = mock_client
        
        # text-embedding-3-small 默认 1536 维
        emb1 = OpenAIEmbedding(model="text-embedding-3-small", api_key="test-key")
        assert emb1.dimensions == 1536
        
        # text-embedding-3-large 默认 3072 维
        emb2 = OpenAIEmbedding(model="text-embedding-3-large", api_key="test-key")
        assert emb2.dimensions == 3072
        
        # text-embedding-ada-002 默认 1536 维
        emb3 = OpenAIEmbedding(model="text-embedding-ada-002", api_key="test-key")
        assert emb3.dimensions == 1536
    
    @patch("src.libs.embedding.openai_embedding.OpenAI")
    def test_initialization_with_custom_base_url(self, mock_openai_cls):
        """测试自定义 API 端点。"""
        mock_client = Mock()
        mock_openai_cls.return_value = mock_client
        
        custom_url = "https://custom.api.com/v1"
        embedding = OpenAIEmbedding(
            model="text-embedding-3-small",
            api_key="test-key",
            base_url=custom_url
        )
        
        assert embedding.base_url == custom_url
        mock_openai_cls.assert_called_once_with(
            api_key="test-key",
            base_url=custom_url,
            timeout=30,
        )
    
    @patch("src.libs.embedding.openai_embedding.OpenAI")
    def test_initialization_failure(self, mock_openai_cls):
        """测试初始化失败的错误处理。"""
        mock_openai_cls.side_effect = Exception("Connection refused")
        
        with pytest.raises(EmbeddingError, match="Failed to initialize"):
            OpenAIEmbedding(model="text-embedding-3-small", api_key="test-key")


class TestOpenAIEmbeddingEmbed:
    """测试 OpenAI Embedding 嵌入功能。"""
    
    @patch("src.libs.embedding.openai_embedding.OpenAI")
    def test_embed_success(self, mock_openai_cls):
        """测试成功的嵌入调用。"""
        # Mock OpenAI 客户端和响应
        mock_client = Mock()
        mock_openai_cls.return_value = mock_client
        
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1, 0.2, 0.3]),
            Mock(embedding=[0.4, 0.5, 0.6]),
        ]
        mock_response.model = "text-embedding-3-small"
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.total_tokens = 10
        
        mock_client.embeddings.create.return_value = mock_response
        
        # 执行测试
        embedding = OpenAIEmbedding(model="text-embedding-3-small", api_key="test-key")
        texts = ["hello world", "test text"]
        response = embedding.embed(texts)
        
        # 验证响应
        assert response.embeddings == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        assert response.model == "text-embedding-3-small"
        assert response.dimensions == 3
        assert response.usage["prompt_tokens"] == 10
        assert response.usage["total_tokens"] == 10
        
        # 验证 API 调用
        mock_client.embeddings.create.assert_called_once_with(
            input=texts,
            model="text-embedding-3-small",
            dimensions=1536,
        )
    
    @patch("src.libs.embedding.openai_embedding.OpenAI")
    def test_embed_single(self, mock_openai_cls):
        """测试单个文本嵌入。"""
        mock_client = Mock()
        mock_openai_cls.return_value = mock_client
        
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_response.model = "text-embedding-3-small"
        mock_response.usage = None
        
        mock_client.embeddings.create.return_value = mock_response
        
        embedding = OpenAIEmbedding(model="text-embedding-3-small", api_key="test-key")
        vector = embedding.embed_single("hello")
        
        assert vector == [0.1, 0.2, 0.3]
    
    @patch("src.libs.embedding.openai_embedding.OpenAI")
    def test_embed_with_custom_dimensions(self, mock_openai_cls):
        """测试使用自定义维度。"""
        mock_client = Mock()
        mock_openai_cls.return_value = mock_client
        
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 512)]
        mock_response.model = "text-embedding-3-small"
        mock_response.usage = None
        
        mock_client.embeddings.create.return_value = mock_response
        
        embedding = OpenAIEmbedding(
            model="text-embedding-3-small",
            api_key="test-key",
            dimensions=512
        )
        response = embedding.embed(["test"], dimensions=512)
        
        # 验证使用了自定义维度
        call_args = mock_client.embeddings.create.call_args
        assert call_args[1]["dimensions"] == 512
    
    @patch("src.libs.embedding.openai_embedding.OpenAI")
    def test_embed_ada_model_no_dimensions_param(self, mock_openai_cls):
        """测试 ada-002 模型不传递 dimensions 参数。"""
        mock_client = Mock()
        mock_openai_cls.return_value = mock_client
        
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 1536)]
        mock_response.model = "text-embedding-ada-002"
        mock_response.usage = None
        
        mock_client.embeddings.create.return_value = mock_response
        
        embedding = OpenAIEmbedding(
            model="text-embedding-ada-002",
            api_key="test-key"
        )
        embedding.embed(["test"])
        
        # 验证 ada-002 模型不传递 dimensions 参数
        call_args = mock_client.embeddings.create.call_args
        assert "dimensions" not in call_args[1]


class TestOpenAIEmbeddingErrorHandling:
    """测试 OpenAI Embedding 错误处理。"""
    
    @patch("src.libs.embedding.openai_embedding.OpenAI")
    def test_embed_empty_list_raises_error(self, mock_openai_cls):
        """测试空文本列表的错误处理。"""
        mock_client = Mock()
        mock_openai_cls.return_value = mock_client
        
        embedding = OpenAIEmbedding(model="text-embedding-3-small", api_key="test-key")
        
        with pytest.raises(EmbeddingError, match="cannot be empty"):
            embedding.embed([])
    
    @patch("src.libs.embedding.openai_embedding.OpenAI")
    def test_embed_non_list_raises_error(self, mock_openai_cls):
        """测试非列表输入的错误处理。"""
        mock_client = Mock()
        mock_openai_cls.return_value = mock_client
        
        embedding = OpenAIEmbedding(model="text-embedding-3-small", api_key="test-key")
        
        with pytest.raises(EmbeddingError, match="must be a list"):
            embedding.embed("not a list")
    
    @patch("src.libs.embedding.openai_embedding.OpenAI")
    def test_embed_non_string_item_raises_error(self, mock_openai_cls):
        """测试非字符串元素的错误处理。"""
        mock_client = Mock()
        mock_openai_cls.return_value = mock_client
        
        embedding = OpenAIEmbedding(model="text-embedding-3-small", api_key="test-key")
        
        with pytest.raises(EmbeddingError, match="must be a string"):
            embedding.embed(["hello", 123])
    
    @patch("src.libs.embedding.openai_embedding.OpenAI")
    def test_embed_empty_string_raises_error(self, mock_openai_cls):
        """测试空字符串的错误处理。"""
        mock_client = Mock()
        mock_openai_cls.return_value = mock_client
        
        embedding = OpenAIEmbedding(model="text-embedding-3-small", api_key="test-key")
        
        with pytest.raises(EmbeddingError, match="empty or whitespace-only"):
            embedding.embed(["hello", "  ", "world"])
    
    @patch("src.libs.embedding.openai_embedding.OpenAI")
    def test_embed_exceeds_batch_size_raises_error(self, mock_openai_cls):
        """测试超过批量大小限制。"""
        mock_client = Mock()
        mock_openai_cls.return_value = mock_client
        
        embedding = OpenAIEmbedding(model="text-embedding-3-small", api_key="test-key")
        
        # 创建超过限制的批量
        large_batch = ["text"] * 2500
        
        with pytest.raises(EmbeddingError, match="exceeds maximum"):
            embedding.embed(large_batch)
    
    @patch("src.libs.embedding.openai_embedding.OpenAI")
    def test_embed_authentication_error(self, mock_openai_cls):
        """测试认证失败场景。"""
        from openai import AuthenticationError
        
        mock_client = Mock()
        mock_openai_cls.return_value = mock_client
        
        # Mock 认证错误
        mock_response = Mock()
        mock_response.status_code = 401
        mock_client.embeddings.create.side_effect = AuthenticationError(
            "Invalid API key",
            response=mock_response,
            body={"error": {"message": "Invalid API key"}}
        )
        
        embedding = OpenAIEmbedding(model="text-embedding-3-small", api_key="invalid-key")
        
        with pytest.raises(EmbeddingError, match="authentication failed"):
            embedding.embed(["test"])
    
    @patch("src.libs.embedding.openai_embedding.OpenAI")
    def test_embed_rate_limit_error(self, mock_openai_cls):
        """测试速率限制场景。"""
        from openai import RateLimitError
        
        mock_client = Mock()
        mock_openai_cls.return_value = mock_client
        
        # Mock 速率限制错误
        mock_response = Mock()
        mock_response.status_code = 429
        mock_client.embeddings.create.side_effect = RateLimitError(
            "Rate limit exceeded",
            response=mock_response,
            body={"error": {"message": "Rate limit exceeded"}}
        )
        
        embedding = OpenAIEmbedding(model="text-embedding-3-small", api_key="test-key")
        
        with pytest.raises(EmbeddingError, match="rate limit exceeded"):
            embedding.embed(["test"])
    
    @patch("src.libs.embedding.openai_embedding.OpenAI")
    def test_embed_connection_error(self, mock_openai_cls):
        """测试连接失败的错误处理。"""
        from openai import APIConnectionError
        
        mock_client = Mock()
        mock_openai_cls.return_value = mock_client
        
        # Mock 连接错误
        mock_request = Mock()
        mock_client.embeddings.create.side_effect = APIConnectionError(
            request=mock_request,
        )
        
        embedding = OpenAIEmbedding(model="text-embedding-3-small", api_key="test-key")
        
        with pytest.raises(EmbeddingError, match="Failed to connect"):
            embedding.embed(["test"])
    
    @patch("src.libs.embedding.openai_embedding.OpenAI")
    def test_embed_input_too_long_error(self, mock_openai_cls):
        """测试输入过长的错误处理。"""
        from openai import APIError
        
        mock_client = Mock()
        mock_openai_cls.return_value = mock_client
        
        # Mock 输入过长错误
        mock_response = Mock()
        mock_response.status_code = 400
        mock_client.embeddings.create.side_effect = APIError(
            "maximum context length exceeded",
            body={"error": "maximum context length exceeded"},
            request=Mock()
        )
        
        embedding = OpenAIEmbedding(model="text-embedding-3-small", api_key="test-key")
        
        with pytest.raises(EmbeddingError, match="maximum context length"):
            embedding.embed(["very long text" * 1000])
    
    @patch("src.libs.embedding.openai_embedding.OpenAI")
    def test_embed_api_error(self, mock_openai_cls):
        """测试一般 API 错误的处理。"""
        from openai import APIError
        
        mock_client = Mock()
        mock_openai_cls.return_value = mock_client
        
        # Mock 一般 API 错误
        mock_response = Mock()
        mock_response.status_code = 500
        mock_client.embeddings.create.side_effect = APIError(
            "Internal server error",
            body={"error": "internal error"},
            request=Mock()
        )
        
        embedding = OpenAIEmbedding(model="text-embedding-3-small", api_key="test-key")
        
        with pytest.raises(EmbeddingError, match="API error"):
            embedding.embed(["test"])
    
    @patch("src.libs.embedding.openai_embedding.OpenAI")
    def test_embed_unexpected_error(self, mock_openai_cls):
        """测试意外错误的处理。"""
        mock_client = Mock()
        mock_openai_cls.return_value = mock_client
        
        # Mock 意外错误
        mock_client.embeddings.create.side_effect = ValueError("Unexpected issue")
        
        embedding = OpenAIEmbedding(model="text-embedding-3-small", api_key="test-key")
        
        with pytest.raises(EmbeddingError, match="Unexpected error"):
            embedding.embed(["test"])


class TestFactoryIntegration:
    """测试 OpenAI Embedding 与工厂的集成。"""
    
    def setup_method(self):
        """每个测试方法前注册 OpenAI 提供商。"""
        # 确保 OpenAI 已注册
        if "openai" not in EmbeddingFactory._registry:
            from src.libs.embedding import openai_embedding
            EmbeddingFactory.register("openai", openai_embedding.OpenAIEmbedding)
    
    @patch("src.libs.embedding.openai_embedding.OpenAI")
    def test_factory_creates_openai_embedding(self, mock_openai_cls):
        """测试工厂能正确创建 OpenAI Embedding 实例。"""
        from core import EmbeddingSettings
        
        mock_client = Mock()
        mock_openai_cls.return_value = mock_client
        
        settings = EmbeddingSettings(
            provider="openai",
            model="text-embedding-3-small",
            api_key="test-key",
            dimensions=1536,
            batch_size=100
        )
        
        embedding = EmbeddingFactory.create_from_embedding_settings(settings)
        
        assert isinstance(embedding, OpenAIEmbedding)
        assert embedding.model == "text-embedding-3-small"
        assert embedding.dimensions == 1536
    
    @patch("src.libs.embedding.openai_embedding.OpenAI")
    def test_factory_creates_with_custom_config(self, mock_openai_cls):
        """测试工厂使用自定义配置创建实例。"""
        from core import EmbeddingSettings
        
        mock_client = Mock()
        mock_openai_cls.return_value = mock_client
        
        settings = EmbeddingSettings(
            provider="openai",
            model="text-embedding-3-large",
            api_key="test-key",
            dimensions=3072,
            batch_size=50
        )
        
        embedding = EmbeddingFactory.create_from_embedding_settings(settings)
        
        assert isinstance(embedding, OpenAIEmbedding)
        assert embedding.model == "text-embedding-3-large"
        assert embedding.dimensions == 3072
        assert embedding.batch_size == 50
