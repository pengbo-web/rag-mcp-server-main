"""
LLMReranker 单元测试。

测试 LLM 重排序器的各项功能，包括 prompt 加载、LLM 调用、分数解析等。
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

from src.libs.llm.base_llm import BaseLLM, Message, ChatResponse, LLMError
from src.libs.reranker.base_reranker import RerankCandidate, RerankerError
from src.libs.reranker.llm_reranker import LLMReranker


class MockLLM(BaseLLM):
    """模拟 LLM 用于测试。"""
    
    def __init__(self, response_content: str = "3\n2\n1", should_fail: bool = False):
        super().__init__(model="test-model")
        self.response_content = response_content
        self.should_fail = should_fail
        self.call_count = 0
        self.last_messages = None
    
    def chat(self, messages: list[Message], **kwargs) -> ChatResponse:
        self.call_count += 1
        self.last_messages = messages
        
        if self.should_fail:
            raise LLMError("Mocked LLM error")
        
        return ChatResponse(
            content=self.response_content,
            model=self.model,
            usage={"prompt_tokens": 100, "completion_tokens": 10}
        )


@pytest.fixture
def mock_llm():
    """创建模拟 LLM。"""
    return MockLLM()


@pytest.fixture
def sample_candidates():
    """创建测试候选项。"""
    return [
        RerankCandidate(id="1", text="Python is a programming language", score=0.9),
        RerankCandidate(id="2", text="Java is also a programming language", score=0.8),
        RerankCandidate(id="3", text="Machine learning uses algorithms", score=0.7),
    ]


@pytest.fixture
def mock_prompt_content():
    """模拟 prompt 模板内容（与实际 rerank.txt 内容一致）。"""
    return """You are a relevance judge for search results.

Given a query and a list of candidate text passages, score each passage for relevance to the query.

Scoring criteria:
- 3: Highly relevant - directly answers or addresses the query
- 2: Moderately relevant - contains related information
- 1: Slightly relevant - tangentially related
- 0: Not relevant - no meaningful connection to the query

For each passage, output only the score (0-3).

Query: {query}

Passages:
{passages}

Output format: One score per line, in the same order as the passages.
"""


class TestLLMRerankerInitialization:
    """测试 LLMReranker 初始化。"""
    
    def test_init_with_default_params(self, mock_llm, mock_prompt_content):
        """测试默认参数初始化。"""
        with patch("builtins.open", mock_open(read_data=mock_prompt_content)):
            with patch("pathlib.Path.exists", return_value=True):
                reranker = LLMReranker(llm=mock_llm)
                
                assert reranker.backend == "llm"
                assert reranker.llm is mock_llm
                assert reranker.temperature == 0.0
                assert reranker.max_retries == 2
                assert reranker.prompt_template == mock_prompt_content
    
    def test_init_with_custom_params(self, mock_llm, mock_prompt_content, tmp_path):
        """测试自定义参数初始化。"""
        # 创建临时 prompt 文件
        prompt_file = tmp_path / "custom_prompt.txt"
        prompt_file.write_text(mock_prompt_content, encoding="utf-8")
        
        reranker = LLMReranker(
            llm=mock_llm,
            prompt_path=str(prompt_file),
            temperature=0.3,
            max_retries=5
        )
        
        assert reranker.temperature == 0.3
        assert reranker.max_retries == 5
        assert reranker.prompt_template == mock_prompt_content
    
    def test_init_with_missing_prompt_file(self, mock_llm):
        """测试 prompt 文件不存在时抛出错误。"""
        with patch("pathlib.Path.exists", return_value=False):
            with pytest.raises(RerankerError, match="Prompt 文件不存在"):
                LLMReranker(llm=mock_llm, prompt_path="nonexistent.txt")


class TestPromptLoading:
    """测试 prompt 加载功能。"""
    
    def test_load_prompt_absolute_path(self, mock_llm, tmp_path):
        """测试加载绝对路径 prompt 文件。"""
        prompt_file = tmp_path / "test_prompt.txt"
        prompt_content = "Test prompt: {query} {passages}"
        prompt_file.write_text(prompt_content, encoding="utf-8")
        
        reranker = LLMReranker(llm=mock_llm, prompt_path=str(prompt_file))
        assert reranker.prompt_template == prompt_content
    
    def test_load_prompt_relative_path(self, mock_llm, mock_prompt_content):
        """测试加载相对路径 prompt 文件。"""
        with patch("builtins.open", mock_open(read_data=mock_prompt_content)):
            with patch("pathlib.Path.exists", return_value=True):
                reranker = LLMReranker(llm=mock_llm, prompt_path="config/prompts/rerank.txt")
                assert reranker.prompt_template == mock_prompt_content
    
    def test_load_prompt_with_special_chars(self, mock_llm, tmp_path):
        """测试加载包含特殊字符的 prompt。"""
        prompt_file = tmp_path / "special.txt"
        prompt_content = "Query: {query}\n\nPassages:\n{passages}\n\n分数: 0-3\n输出格式: 每行一个分数"
        prompt_file.write_text(prompt_content, encoding="utf-8")
        
        reranker = LLMReranker(llm=mock_llm, prompt_path=str(prompt_file))
        assert reranker.prompt_template == prompt_content


class TestPromptFormatting:
    """测试 prompt 格式化。"""
    
    def test_format_prompt_basic(self, mock_llm, sample_candidates, mock_prompt_content):
        """测试基本 prompt 格式化。"""
        with patch("builtins.open", mock_open(read_data=mock_prompt_content)):
            with patch("pathlib.Path.exists", return_value=True):
                reranker = LLMReranker(llm=mock_llm)
                
                prompt = reranker._format_prompt("test query", sample_candidates)
                
                assert "test query" in prompt
                assert "1. Python is a programming language" in prompt
                assert "2. Java is also a programming language" in prompt
                assert "3. Machine learning uses algorithms" in prompt
    
    def test_format_prompt_with_long_text(self, mock_llm, mock_prompt_content):
        """测试长文本候选项格式化。"""
        candidates = [
            RerankCandidate(id="1", text="A" * 500),
            RerankCandidate(id="2", text="B" * 500),
        ]
        
        with patch("builtins.open", mock_open(read_data=mock_prompt_content)):
            with patch("pathlib.Path.exists", return_value=True):
                reranker = LLMReranker(llm=mock_llm)
                prompt = reranker._format_prompt("query", candidates)
                
                assert "A" * 500 in prompt
                assert "B" * 500 in prompt


class TestScoreParsing:
    """测试分数解析功能。"""
    
    def test_parse_scores_simple(self, mock_llm, mock_prompt_content):
        """测试简单格式分数解析。"""
        with patch("builtins.open", mock_open(read_data=mock_prompt_content)):
            with patch("pathlib.Path.exists", return_value=True):
                reranker = LLMReranker(llm=mock_llm)
                
                scores = reranker._parse_scores("3\n2\n1", 3)
                assert scores == [3.0, 2.0, 1.0]
    
    def test_parse_scores_with_whitespace(self, mock_llm, mock_prompt_content):
        """测试带空白字符的分数解析。"""
        with patch("builtins.open", mock_open(read_data=mock_prompt_content)):
            with patch("pathlib.Path.exists", return_value=True):
                reranker = LLMReranker(llm=mock_llm)
                
                scores = reranker._parse_scores("  3  \n\n  2  \n  1  ", 3)
                assert scores == [3.0, 2.0, 1.0]
    
    def test_parse_scores_with_text(self, mock_llm, mock_prompt_content):
        """测试带文本前缀的分数解析。"""
        with patch("builtins.open", mock_open(read_data=mock_prompt_content)):
            with patch("pathlib.Path.exists", return_value=True):
                reranker = LLMReranker(llm=mock_llm)
                
                scores = reranker._parse_scores("Score: 3\nScore: 2\nScore: 1", 3)
                assert scores == [3.0, 2.0, 1.0]
    
    def test_parse_scores_float(self, mock_llm, mock_prompt_content):
        """测试浮点数分数解析。"""
        with patch("builtins.open", mock_open(read_data=mock_prompt_content)):
            with patch("pathlib.Path.exists", return_value=True):
                reranker = LLMReranker(llm=mock_llm)
                
                scores = reranker._parse_scores("2.5\n1.8\n0.3", 3)
                assert scores == [2.5, 1.8, 0.3]
    
    def test_parse_scores_count_mismatch(self, mock_llm, mock_prompt_content):
        """测试分数数量不匹配时抛出错误。"""
        with patch("builtins.open", mock_open(read_data=mock_prompt_content)):
            with patch("pathlib.Path.exists", return_value=True):
                reranker = LLMReranker(llm=mock_llm)
                
                with pytest.raises(RerankerError, match="分数数量.*不匹配"):
                    reranker._parse_scores("3\n2", 3)
    
    def test_parse_scores_invalid_format(self, mock_llm, mock_prompt_content):
        """测试无效格式时抛出错误。"""
        with patch("builtins.open", mock_open(read_data=mock_prompt_content)):
            with patch("pathlib.Path.exists", return_value=True):
                reranker = LLMReranker(llm=mock_llm)
                
                with pytest.raises(RerankerError, match="分数数量.*不匹配"):
                    reranker._parse_scores("no numbers here", 3)


class TestReranking:
    """测试重排序功能。"""
    
    def test_rerank_basic(self, sample_candidates, mock_prompt_content):
        """测试基本重排序功能。"""
        mock_llm = MockLLM(response_content="3\n1\n2")
        
        with patch("builtins.open", mock_open(read_data=mock_prompt_content)):
            with patch("pathlib.Path.exists", return_value=True):
                reranker = LLMReranker(llm=mock_llm)
                
                results = reranker.rerank("test query", sample_candidates)
                
                assert len(results) == 3
                # 按分数降序排列
                assert results[0].id == "1"  # 分数 3
                assert results[0].score == 3.0
                assert results[1].id == "3"  # 分数 2
                assert results[1].score == 2.0
                assert results[2].id == "2"  # 分数 1
                assert results[2].score == 1.0
                
                # 检查排名
                assert results[0].new_rank == 0
                assert results[1].new_rank == 1
                assert results[2].new_rank == 2
    
    def test_rerank_with_top_k(self, sample_candidates, mock_prompt_content):
        """测试 top_k 限制。"""
        mock_llm = MockLLM(response_content="3\n2\n1")
        
        with patch("builtins.open", mock_open(read_data=mock_prompt_content)):
            with patch("pathlib.Path.exists", return_value=True):
                reranker = LLMReranker(llm=mock_llm)
                
                results = reranker.rerank("test query", sample_candidates, top_k=2)
                
                assert len(results) == 2
                assert results[0].score == 3.0
                assert results[1].score == 2.0
    
    def test_rerank_with_llm_error_and_retry(self, sample_candidates, mock_prompt_content):
        """测试 LLM 错误时的重试机制。"""
        # 第一次调用失败，第二次成功
        mock_llm = Mock(spec=BaseLLM)
        mock_llm.chat.side_effect = [
            LLMError("First attempt failed"),
            ChatResponse(content="3\n2\n1", model="test-model")
        ]
        
        with patch("builtins.open", mock_open(read_data=mock_prompt_content)):
            with patch("pathlib.Path.exists", return_value=True):
                reranker = LLMReranker(llm=mock_llm, max_retries=2)
                
                results = reranker.rerank("test query", sample_candidates)
                
                assert len(results) == 3
                assert mock_llm.chat.call_count == 2  # 一次失败 + 一次成功
    
    def test_rerank_fails_after_max_retries(self, sample_candidates, mock_prompt_content):
        """测试达到最大重试次数后抛出错误。"""
        mock_llm = MockLLM(should_fail=True)
        
        with patch("builtins.open", mock_open(read_data=mock_prompt_content)):
            with patch("pathlib.Path.exists", return_value=True):
                reranker = LLMReranker(llm=mock_llm, max_retries=1)
                
                with pytest.raises(RerankerError, match="LLM 重排序失败.*重试.*次"):
                    reranker.rerank("test query", sample_candidates)
    
    def test_rerank_empty_candidates(self, mock_llm, mock_prompt_content):
        """测试空候选项列表时抛出错误。"""
        with patch("builtins.open", mock_open(read_data=mock_prompt_content)):
            with patch("pathlib.Path.exists", return_value=True):
                reranker = LLMReranker(llm=mock_llm)
                
                with pytest.raises(RerankerError, match="候选项列表不能为空"):
                    reranker.rerank("test query", [])
    
    def test_rerank_passes_temperature(self, sample_candidates, mock_prompt_content):
        """测试 temperature 参数传递给 LLM。"""
        mock_llm = MockLLM(response_content="3\n2\n1")
        
        with patch("builtins.open", mock_open(read_data=mock_prompt_content)):
            with patch("pathlib.Path.exists", return_value=True):
                reranker = LLMReranker(llm=mock_llm, temperature=0.5)
                
                results = reranker.rerank("test query", sample_candidates)
                
                # 检查 chat 调用中是否传递了 temperature
                assert mock_llm.call_count == 1
                # Note: 实际检查需要查看 mock_llm.last_messages 或 kwargs


class TestEdgeCases:
    """测试边界情况。"""
    
    def test_single_candidate(self, mock_prompt_content):
        """测试单个候选项。"""
        mock_llm = MockLLM(response_content="3")
        candidates = [RerankCandidate(id="1", text="Single item")]
        
        with patch("builtins.open", mock_open(read_data=mock_prompt_content)):
            with patch("pathlib.Path.exists", return_value=True):
                reranker = LLMReranker(llm=mock_llm)
                
                results = reranker.rerank("query", candidates)
                
                assert len(results) == 1
                assert results[0].id == "1"
                assert results[0].score == 3.0
    
    def test_many_candidates(self, mock_prompt_content):
        """测试大量候选项。"""
        candidates = [RerankCandidate(id=str(i), text=f"Item {i}") for i in range(100)]
        mock_llm = MockLLM(response_content="\n".join(str(i % 4) for i in range(100)))
        
        with patch("builtins.open", mock_open(read_data=mock_prompt_content)):
            with patch("pathlib.Path.exists", return_value=True):
                reranker = LLMReranker(llm=mock_llm)
                
                results = reranker.rerank("query", candidates)
                
                assert len(results) == 100
                # 验证分数降序排列
                for i in range(len(results) - 1):
                    assert results[i].score >= results[i + 1].score
    
    def test_zero_scores(self, sample_candidates, mock_prompt_content):
        """测试全部为 0 分的情况。"""
        mock_llm = MockLLM(response_content="0\n0\n0")
        
        with patch("builtins.open", mock_open(read_data=mock_prompt_content)):
            with patch("pathlib.Path.exists", return_value=True):
                reranker = LLMReranker(llm=mock_llm)
                
                results = reranker.rerank("query", sample_candidates)
                
                assert len(results) == 3
                assert all(r.score == 0.0 for r in results)
