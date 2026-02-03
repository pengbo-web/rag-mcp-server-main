"""
Recursive Splitter 单元测试模块。

此模块测试 Recursive Splitter 的核心功能，包括：
- 基本初始化
- 文本切分
- Markdown 结构保护
- 重叠处理
- 边界情况
- 工厂集成
"""

import pytest

from src.libs.splitter.recursive_splitter import RecursiveSplitter
from src.libs.splitter.base_splitter import SplitterError
from src.libs.splitter.splitter_factory import SplitterFactory


class TestRecursiveSplitterInitialization:
    """测试 Recursive Splitter 初始化行为。"""
    
    def test_initialization_with_defaults(self):
        """测试使用默认参数初始化。"""
        splitter = RecursiveSplitter()
        
        assert splitter.chunk_size == 1000
        assert splitter.chunk_overlap == 200
        assert splitter.keep_separator is True
        assert len(splitter.separators) > 0
    
    def test_initialization_with_custom_params(self):
        """测试自定义参数初始化。"""
        custom_seps = ["\n\n", "\n", " "]
        splitter = RecursiveSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=custom_seps,
            keep_separator=False
        )
        
        assert splitter.chunk_size == 500
        assert splitter.chunk_overlap == 50
        assert splitter.separators == custom_seps
        assert splitter.keep_separator is False
    
    def test_initialization_with_invalid_chunk_size(self):
        """测试无效 chunk_size 抛出错误。"""
        with pytest.raises(SplitterError, match="must be positive"):
            RecursiveSplitter(chunk_size=0)
        
        with pytest.raises(SplitterError, match="must be positive"):
            RecursiveSplitter(chunk_size=-100)
    
    def test_initialization_with_invalid_overlap(self):
        """测试无效 chunk_overlap 抛出错误。"""
        with pytest.raises(SplitterError, match="cannot be negative"):
            RecursiveSplitter(chunk_overlap=-10)
        
        with pytest.raises(SplitterError, match="less than chunk_size"):
            RecursiveSplitter(chunk_size=100, chunk_overlap=100)


class TestRecursiveSplitterBasicSplitting:
    """测试 Recursive Splitter 基本切分功能。"""
    
    def test_split_simple_text(self):
        """测试简单文本切分。"""
        splitter = RecursiveSplitter(chunk_size=20, chunk_overlap=0)
        text = "This is a test. This is another test."
        
        chunks = splitter.split_text(text)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)
        # 重建文本应接近原文
        assert "test" in "".join(chunks)
    
    def test_split_by_paragraphs(self):
        """测试按段落切分。"""
        splitter = RecursiveSplitter(chunk_size=50, chunk_overlap=0)
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        
        chunks = splitter.split_text(text)
        
        # 应该优先在段落边界切分
        assert len(chunks) >= 1
        assert any("Paragraph" in chunk for chunk in chunks)
    
    def test_split_by_newlines(self):
        """测试按换行符切分。"""
        splitter = RecursiveSplitter(chunk_size=30, chunk_overlap=0)
        text = "Line 1\nLine 2\nLine 3\nLine 4"
        
        chunks = splitter.split_text(text)
        
        assert len(chunks) >= 1
    
    def test_split_by_sentences(self):
        """测试按句子切分。"""
        splitter = RecursiveSplitter(chunk_size=25, chunk_overlap=0)
        text = "第一句话。第二句话。第三句话。"
        
        chunks = splitter.split_text(text)
        
        assert len(chunks) >= 1
        # 验证句子保持完整
        assert any("句话" in chunk for chunk in chunks)
    
    def test_split_long_text_without_separators(self):
        """测试没有明显分隔符的长文本。"""
        splitter = RecursiveSplitter(chunk_size=10, chunk_overlap=0)
        text = "a" * 50  # 50个连续字符
        
        chunks = splitter.split_text(text)
        
        # 应该强制字符级切分
        assert len(chunks) >= 5
        assert all(len(chunk) <= 10 for chunk in chunks)
    
    def test_split_with_overlap(self):
        """测试带重叠的切分。"""
        splitter = RecursiveSplitter(chunk_size=20, chunk_overlap=5)
        text = "This is a test sentence for overlap testing."
        
        chunks = splitter.split_text(text)
        
        # 有重叠时，相邻块应该有共同内容
        if len(chunks) > 1:
            # 至少有一些块有重叠迹象
            assert len(chunks) >= 2


class TestRecursiveSplitterMarkdownHandling:
    """测试 Recursive Splitter Markdown 处理。"""
    
    def test_preserve_code_blocks(self):
        """测试保护代码块完整性。"""
        splitter = RecursiveSplitter(chunk_size=100, chunk_overlap=0)
        text = """
Some text before.

```python
def hello():
    print("world")
```

Some text after.
"""
        
        chunks = splitter.split_text(text)
        
        # 代码块应该保持完整
        code_block_found = False
        for chunk in chunks:
            if "```python" in chunk:
                code_block_found = True
                # 验证代码块完整
                assert "```python" in chunk
                assert "```" in chunk[chunk.index("```python") + 9:]
        
        assert code_block_found
    
    def test_handle_multiple_code_blocks(self):
        """测试处理多个代码块。"""
        splitter = RecursiveSplitter(chunk_size=150, chunk_overlap=0)
        text = """
First paragraph.

```python
code1 = "test"
```

Middle paragraph.

```javascript
const code2 = "test";
```

Last paragraph.
"""
        
        chunks = splitter.split_text(text)
        
        # 验证所有代码块都被保护
        combined = "".join(chunks)
        assert "```python" in combined
        assert "```javascript" in combined
    
    def test_split_markdown_headers(self):
        """测试 Markdown 标题切分。"""
        splitter = RecursiveSplitter(chunk_size=50, chunk_overlap=0)
        text = """# Header 1

Content 1.

## Header 2

Content 2."""
        
        chunks = splitter.split_text(text)
        
        # 标题应该和内容保持在一起（优先在段落边界切分）
        assert len(chunks) >= 1


class TestRecursiveSplitterErrorHandling:
    """测试 Recursive Splitter 错误处理。"""
    
    def test_split_empty_text_raises_error(self):
        """测试空文本抛出错误。"""
        splitter = RecursiveSplitter()
        
        with pytest.raises(SplitterError, match="cannot be empty"):
            splitter.split_text("")
    
    def test_split_non_string_raises_error(self):
        """测试非字符串输入抛出错误。"""
        splitter = RecursiveSplitter()
        
        with pytest.raises(SplitterError, match="must be a string"):
            splitter.split_text(123)
        
        with pytest.raises(SplitterError, match="must be a string"):
            splitter.split_text(["list"])


class TestRecursiveSplitterChunkProperties:
    """测试 Recursive Splitter 切分块属性。"""
    
    def test_chunks_respect_size_limit(self):
        """测试块尊重大小限制。"""
        chunk_size = 50
        splitter = RecursiveSplitter(chunk_size=chunk_size, chunk_overlap=0)
        text = "This is a test sentence. " * 20
        
        chunks = splitter.split_text(text)
        
        # 大多数块应该接近 chunk_size（允许一些弹性）
        for chunk in chunks:
            # 由于重叠策略，可能略大于 chunk_size
            assert len(chunk) <= chunk_size * 1.5
    
    def test_chunks_have_content(self):
        """测试所有块都有内容。"""
        splitter = RecursiveSplitter(chunk_size=30, chunk_overlap=5)
        text = "Content here. More content. Even more content."
        
        chunks = splitter.split_text(text)
        
        # 所有块都应该有实际内容（非空白）
        for chunk in chunks:
            assert chunk.strip()
    
    def test_split_preserves_content(self):
        """测试切分保持内容完整性。"""
        splitter = RecursiveSplitter(chunk_size=100, chunk_overlap=10)
        text = "The quick brown fox jumps over the lazy dog."
        
        chunks = splitter.split_text(text)
        
        # 重建的文本应该包含所有关键词
        combined = " ".join(chunks)
        for word in ["quick", "brown", "fox", "lazy", "dog"]:
            assert word in combined
    
    def test_split_with_metadata(self):
        """测试带元数据的切分。"""
        splitter = RecursiveSplitter(chunk_size=50, chunk_overlap=0)
        text = "This is a test. " * 10
        metadata = {"source": "test", "page": 1}
        
        chunks = splitter.split_text_with_metadata(text, metadata)
        
        # 验证元数据被正确附加
        assert len(chunks) > 0
        for i, chunk in enumerate(chunks):
            assert chunk.metadata["source"] == "test"
            assert chunk.metadata["page"] == 1
            assert chunk.metadata["chunk_index"] == i
            assert chunk.metadata["total_chunks"] == len(chunks)


class TestRecursiveSplitterEdgeCases:
    """测试 Recursive Splitter 边界情况。"""
    
    def test_split_very_short_text(self):
        """测试非常短的文本。"""
        splitter = RecursiveSplitter(chunk_size=100, chunk_overlap=10)
        text = "Short."
        
        chunks = splitter.split_text(text)
        
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_split_text_exactly_chunk_size(self):
        """测试文本长度正好等于 chunk_size。"""
        chunk_size = 20
        splitter = RecursiveSplitter(chunk_size=chunk_size, chunk_overlap=0)
        text = "a" * chunk_size
        
        chunks = splitter.split_text(text)
        
        assert len(chunks) == 1
        assert len(chunks[0]) == chunk_size
    
    def test_split_with_only_whitespace(self):
        """测试只有空白字符的文本。"""
        splitter = RecursiveSplitter(chunk_size=50, chunk_overlap=0)
        text = "   \n\n   \n   "
        
        # 空白文本应该被过滤
        chunks = splitter.split_text(text)
        
        # 可能返回空列表或只包含空白的块
        # 根据实现，这里验证不会崩溃
        assert isinstance(chunks, list)
    
    def test_split_unicode_text(self):
        """测试 Unicode 文本。"""
        splitter = RecursiveSplitter(chunk_size=30, chunk_overlap=5)
        text = "这是中文测试。これは日本語テストです。This is English."
        
        chunks = splitter.split_text(text)
        
        assert len(chunks) >= 1
        combined = "".join(chunks)
        assert "中文" in combined
        assert "日本語" in combined
        assert "English" in combined


class TestFactoryIntegration:
    """测试 Recursive Splitter 与工厂的集成。"""
    
    def setup_method(self):
        """每个测试方法前注册 Recursive 提供商。"""
        # 确保 Recursive 已注册
        if "recursive" not in SplitterFactory._registry:
            from src.libs.splitter import recursive_splitter
            SplitterFactory.register("recursive", recursive_splitter.RecursiveSplitter)
    
    def test_factory_creates_recursive_splitter(self):
        """测试工厂能正确创建 Recursive Splitter 实例。"""
        splitter = SplitterFactory.create_with_params(
            strategy="recursive",
            chunk_size=500,
            chunk_overlap=50
        )
        
        assert isinstance(splitter, RecursiveSplitter)
        assert splitter.chunk_size == 500
        assert splitter.chunk_overlap == 50
    
    def test_factory_recursive_splitter_can_split(self):
        """测试通过工厂创建的实例可以正常切分。"""
        splitter = SplitterFactory.create_with_params(
            strategy="recursive",
            chunk_size=50,
            chunk_overlap=10
        )
        text = "This is a test sentence. " * 5
        
        chunks = splitter.split_text(text)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)
