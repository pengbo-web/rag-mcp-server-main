"""
Recursive Splitter 实现模块。

此模块实现递归文本切分器，支持 Markdown 结构感知切分。
优先在自然边界（段落、标题、代码块）切分，保持文本完整性。
"""

import re
from typing import List, Optional

from .base_splitter import BaseSplitter, SplitterError


class RecursiveSplitter(BaseSplitter):
    """
    递归文本切分器。
    
    按优先级递归尝试不同的分隔符：
    1. 双换行（段落边界）
    2. 单换行（行边界）
    3. 句子边界（。！？等）
    4. 词语边界（空格、标点）
    5. 字符级切分（最后手段）
    
    特点：
    - Markdown 结构感知（保持标题、代码块完整）
    - 保持语义连贯性
    - 支持配置化的 chunk_size 和 chunk_overlap
    """
    
    # 默认分隔符优先级（从高到低）
    DEFAULT_SEPARATORS = [
        "\n\n",      # 段落边界
        "\n",        # 行边界
        "。",        # 中文句号
        "！",        # 中文感叹号
        "？",        # 中文问号
        ".",         # 英文句号
        "!",         # 英文感叹号
        "?",         # 英文问号
        ";",         # 分号
        ",",         # 逗号
        " ",         # 空格
        "",          # 字符级（最后手段）
    ]
    
    # Markdown 代码块模式
    CODE_BLOCK_PATTERN = re.compile(r"```[\s\S]*?```", re.MULTILINE)
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
        keep_separator: bool = True,
        **kwargs
    ):
        """
        初始化 Recursive Splitter 实例。
        
        Args:
            chunk_size: 目标块大小（字符数）
            chunk_overlap: 块之间的重叠字符数
            separators: 自定义分隔符列表（优先级从高到低）
            keep_separator: 是否保留分隔符
            **kwargs: 其他参数
        """
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            **kwargs
        )
        
        self.separators = separators if separators is not None else self.DEFAULT_SEPARATORS
        self.keep_separator = keep_separator
    
    def split_text(
        self,
        text: str,
        **kwargs
    ) -> List[str]:
        """
        递归切分文本。
        
        Args:
            text: 要切分的文本
            **kwargs: 可选的切分参数
            
        Returns:
            List[str]: 切分后的文本块列表
            
        Raises:
            SplitterError: 当切分失败时抛出
        """
        if not text:
            raise SplitterError("Text cannot be empty")
        
        if not isinstance(text, str):
            raise SplitterError(f"Text must be a string, got {type(text)}")
        
        # 预处理：保护 Markdown 代码块
        code_blocks = []
        text_with_placeholders = text
        
        for i, match in enumerate(self.CODE_BLOCK_PATTERN.finditer(text)):
            placeholder = f"__CODE_BLOCK_{i}__"
            code_blocks.append(match.group(0))
            text_with_placeholders = text_with_placeholders.replace(
                match.group(0), placeholder, 1
            )
        
        # 执行递归切分
        chunks = self._split_recursive(text_with_placeholders, self.separators)
        
        # 恢复代码块
        final_chunks = []
        for chunk in chunks:
            restored_chunk = chunk
            for i, code_block in enumerate(code_blocks):
                placeholder = f"__CODE_BLOCK_{i}__"
                restored_chunk = restored_chunk.replace(placeholder, code_block)
            
            # 只添加非空块
            if restored_chunk.strip():
                final_chunks.append(restored_chunk)
        
        return final_chunks
    
    def _split_recursive(
        self,
        text: str,
        separators: List[str]
    ) -> List[str]:
        """
        递归切分文本的内部实现。
        
        Args:
            text: 要切分的文本
            separators: 分隔符列表（优先级从高到低）
            
        Returns:
            List[str]: 切分后的文本块
        """
        final_chunks = []
        
        # 当前使用的分隔符
        separator = separators[-1] if separators else ""
        new_separators = []
        
        # 找到第一个有效的分隔符
        for i, sep in enumerate(separators):
            if sep == "":
                separator = sep
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break
        
        # 使用分隔符切分文本
        if separator == "":
            # 字符级切分
            splits = list(text)
        else:
            splits = self._split_text_with_separator(text, separator)
        
        # 合并小块
        good_splits = []
        current_chunk = ""
        
        for split in splits:
            if len(current_chunk) + len(split) <= self.chunk_size:
                # 可以合并
                current_chunk += split
            else:
                if current_chunk:
                    good_splits.append(current_chunk)
                
                # 如果单个 split 超过 chunk_size，需要进一步切分
                if len(split) > self.chunk_size:
                    if new_separators:
                        # 递归使用下一级分隔符
                        sub_chunks = self._split_recursive(split, new_separators)
                        good_splits.extend(sub_chunks)
                        current_chunk = ""
                    else:
                        # 强制字符级切分
                        for i in range(0, len(split), self.chunk_size):
                            good_splits.append(split[i:i + self.chunk_size])
                        current_chunk = ""
                else:
                    current_chunk = split
        
        if current_chunk:
            good_splits.append(current_chunk)
        
        # 添加重叠
        if self.chunk_overlap > 0 and len(good_splits) > 1:
            final_chunks = self._add_overlap(good_splits)
        else:
            final_chunks = good_splits
        
        return final_chunks
    
    def _split_text_with_separator(
        self,
        text: str,
        separator: str
    ) -> List[str]:
        """
        使用指定分隔符切分文本。
        
        Args:
            text: 要切分的文本
            separator: 分隔符
            
        Returns:
            List[str]: 切分后的文本块（可能保留分隔符）
        """
        if not separator:
            return [text]
        
        splits = text.split(separator)
        
        if self.keep_separator:
            # 将分隔符添加回块的末尾（除了最后一块）
            result = []
            for i, split in enumerate(splits):
                if i < len(splits) - 1:
                    result.append(split + separator)
                else:
                    result.append(split)
            return result
        else:
            return splits
    
    def _add_overlap(
        self,
        chunks: List[str]
    ) -> List[str]:
        """
        在相邻块之间添加重叠。
        
        Args:
            chunks: 原始文本块列表
            
        Returns:
            List[str]: 添加重叠后的文本块列表
        """
        if not chunks or len(chunks) == 1:
            return chunks
        
        overlapped_chunks = []
        
        for i, chunk in enumerate(chunks):
            if i == 0:
                # 第一块：只添加后向重叠
                if len(chunk) + len(chunks[i + 1][:self.chunk_overlap]) <= self.chunk_size * 1.2:
                    overlapped_chunk = chunk + chunks[i + 1][:self.chunk_overlap]
                else:
                    overlapped_chunk = chunk
            elif i == len(chunks) - 1:
                # 最后一块：只添加前向重叠
                prev_overlap = chunks[i - 1][-self.chunk_overlap:]
                if len(prev_overlap) + len(chunk) <= self.chunk_size * 1.2:
                    overlapped_chunk = prev_overlap + chunk
                else:
                    overlapped_chunk = chunk
            else:
                # 中间块：添加前后重叠
                prev_overlap = chunks[i - 1][-self.chunk_overlap:]
                next_overlap = chunks[i + 1][:self.chunk_overlap]
                
                overlapped_chunk = prev_overlap + chunk
                if len(overlapped_chunk) + len(next_overlap) <= self.chunk_size * 1.2:
                    overlapped_chunk += next_overlap
            
            overlapped_chunks.append(overlapped_chunk)
        
        return overlapped_chunks
