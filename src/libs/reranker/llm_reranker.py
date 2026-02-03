"""
LLMReranker 实现。

使用 LLM 对候选项进行重排序，支持 prompt 模板读取和结构化输出。
适用于需要深度语义理解的重排序场景。
"""

import json
import re
from pathlib import Path
from typing import Optional

from src.libs.llm.base_llm import BaseLLM, Message, LLMError
from .base_reranker import BaseReranker, RerankCandidate, RerankResult, RerankerError


class LLMReranker(BaseReranker):
    """使用 LLM 进行重排序的 Reranker。
    
    通过 LLM 对候选项进行深度语义分析，输出重排序分数。
    支持：
    - 从文件加载 prompt 模板
    - 结构化输出解析（每行一个分数）
    - 错误处理和降级策略
    
    Args:
        llm: LLM 实例，用于生成重排序分数
        prompt_path: prompt 模板文件路径（可选，默认为 config/prompts/rerank.txt）
        temperature: LLM 生成温度（可选，默认 0.0 以提高一致性）
        max_retries: 最大重试次数（可选，默认 2）
        **kwargs: 其他配置参数
    """
    
    def __init__(
        self,
        llm: BaseLLM,
        prompt_path: Optional[str] = None,
        temperature: float = 0.0,
        max_retries: int = 2,
        **kwargs
    ):
        """初始化 LLMReranker。
        
        Args:
            llm: LLM 实例
            prompt_path: prompt 模板文件路径
            temperature: LLM 生成温度
            max_retries: 最大重试次数
            **kwargs: 其他配置参数
        """
        super().__init__(backend="llm", **kwargs)
        self.llm = llm
        self.temperature = temperature
        self.max_retries = max_retries
        
        # 加载 prompt 模板
        if prompt_path is None:
            # 默认路径：相对于项目根目录
            prompt_path = "config/prompts/rerank.txt"
        
        self.prompt_template = self._load_prompt(prompt_path)
    
    def _load_prompt(self, prompt_path: str) -> str:
        """从文件加载 prompt 模板。
        
        Args:
            prompt_path: prompt 文件路径
            
        Returns:
            prompt 模板字符串
            
        Raises:
            RerankerError: 如果文件不存在或读取失败
        """
        try:
            path = Path(prompt_path)
            if not path.is_absolute():
                # 如果是相对路径，从项目根目录解析
                # 假设 src/libs/reranker 是从项目根目录的相对位置
                root = Path(__file__).parent.parent.parent.parent
                path = root / prompt_path
            
            if not path.exists():
                raise RerankerError(f"Prompt 文件不存在: {path}")
            
            return path.read_text(encoding="utf-8")
        except Exception as e:
            raise RerankerError(f"无法加载 prompt 模板: {e}")
    
    def _format_prompt(self, query: str, candidates: list[RerankCandidate]) -> str:
        """格式化 prompt 模板。
        
        Args:
            query: 查询文本
            candidates: 候选项列表
            
        Returns:
            格式化后的 prompt
        """
        # 格式化候选项为编号列表
        passages = []
        for i, candidate in enumerate(candidates):
            passages.append(f"{i+1}. {candidate.text}")
        passages_text = "\n".join(passages)
        
        # 替换模板中的占位符
        prompt = self.prompt_template.format(
            query=query,
            passages=passages_text
        )
        
        return prompt
    
    def _parse_scores(self, output: str, num_candidates: int) -> list[float]:
        """解析 LLM 输出为分数列表。
        
        Args:
            output: LLM 原始输出
            num_candidates: 候选项数量
            
        Returns:
            分数列表
            
        Raises:
            RerankerError: 如果解析失败或分数数量不匹配
        """
        # 提取所有数字（支持整数和浮点数）
        lines = output.strip().split("\n")
        scores = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 尝试提取数字（支持 "3", "2.5", "Score: 3" 等格式）
            match = re.search(r"(\d+\.?\d*)", line)
            if match:
                try:
                    score = float(match.group(1))
                    scores.append(score)
                except ValueError:
                    continue
        
        # 验证分数数量
        if len(scores) != num_candidates:
            raise RerankerError(
                f"分数数量 ({len(scores)}) 与候选项数量 ({num_candidates}) 不匹配。"
                f"LLM 输出: {output[:200]}"
            )
        
        return scores
    
    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        top_k: Optional[int] = None,
        **kwargs
    ) -> list[RerankResult]:
        """使用 LLM 对候选项进行重排序。
        
        Args:
            query: 查询文本
            candidates: 候选项列表
            top_k: 返回前 k 个结果（可选，默认返回全部）
            **kwargs: 额外参数（如 trace context）
            
        Returns:
            重排序后的结果列表，按新分数降序排列
            
        Raises:
            RerankerError: 重排序失败时抛出
        """
        # 验证输入
        self.validate_candidates(candidates)
        
        # 格式化 prompt
        prompt = self._format_prompt(query, candidates)
        
        # 调用 LLM，带重试机制
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                # 构造消息
                messages = [Message(role="user", content=prompt)]
                
                # 调用 LLM
                response = self.llm.chat(
                    messages,
                    temperature=self.temperature,
                    **kwargs
                )
                
                # 解析分数
                scores = self._parse_scores(response.content, len(candidates))
                
                # 创建结果
                results = self.create_results(candidates, scores)
                
                # 应用 top_k 限制
                if top_k is not None and top_k > 0:
                    results = results[:top_k]
                
                return results
                
            except (LLMError, RerankerError) as e:
                last_error = e
                if attempt < self.max_retries:
                    # 继续重试
                    continue
                else:
                    # 最后一次重试失败，抛出错误
                    raise RerankerError(
                        f"LLM 重排序失败（重试 {self.max_retries} 次后）: {last_error}"
                    )
        
        # 理论上不会到达这里，但为了类型安全
        raise RerankerError(f"LLM 重排序失败: {last_error}")
