"""
Libs 层 - 可插拔抽象层与工厂模式。

此包提供：
- 所有可插拔组件的抽象基类
- 组件实例化的工厂函数
- 每种组件类型的默认实现

支持的组件类型：
- LLM: 大语言模型客户端
- Embedding: 文本嵌入服务
- Splitter: 文本切分策略
- VectorStore: 向量数据库后端
- Reranker: 结果重排序策略
- Evaluator: RAG 评估指标
- Loader: 文档加载工具
"""

from . import llm, embedding, splitter, vector_store

__all__ = ["llm", "embedding", "splitter", "vector_store"]
