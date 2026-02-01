# 样例 Markdown 文档

这是一个用于测试的 Markdown 文档。

## 项目简介

本项目是一个基于 RAG (Retrieval-Augmented Generation) 的知识问答系统。

### 核心特性

- **混合检索**: 结合稠密检索和稀疏检索
- **可插拔架构**: 支持多种 LLM 和 Embedding 提供商
- **MCP 协议**: 与 GitHub Copilot 无缝集成

### 技术栈

| 组件 | 技术选型 |
|------|---------|
| LLM | OpenAI, Azure, Ollama |
| Embedding | OpenAI, Sentence-Transformers |
| Vector Store | ChromaDB |
| 测试框架 | pytest |

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 运行测试
pytest tests/

# 启动服务
python main.py
```

## 配置示例

```yaml
llm:
  provider: openai
  model: gpt-4o-mini
  temperature: 0.7

embedding:
  provider: openai
  model: text-embedding-3-small
```

## 许可证

MIT License

---

**注意**: 这是一个测试文档，内容仅用于验证文档解析功能。
