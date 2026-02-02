"""
Core 层 - RAG 系统的核心业务逻辑。

此包包含：
- 设置管理和配置加载
- 查询引擎组件（处理器、检索器、融合、重排序）
- 响应构建和格式化
- 用于可观测性的跟踪上下文
"""

from .settings import (
    Settings,
    LLMSettings,
    EmbeddingSettings,
    VectorStoreSettings,
    RetrievalSettings,
    RerankSettings,
    EvaluationSettings,
    ObservabilitySettings,
    IngestionSettings,
    load_settings,
    validate_settings,
    SettingsError,
)

__all__ = [
    "Settings",
    "LLMSettings",
    "EmbeddingSettings",
    "VectorStoreSettings",
    "RetrievalSettings",
    "RerankSettings",
    "EvaluationSettings",
    "ObservabilitySettings",
    "IngestionSettings",
    "load_settings",
    "validate_settings",
    "SettingsError",
]
