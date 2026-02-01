"""
设置模块 - 配置加载与验证。

此模块提供：
- Settings 数据类：结构化配置容器
- load_settings()：从 YAML 文件加载和解析设置
- validate_settings()：验证必需的配置字段
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class SettingsError(Exception):
    """配置错误时抛出的异常。"""
    pass


@dataclass
class LLMSettings:
    """LLM 配置设置。"""
    provider: str
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: int = 30


@dataclass
class EmbeddingSettings:
    """嵌入模型配置设置。"""
    provider: str
    model: str
    api_key: Optional[str] = None
    dimensions: int = 1536
    batch_size: int = 100


@dataclass
class VectorStoreSettings:
    """向量存储配置设置。"""
    provider: str
    persist_directory: str = "./data/db/chroma"
    collection_name: str = "default"


@dataclass
class RetrievalSettings:
    """检索配置设置。"""
    dense_weight: float = 0.7
    sparse_weight: float = 0.3
    top_k: int = 10
    rrf_k: int = 60


@dataclass
class RerankSettings:
    """重排序配置设置。"""
    provider: str = "none"
    model: Optional[str] = None
    top_k: int = 5


@dataclass
class EvaluationSettings:
    """评估配置设置。"""
    provider: str = "custom"
    metrics: List[str] = field(default_factory=lambda: ["hit_rate", "mrr"])


@dataclass
class ObservabilitySettings:
    """可观测性配置设置。"""
    log_level: str = "INFO"
    trace_enabled: bool = True
    trace_output: str = "./logs/traces.jsonl"
    structured_logging: bool = True


@dataclass
class IngestionSettings:
    """数据摄取配置设置。"""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    batch_size: int = 50


@dataclass
class Settings:
    """
    RAG MCP Server 的主设置容器。
    
    包含所有配置部分：
    - llm: LLM 提供商设置
    - embedding: 嵌入模型服务设置
    - vector_store: 向量数据库设置
    - retrieval: 检索参数
    - rerank: 重排序设置
    - evaluation: 评估指标设置
    - observability: 日志和跟踪设置
    - ingestion: 数据摄取设置
    """
    llm: LLMSettings
    embedding: EmbeddingSettings
    vector_store: VectorStoreSettings
    retrieval: RetrievalSettings
    rerank: RerankSettings
    evaluation: EvaluationSettings
    observability: ObservabilitySettings
    ingestion: IngestionSettings


def _resolve_env_vars(value: Any) -> Any:
    """
    解析配置值中的环境变量引用。
    
    支持 ${VAR_NAME} 语法进行环境变量替换。
    
    Args:
        value: 可能包含环境变量引用的配置值。
        
    Returns:
        解析后的值，已替换环境变量。
    """
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        env_var = value[2:-1]
        return os.environ.get(env_var)
    return value


def _parse_section(data: Dict[str, Any], section_name: str, dataclass_type: type) -> Any:
    """
    将配置部分解析为数据类实例。
    
    Args:
        data: 原始配置字典。
        section_name: 要解析的部分名称。
        dataclass_type: 目标数据类类型。
        
    Returns:
        带有解析值的数据类实例。
        
    Raises:
        SettingsError: 如果缺少必需部分。
    """
    section_data = data.get(section_name, {})
    if section_data is None:
        section_data = {}
    
    # 解析值中的环境变量
    resolved_data = {
        key: _resolve_env_vars(value) 
        for key, value in section_data.items()
    }
    
    try:
        return dataclass_type(**resolved_data)
    except TypeError as e:
        raise SettingsError(f"Invalid configuration for '{section_name}': {e}")


def validate_settings(settings: Settings) -> None:
    """
    验证所有必需设置是否存在且有效。
    
    Args:
        settings: 要验证的 Settings 实例。
        
    Raises:
        SettingsError: 如果验证失败，并包含缺失/无效字段的详细信息。
    """
    errors = []
    
    # 验证 LLM 设置
    if not settings.llm.provider:
        errors.append("llm.provider is required")
    if not settings.llm.model:
        errors.append("llm.model is required")
    
    # 验证 Embedding 设置
    if not settings.embedding.provider:
        errors.append("embedding.provider is required")
    if not settings.embedding.model:
        errors.append("embedding.model is required")
    
    # 验证 Vector Store 设置
    if not settings.vector_store.provider:
        errors.append("vector_store.provider is required")
    
    # 验证检索权重总和为 1.0（带容差）
    weight_sum = settings.retrieval.dense_weight + settings.retrieval.sparse_weight
    if abs(weight_sum - 1.0) > 0.01:
        errors.append(
            f"retrieval.dense_weight + retrieval.sparse_weight must equal 1.0 "
            f"(got {weight_sum})"
        )
    
    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise SettingsError(error_msg)


def load_settings(path: str) -> Settings:
    """
    从 YAML 配置文件加载设置。
    
    Args:
        path: settings.yaml 文件的路径。
        
    Returns:
        Settings: 解析并验证后的设置对象。
        
    Raises:
        SettingsError: 如果文件未找到、解析错误或验证失败。
    """
    config_path = Path(path)
    
    if not config_path.exists():
        raise SettingsError(f"Configuration file not found: {path}")
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise SettingsError(f"Failed to parse configuration file: {e}")
    
    if data is None:
        raise SettingsError("Configuration file is empty")
    
    # 解析每个部分
    settings = Settings(
        llm=_parse_section(data, "llm", LLMSettings),
        embedding=_parse_section(data, "embedding", EmbeddingSettings),
        vector_store=_parse_section(data, "vector_store", VectorStoreSettings),
        retrieval=_parse_section(data, "retrieval", RetrievalSettings),
        rerank=_parse_section(data, "rerank", RerankSettings),
        evaluation=_parse_section(data, "evaluation", EvaluationSettings),
        observability=_parse_section(data, "observability", ObservabilitySettings),
        ingestion=_parse_section(data, "ingestion", IngestionSettings),
    )
    
    # 验证设置
    validate_settings(settings)
    
    return settings
