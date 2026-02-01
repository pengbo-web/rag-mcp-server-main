"""
Settings Module - Configuration loading and validation.

This module provides:
- Settings dataclass: Structured configuration container
- load_settings(): Load and parse settings from YAML file
- validate_settings(): Validate required configuration fields
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class SettingsError(Exception):
    """Exception raised for configuration errors."""
    pass


@dataclass
class LLMSettings:
    """LLM configuration settings."""
    provider: str
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: int = 30


@dataclass
class EmbeddingSettings:
    """Embedding configuration settings."""
    provider: str
    model: str
    api_key: Optional[str] = None
    dimensions: int = 1536
    batch_size: int = 100


@dataclass
class VectorStoreSettings:
    """Vector store configuration settings."""
    provider: str
    persist_directory: str = "./data/db/chroma"
    collection_name: str = "default"


@dataclass
class RetrievalSettings:
    """Retrieval configuration settings."""
    dense_weight: float = 0.7
    sparse_weight: float = 0.3
    top_k: int = 10
    rrf_k: int = 60


@dataclass
class RerankSettings:
    """Reranker configuration settings."""
    provider: str = "none"
    model: Optional[str] = None
    top_k: int = 5


@dataclass
class EvaluationSettings:
    """Evaluation configuration settings."""
    provider: str = "custom"
    metrics: List[str] = field(default_factory=lambda: ["hit_rate", "mrr"])


@dataclass
class ObservabilitySettings:
    """Observability configuration settings."""
    log_level: str = "INFO"
    trace_enabled: bool = True
    trace_output: str = "./logs/traces.jsonl"
    structured_logging: bool = True


@dataclass
class IngestionSettings:
    """Ingestion configuration settings."""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    batch_size: int = 50


@dataclass
class Settings:
    """
    Main settings container for the RAG MCP Server.
    
    Contains all configuration sections:
    - llm: LLM provider settings
    - embedding: Embedding service settings
    - vector_store: Vector database settings
    - retrieval: Retrieval parameters
    - rerank: Reranker settings
    - evaluation: Evaluation metrics settings
    - observability: Logging and tracing settings
    - ingestion: Data ingestion settings
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
    Resolve environment variable references in configuration values.
    
    Supports ${VAR_NAME} syntax for environment variable substitution.
    
    Args:
        value: Configuration value that may contain env var references.
        
    Returns:
        Resolved value with environment variables substituted.
    """
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        env_var = value[2:-1]
        return os.environ.get(env_var)
    return value


def _parse_section(data: Dict[str, Any], section_name: str, dataclass_type: type) -> Any:
    """
    Parse a configuration section into a dataclass instance.
    
    Args:
        data: Raw configuration dictionary.
        section_name: Name of the section to parse.
        dataclass_type: Target dataclass type.
        
    Returns:
        Instance of the dataclass with parsed values.
        
    Raises:
        SettingsError: If required section is missing.
    """
    section_data = data.get(section_name, {})
    if section_data is None:
        section_data = {}
    
    # Resolve environment variables in values
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
    Validate that all required settings are present and valid.
    
    Args:
        settings: Settings instance to validate.
        
    Raises:
        SettingsError: If validation fails with details about missing/invalid fields.
    """
    errors = []
    
    # Validate LLM settings
    if not settings.llm.provider:
        errors.append("llm.provider is required")
    if not settings.llm.model:
        errors.append("llm.model is required")
    
    # Validate Embedding settings
    if not settings.embedding.provider:
        errors.append("embedding.provider is required")
    if not settings.embedding.model:
        errors.append("embedding.model is required")
    
    # Validate Vector Store settings
    if not settings.vector_store.provider:
        errors.append("vector_store.provider is required")
    
    # Validate retrieval weights sum to 1.0 (with tolerance)
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
    Load settings from a YAML configuration file.
    
    Args:
        path: Path to the settings.yaml file.
        
    Returns:
        Settings: Parsed and validated settings object.
        
    Raises:
        SettingsError: If file not found, parse error, or validation fails.
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
    
    # Parse each section
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
    
    # Validate settings
    validate_settings(settings)
    
    return settings
