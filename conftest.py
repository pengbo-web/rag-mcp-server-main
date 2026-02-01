"""
Pytest 配置和共享 Fixtures

本模块提供：
- 通用测试 fixtures
- Pytest 钩子和配置
- 共享测试工具
"""

import os
import sys
from pathlib import Path

import pytest

# 确保 src 在路径中以便导入
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


@pytest.fixture
def project_root() -> Path:
    """返回项目根目录。"""
    return PROJECT_ROOT


@pytest.fixture
def config_path(project_root: Path) -> Path:
    """返回测试配置文件的路径。"""
    return project_root / "config" / "settings.yaml"


@pytest.fixture
def sample_settings_dict() -> dict:
    """返回用于测试的样例配置字典。"""
    return {
        "llm": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "temperature": 0.7,
            "max_tokens": 2048,
        },
        "embedding": {
            "provider": "openai",
            "model": "text-embedding-3-small",
            "dimensions": 1536,
        },
        "vector_store": {
            "provider": "chroma",
            "persist_directory": "./data/db/chroma",
        },
        "retrieval": {
            "dense_weight": 0.7,
            "sparse_weight": 0.3,
            "top_k": 10,
        },
        "rerank": {
            "provider": "none",
        },
        "evaluation": {
            "provider": "custom",
            "metrics": ["hit_rate", "mrr"],
        },
        "observability": {
            "log_level": "INFO",
            "trace_enabled": True,
        },
        "ingestion": {
            "chunk_size": 1000,
            "chunk_overlap": 200,
        },
    }


@pytest.fixture
def temp_config_file(tmp_path: Path, sample_settings_dict: dict) -> Path:
    """创建用于测试的临时配置文件。"""
    import yaml
    
    config_file = tmp_path / "settings.yaml"
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(sample_settings_dict, f)
    return config_file
