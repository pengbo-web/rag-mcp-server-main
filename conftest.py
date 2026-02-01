"""
Pytest Configuration and Shared Fixtures

This module provides:
- Common test fixtures
- Pytest hooks and configuration
- Shared test utilities
"""

import os
import sys
from pathlib import Path

import pytest

# Ensure src is in path for imports
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return PROJECT_ROOT


@pytest.fixture
def config_path(project_root: Path) -> Path:
    """Return the path to the test configuration file."""
    return project_root / "config" / "settings.yaml"


@pytest.fixture
def sample_settings_dict() -> dict:
    """Return a sample settings dictionary for testing."""
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
    """Create a temporary configuration file for testing."""
    import yaml
    
    config_file = tmp_path / "settings.yaml"
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(sample_settings_dict, f)
    return config_file
