"""
Core Layer - Core business logic for the RAG system.

This package contains:
- Settings management and configuration loading
- Query engine components (processor, retriever, fusion, reranker)
- Response building and formatting
- Trace context for observability
"""

from .settings import Settings, load_settings, validate_settings, SettingsError

__all__ = ["Settings", "load_settings", "validate_settings", "SettingsError"]
