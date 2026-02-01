"""
Observability Layer - Logging, tracing, and monitoring.

This package provides:
- Structured logging (JSON format)
- Request tracing (trace_id, stages, metrics)
- Web dashboard (Streamlit)
- Evaluation metrics and reporting
"""

__all__ = ["get_logger"]

from observability.logger import get_logger
