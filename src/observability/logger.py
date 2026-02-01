"""
结构化日志器 - RAG MCP Server 的日志工具。

提供一致的日志接口，支持结构化输出。
"""

import logging
import sys
from typing import Optional


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    获取具有一致配置的日志器实例。
    
    参数:
        name: 日志器名称。如果为 None，返回根日志器。
        
    返回:
        logging.Logger: 配置好的日志器实例
    """
    logger = logging.getLogger(name)
    
    # 仅在没有 handler 时配置（避免重复的 handler）
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    return logger
