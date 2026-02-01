#!/usr/bin/env python3
"""
模块化 RAG MCP Server - 主入口文件

这是 MCP Server 的主入口点。
它加载配置、初始化服务器并开始监听请求。
"""

import sys
from pathlib import Path

# 将 src 添加到 Python 路径以便导入
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))


def main() -> int:
    """
    MCP Server 的主入口函数。
    
    返回:
        int: 退出码（0 表示成功，非零表示错误）
    """
    from core.settings import load_settings, SettingsError
    from observability.logger import get_logger
    
    logger = get_logger(__name__)
    
    try:
        # 加载并验证配置
        config_path = Path(__file__).parent / "config" / "settings.yaml"
        logger.info(f"Loading configuration from {config_path}")
        
        settings = load_settings(str(config_path))
        logger.info("Configuration loaded successfully")
        
        # TODO: 初始化 MCP Server (阶段 E)
        logger.info("MCP Server initialization placeholder - to be implemented in Phase E")
        
        return 0
        
    except SettingsError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
