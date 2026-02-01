#!/usr/bin/env python3
"""
Modular RAG MCP Server - Main Entry Point

This is the main entry point for the MCP Server.
It loads configuration, initializes the server, and starts listening for requests.
"""

import sys
from pathlib import Path

# Add src to Python path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))


def main() -> int:
    """
    Main entry point for the MCP Server.
    
    Returns:
        int: Exit code (0 for success, non-zero for errors)
    """
    from core.settings import load_settings, SettingsError
    from observability.logger import get_logger
    
    logger = get_logger(__name__)
    
    try:
        # Load and validate configuration
        config_path = Path(__file__).parent / "config" / "settings.yaml"
        logger.info(f"Loading configuration from {config_path}")
        
        settings = load_settings(str(config_path))
        logger.info("Configuration loaded successfully")
        
        # TODO: Initialize MCP Server (Phase E)
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
