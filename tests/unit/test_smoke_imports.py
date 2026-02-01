"""
冒烟测试 (Smoke Tests) - 验证关键模块可以正常导入。

这些测试确保项目的基本结构正确，所有关键包都可以被导入。
如果这些测试失败，说明项目的基础设施存在问题。
"""

import pytest


class TestCoreImports:
    """测试核心模块导入。"""
    
    @pytest.mark.unit
    def test_import_core_settings(self):
        """验证可以导入 core.settings 模块。"""
        from core import settings
        assert hasattr(settings, "Settings")
        assert hasattr(settings, "load_settings")
        assert hasattr(settings, "validate_settings")
    
    @pytest.mark.unit
    def test_import_core_package(self):
        """验证可以从 core 包导入关键类。"""
        from core import Settings, load_settings, validate_settings, SettingsError
        assert Settings is not None
        assert callable(load_settings)
        assert callable(validate_settings)
        assert issubclass(SettingsError, Exception)


class TestObservabilityImports:
    """测试可观测性模块导入。"""
    
    @pytest.mark.unit
    def test_import_logger(self):
        """验证可以导入日志模块。"""
        from observability import logger
        assert hasattr(logger, "get_logger")
    
    @pytest.mark.unit
    def test_import_get_logger(self):
        """验证可以导入 get_logger 函数。"""
        from observability import get_logger
        assert callable(get_logger)
    
    @pytest.mark.unit
    def test_get_logger_returns_logger(self):
        """验证 get_logger 返回日志器实例。"""
        from observability import get_logger
        import logging
        
        logger = get_logger("test")
        assert isinstance(logger, logging.Logger)


class TestLibsImports:
    """测试 libs 层模块导入。"""
    
    @pytest.mark.unit
    def test_import_libs_llm(self):
        """验证可以导入 LLM 模块。"""
        from libs import llm
        assert llm is not None
    
    @pytest.mark.unit
    def test_import_libs_embedding(self):
        """验证可以导入 Embedding 模块。"""
        from libs import embedding
        assert embedding is not None
    
    @pytest.mark.unit
    def test_import_libs_vector_store(self):
        """验证可以导入 VectorStore 模块。"""
        from libs import vector_store
        assert vector_store is not None
    
    @pytest.mark.unit
    def test_import_libs_splitter(self):
        """验证可以导入 Splitter 模块。"""
        from libs import splitter
        assert splitter is not None
    
    @pytest.mark.unit
    def test_import_libs_reranker(self):
        """验证可以导入 Reranker 模块。"""
        from libs import reranker
        assert reranker is not None


class TestIngestionImports:
    """测试 Ingestion 层模块导入。"""
    
    @pytest.mark.unit
    def test_import_ingestion(self):
        """验证可以导入 Ingestion 模块。"""
        from ingestion import transform, embedding, storage
        assert transform is not None
        assert embedding is not None
        assert storage is not None


class TestMCPServerImports:
    """测试 MCP Server 层模块导入。"""
    
    @pytest.mark.unit
    def test_import_mcp_server(self):
        """验证可以导入 MCP Server 模块。"""
        from mcp_server import tools
        assert tools is not None


class TestProjectStructure:
    """测试项目结构完整性。"""
    
    @pytest.mark.unit
    def test_config_file_exists(self, config_path):
        """验证配置文件存在。"""
        assert config_path.exists(), f"配置文件不存在: {config_path}"
    
    @pytest.mark.unit
    def test_main_module_exists(self, project_root):
        """验证主模块文件存在。"""
        main_file = project_root / "main.py"
        assert main_file.exists(), "main.py 文件不存在"
    
    @pytest.mark.unit
    def test_pyproject_toml_exists(self, project_root):
        """验证 pyproject.toml 存在。"""
        pyproject = project_root / "pyproject.toml"
        assert pyproject.exists(), "pyproject.toml 文件不存在"
