"""
测试 Settings 模块

测试 src/core/settings.py 的功能
"""

import pytest
from pathlib import Path


class TestSettingsImport:
    """测试 settings 模块可以被导入。"""
    
    @pytest.mark.unit
    def test_import_settings_module(self):
        """验证 settings 模块导入正确。"""
        from core import settings
        assert hasattr(settings, "Settings")
        assert hasattr(settings, "load_settings")
        assert hasattr(settings, "validate_settings")
    
    @pytest.mark.unit
    def test_import_settings_class(self):
        """验证 Settings 类可以被导入。"""
        from core.settings import Settings
        assert Settings is not None
    
    @pytest.mark.unit
    def test_import_load_settings(self):
        """验证 load_settings 函数可以被导入。"""
        from core.settings import load_settings
        assert callable(load_settings)
    
    @pytest.mark.unit
    def test_import_from_core(self):
        """验证从 core 包的导出。"""
        from core import Settings, load_settings, validate_settings, SettingsError
        assert Settings is not None
        assert callable(load_settings)
        assert callable(validate_settings)
        assert issubclass(SettingsError, Exception)


class TestLoadSettings:
    """测试配置加载功能。"""
    
    @pytest.mark.unit
    def test_load_settings_success(self, temp_config_file: Path):
        """测试从文件成功加载配置。"""
        from core.settings import load_settings
        
        settings = load_settings(str(temp_config_file))
        
        assert settings.llm.provider == "openai"
        assert settings.llm.model == "gpt-4o-mini"
        assert settings.embedding.provider == "openai"
        assert settings.vector_store.provider == "chroma"
    
    @pytest.mark.unit
    def test_load_settings_file_not_found(self):
        """测试配置文件不存在时的错误。"""
        from core.settings import load_settings, SettingsError
        
        with pytest.raises(SettingsError, match="not found"):
            load_settings("nonexistent/path/settings.yaml")
    
    @pytest.mark.unit
    def test_validate_settings_invalid_weights(self, temp_config_file: Path, sample_settings_dict: dict):
        """测试检索权重总和不为 1.0 时验证失败。"""
        import yaml
        from core.settings import load_settings, SettingsError
        
        # 将权重修改为无效值
        sample_settings_dict["retrieval"]["dense_weight"] = 0.5
        sample_settings_dict["retrieval"]["sparse_weight"] = 0.2
        
        with open(temp_config_file, "w", encoding="utf-8") as f:
            yaml.dump(sample_settings_dict, f)
        
        with pytest.raises(SettingsError, match="must equal 1.0"):
            load_settings(str(temp_config_file))


class TestSettingsDataclass:
    """测试 Settings 数据类结构。"""
    
    @pytest.mark.unit
    def test_settings_has_all_sections(self, temp_config_file: Path):
        """测试 Settings 包含所有必需的配置段。"""
        from core.settings import load_settings
        
        settings = load_settings(str(temp_config_file))
        
        assert hasattr(settings, "llm")
        assert hasattr(settings, "embedding")
        assert hasattr(settings, "vector_store")
        assert hasattr(settings, "retrieval")
        assert hasattr(settings, "rerank")
        assert hasattr(settings, "evaluation")
        assert hasattr(settings, "observability")
        assert hasattr(settings, "ingestion")
