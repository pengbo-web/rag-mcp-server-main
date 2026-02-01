"""
Test Settings Module

Tests for src/core/settings.py
"""

import pytest
from pathlib import Path


class TestSettingsImport:
    """Test that settings module can be imported."""
    
    @pytest.mark.unit
    def test_import_settings_module(self):
        """Verify settings module imports correctly."""
        from core import settings
        assert hasattr(settings, "Settings")
        assert hasattr(settings, "load_settings")
        assert hasattr(settings, "validate_settings")
    
    @pytest.mark.unit
    def test_import_settings_class(self):
        """Verify Settings class can be imported."""
        from core.settings import Settings
        assert Settings is not None
    
    @pytest.mark.unit
    def test_import_load_settings(self):
        """Verify load_settings function can be imported."""
        from core.settings import load_settings
        assert callable(load_settings)
    
    @pytest.mark.unit
    def test_import_from_core(self):
        """Verify exports from core package."""
        from core import Settings, load_settings, validate_settings, SettingsError
        assert Settings is not None
        assert callable(load_settings)
        assert callable(validate_settings)
        assert issubclass(SettingsError, Exception)


class TestLoadSettings:
    """Test settings loading functionality."""
    
    @pytest.mark.unit
    def test_load_settings_success(self, temp_config_file: Path):
        """Test successful settings loading from file."""
        from core.settings import load_settings
        
        settings = load_settings(str(temp_config_file))
        
        assert settings.llm.provider == "openai"
        assert settings.llm.model == "gpt-4o-mini"
        assert settings.embedding.provider == "openai"
        assert settings.vector_store.provider == "chroma"
    
    @pytest.mark.unit
    def test_load_settings_file_not_found(self):
        """Test error when config file doesn't exist."""
        from core.settings import load_settings, SettingsError
        
        with pytest.raises(SettingsError, match="not found"):
            load_settings("nonexistent/path/settings.yaml")
    
    @pytest.mark.unit
    def test_validate_settings_invalid_weights(self, temp_config_file: Path, sample_settings_dict: dict):
        """Test validation fails when retrieval weights don't sum to 1.0."""
        import yaml
        from core.settings import load_settings, SettingsError
        
        # Modify weights to invalid values
        sample_settings_dict["retrieval"]["dense_weight"] = 0.5
        sample_settings_dict["retrieval"]["sparse_weight"] = 0.2
        
        with open(temp_config_file, "w", encoding="utf-8") as f:
            yaml.dump(sample_settings_dict, f)
        
        with pytest.raises(SettingsError, match="must equal 1.0"):
            load_settings(str(temp_config_file))


class TestSettingsDataclass:
    """Test Settings dataclass structure."""
    
    @pytest.mark.unit
    def test_settings_has_all_sections(self, temp_config_file: Path):
        """Test that Settings has all required configuration sections."""
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
