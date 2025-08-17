"""
Configuration management for the misinformation detection tool.

This module handles loading and managing configuration from both
environment variables and YAML configuration files.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, List
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class LLMProviderConfig(BaseModel):
    """Configuration for a single LLM provider."""
    model: str
    max_tokens: int = 4096
    temperature: float = 0.1
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: int = 1


class LLMConfig(BaseModel):
    """Configuration for LLM providers and settings."""
    providers: Dict[str, LLMProviderConfig]
    default_provider: str = "gemini"
    fallback_provider: Optional[str] = None


class AnalysisConfig(BaseModel):
    """Configuration for content analysis settings."""
    
    class TextConfig(BaseModel):
        max_length: int = 2000
        timeout: int = 10
        chunk_size: int = 1000
        overlap: int = 100
    
    class ImageConfig(BaseModel):
        max_size: int = 5242880  # 5MB
        supported_formats: List[str] = ["jpg", "jpeg", "png", "gif", "webp"]
        timeout: int = 15
        max_dimensions: List[int] = [4096, 4096]
    
    class URLConfig(BaseModel):
        timeout: int = 10
        max_redirects: int = 5
        user_agent: str = "MisinformationDetectionTool/1.0"
        max_content_size: int = 10485760  # 10MB
    
    text: TextConfig = TextConfig()
    image: ImageConfig = ImageConfig()
    url: URLConfig = URLConfig()


class RiskAssessmentConfig(BaseModel):
    """Configuration for risk assessment calculations."""
    
    class Thresholds(BaseModel):
        low: float = 0.3
        medium: float = 0.6
        high: float = 0.8
    
    class IndicatorConfig(BaseModel):
        weight: float
        severity_multiplier: float = 1.0
    
    thresholds: Thresholds = Thresholds()
    confidence_threshold: float = 0.5
    indicators: Dict[str, IndicatorConfig] = {}


class CacheConfig(BaseModel):
    """Configuration for caching settings."""
    enabled: bool = True
    ttl: int = 3600  # 1 hour
    max_size: int = 1000
    
    class RedisConfig(BaseModel):
        enabled: bool = False
        url: str = "redis://localhost:6379/0"
        key_prefix: str = "misinformation_tool:"
    
    redis: RedisConfig = RedisConfig()


class Settings(BaseSettings):
    """Main application settings loaded from environment variables."""
    
    # LLM Provider Settings
    gemini_api_key: Optional[str] = Field(None, env="GEMINI_API_KEY")
    groq_api_key: Optional[str] = Field(None, env="GROQ_API_KEY")
    default_llm_provider: str = Field("gemini", env="DEFAULT_LLM_PROVIDER")
    
    # Application Settings
    log_level: str = Field("INFO", env="LOG_LEVEL")
    max_file_size: int = Field(5242880, env="MAX_FILE_SIZE")
    analysis_timeout: int = Field(30, env="ANALYSIS_TIMEOUT")
    cache_ttl: int = Field(3600, env="CACHE_TTL")
    
    # FastAPI Settings
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")
    debug: bool = Field(False, env="DEBUG")
    
    # Streamlit Settings
    streamlit_host: str = Field("0.0.0.0", env="STREAMLIT_HOST")
    streamlit_port: int = Field(8501, env="STREAMLIT_PORT")
    
    # Redis Settings
    redis_url: str = Field("redis://localhost:6379/0", env="REDIS_URL")
    redis_enabled: bool = Field(False, env="REDIS_ENABLED")
    
    # Security Settings
    secret_key: str = Field("change-me-in-production", env="SECRET_KEY")
    cors_origins: List[str] = Field(
        ["http://localhost:3000", "http://localhost:8501"], 
        env="CORS_ORIGINS"
    )
    
    # External API Settings
    search_api_key: Optional[str] = Field(None, env="SEARCH_API_KEY")
    search_engine_id: Optional[str] = Field(None, env="SEARCH_ENGINE_ID")
    
    # File Storage Settings
    upload_dir: str = Field("./uploads", env="UPLOAD_DIR")
    temp_dir: str = Field("./temp", env="TEMP_DIR")
    max_upload_size: int = Field(5242880, env="MAX_UPLOAD_SIZE")
    
    # Monitoring Settings
    enable_metrics: bool = Field(True, env="ENABLE_METRICS")
    log_file: str = Field("./logs/app.log", env="LOG_FILE")
    log_rotation: str = Field("daily", env="LOG_ROTATION")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


class ConfigManager:
    """Manages application configuration from multiple sources."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the configuration manager.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = config_path or "config/config.yaml"
        self.settings = Settings()
        self._yaml_config: Dict[str, Any] = {}
        self._load_yaml_config()
    
    def _load_yaml_config(self) -> None:
        """Load configuration from YAML file."""
        config_file = Path(self.config_path)
        
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    self._yaml_config = yaml.safe_load(f) or {}
            except Exception as e:
                print(f"Warning: Could not load config file {config_file}: {e}")
                self._yaml_config = {}
        else:
            print(f"Warning: Config file {config_file} not found, using defaults")
            self._yaml_config = {}
    
    @property
    def llm(self) -> LLMConfig:
        """Get LLM configuration."""
        llm_config = self._yaml_config.get("llm", {})
        
        # Convert provider configs to LLMProviderConfig objects
        providers = {}
        for name, config in llm_config.get("providers", {}).items():
            providers[name] = LLMProviderConfig(**config)
        
        return LLMConfig(
            providers=providers,
            default_provider=llm_config.get("default_provider", self.settings.default_llm_provider),
            fallback_provider=llm_config.get("fallback_provider")
        )
    
    @property
    def analysis(self) -> AnalysisConfig:
        """Get analysis configuration."""
        analysis_config = self._yaml_config.get("analysis", {})
        return AnalysisConfig(**analysis_config)
    
    @property
    def risk_assessment(self) -> RiskAssessmentConfig:
        """Get risk assessment configuration."""
        risk_config = self._yaml_config.get("risk_assessment", {})
        return RiskAssessmentConfig(**risk_config)
    
    @property
    def cache(self) -> CacheConfig:
        """Get cache configuration."""
        cache_config = self._yaml_config.get("cache", {})
        
        # Override with environment settings
        if self.settings.redis_enabled:
            cache_config.setdefault("redis", {})["enabled"] = True
            cache_config.setdefault("redis", {})["url"] = self.settings.redis_url
        
        return CacheConfig(**cache_config)
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for the specified provider.
        
        Args:
            provider: Name of the LLM provider
            
        Returns:
            API key if available, None otherwise
        """
        if provider.lower() == "gemini":
            return self.settings.gemini_api_key
        elif provider.lower() == "groq":
            return self.settings.groq_api_key
        return None
    
    def validate_configuration(self) -> List[str]:
        """Validate the current configuration.
        
        Returns:
            List of validation errors, empty if configuration is valid
        """
        errors = []
        
        # Check for required API keys
        default_provider = self.llm.default_provider
        if not self.get_api_key(default_provider):
            errors.append(f"Missing API key for default provider: {default_provider}")
        
        # Check file paths
        upload_dir = Path(self.settings.upload_dir)
        temp_dir = Path(self.settings.temp_dir)
        
        try:
            upload_dir.mkdir(parents=True, exist_ok=True)
            temp_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create directories: {e}")
        
        # Check log file directory
        log_file = Path(self.settings.log_file)
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create log directory: {e}")
        
        return errors
    
    def get_yaml_section(self, section: str) -> Dict[str, Any]:
        """Get a specific section from the YAML configuration.
        
        Args:
            section: Name of the configuration section
            
        Returns:
            Configuration section as dictionary
        """
        return self._yaml_config.get(section, {})


# Global configuration instance
config = ConfigManager()


def get_config() -> ConfigManager:
    """Get the global configuration instance."""
    return config


def reload_config(config_path: Optional[str] = None) -> ConfigManager:
    """Reload configuration from files.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        New configuration manager instance
    """
    global config
    config = ConfigManager(config_path)
    return config