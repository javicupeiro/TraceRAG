"""
Configuration manager for loading and managing application settings.

This module handles loading configuration from YAML files and environment variables.
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Get a logger for the current module
logger = logging.getLogger(__name__)

class ConfigManager:
    """
    Manages application configuration from YAML files and environment variables.
    """
    
    def __init__(self, config_env: str = "dev"):
        """
        Initialize the configuration manager.
        
        Args:
            config_env (str): Environment name (dev, test, production)
        """
        self.config_env = config_env
        self.config_data = {}
        self._load_config()
        
    def _get_config_path(self) -> Path:
        """
        Get the path to the configuration file based on environment.
        
        Returns:
            Path: Path to the configuration file
        """
        # Get the project root directory (assuming this file is in src/core/)
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent  # Go up to project root
        
        config_path = project_root / "config" / self.config_env / "config.yaml"
        
        if not config_path.exists():
            logger.warning(f"Config file not found at {config_path}, using defaults")
            return None
            
        return config_path
    
    def _load_config(self):
        """
        Load configuration from YAML file.
        """
        config_path = self._get_config_path()
        
        if config_path and config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.config_data = yaml.safe_load(f) or {}
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.error(f"Failed to load config from {config_path}: {e}")
                self.config_data = {}
        else:
            logger.warning("No config file found, using default values")
            self.config_data = {}
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Args:
            key_path (str): Dot-separated path to the config value (e.g., 'LLM.GROQ.MODEL')
            default (Any): Default value if key is not found
            
        Returns:
            Any: Configuration value or default
        """
        keys = key_path.split('.')
        value = self.config_data
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            logger.debug(f"Config key '{key_path}' not found, using default: {default}")
            return default
    
    def get_llm_config(self, provider: str) -> Dict[str, Any]:
        """
        Get LLM configuration for a specific provider.
        
        Args:
            provider (str): Provider name ('GROQ' or 'GEMINI')
            
        Returns:
            Dict[str, Any]: Provider configuration
        """
        provider_upper = provider.upper()
        
        config = {
            'model': self.get(f'LLM.{provider_upper}.MODEL'),
            'rate_limit': self.get(f'LLM.{provider_upper}.RATE_LIMIT'),
            'max_output_tokens': self.get(f'LLM.{provider_upper}.MAX_OUTPUT_TOKENS'),
            'temperature': self.get(f'LLM.{provider_upper}.TEMPERATURE')
        }
        
        # Filter out None values
        return {k: v for k, v in config.items() if v is not None}
    
    def get_embedding_config(self) -> Dict[str, Any]:
        """
        Get embedding configuration.
        
        Returns:
            Dict[str, Any]: Embedding configuration
        """
        return {
            'name': self.get('EMBEDDING.NAME'),
            'dimensions': self.get('EMBEDDING.DIMENSIONS')
        }
    
    def get_database_config(self) -> Dict[str, Any]:
        """
        Get database configuration.
        
        Returns:
            Dict[str, Any]: Database configuration
        """
        return {
            'sql_path': self.get('DATABASE.SQL.PATH'),
            'milvus_host': self.get('DATABASE.MILVUS.HOST'),
            'milvus_port': self.get('DATABASE.MILVUS.PORT'),
            'collection_name': self.get('DATABASE.MILVUS.COLLECTION_NAME')
        }
    
    def get_env_or_config(self, env_key: str, config_key: str, default: Any = None) -> Any:
        """
        Get value from environment variable first, then config file, then default.
        
        Args:
            env_key (str): Environment variable key
            config_key (str): Config file key path
            default (Any): Default value
            
        Returns:
            Any: Value from env, config, or default
        """
        # First try environment variable
        env_value = os.getenv(env_key)
        if env_value is not None:
            return env_value
        
        # Then try config file
        config_value = self.get(config_key)
        if config_value is not None:
            return config_value
        
        # Finally use default
        return default
    
    def reload_config(self):
        """
        Reload configuration from file.
        """
        self._load_config()
        logger.info("Configuration reloaded")


# Global config manager instance
_config_manager = None

def get_config_manager(config_env: Optional[str] = None) -> ConfigManager:
    """
    Get the global configuration manager instance.
    
    Args:
        config_env (Optional[str]): Environment name, uses existing or defaults to 'dev'
        
    Returns:
        ConfigManager: Global configuration manager instance
    """
    global _config_manager
    
    if _config_manager is None or (config_env and config_env != _config_manager.config_env):
        env = config_env or os.getenv('CONFIG_ENV', 'dev')
        _config_manager = ConfigManager(env)
        logger.info(f"Initialized global config manager for environment: {env}")
    
    return _config_manager

def get_llm_config(provider: str) -> Dict[str, Any]:
    """
    Convenience function to get LLM configuration.
    
    Args:
        provider (str): Provider name
        
    Returns:
        Dict[str, Any]: LLM configuration
    """
    return get_config_manager().get_llm_config(provider)

def get_embedding_config() -> Dict[str, Any]:
    """
    Convenience function to get embedding configuration.
    
    Returns:
        Dict[str, Any]: Embedding configuration
    """
    return get_config_manager().get_embedding_config()

def get_database_config() -> Dict[str, Any]:
    """
    Convenience function to get database configuration.
    
    Returns:
        Dict[str, Any]: Database configuration
    """
    return get_config_manager().get_database_config()