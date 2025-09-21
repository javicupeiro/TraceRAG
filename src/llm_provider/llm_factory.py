"""
Factory for creating and managing LLM providers.

This module provides utilities for easily creating and configuring
different LLM providers based on configuration files and environment variables.
"""

import os
import logging
from typing import Optional, Dict, Any
from .base_llm_provider import BaseLLMProvider
from .groq_provider import GroqProvider
from .gemini_provider import GeminiProvider

# Get a logger for the current module
logger = logging.getLogger(__name__)

class LLMProviderFactory:
    """
    Factory class for creating LLM providers with configuration management.
    """
    
    SUPPORTED_PROVIDERS = {
        'groq': GroqProvider,
        'gemini': GeminiProvider
    }
    
    # Fallback defaults if config is not available
    FALLBACK_DEFAULTS = {
        'groq': {
            'model': 'meta-llama/llama-4-scout-17b-16e-instruct',
            'rate_limit': 0.3,
            'temperature': 1.0,
            'max_output_tokens': 1024
        },
        'gemini': {
            'model': 'gemini-1.5-flash',
            'rate_limit': 0.3,
            'temperature': 1.0,
            'max_output_tokens': 1024
        }
    }
    
    @classmethod
    def _get_config_manager(cls):
        """
        Get config manager, with fallback if not available.
        """
        try:
            # Try absolute import first
            from src.core.config_manager import get_config_manager
            return get_config_manager()
        except ImportError:
            try:
                # Try relative import
                from core.config_manager import get_config_manager
                return get_config_manager()
            except ImportError:
                logger.debug("Config manager not available, using fallback defaults")
                return None

    @classmethod
    def _get_provider_config(cls, provider_name: str) -> Dict[str, Any]:
        """
        Get configuration for a provider from config file or fallback.
        
        Args:
            provider_name (str): Provider name ('groq' or 'gemini')
            
        Returns:
            Dict[str, Any]: Provider configuration
        """
        config_manager = cls._get_config_manager()
        
        if config_manager:
            try:
                config = config_manager.get_llm_config(provider_name)
                # Map config keys to expected parameter names
                mapped_config = {
                    'model_name': config.get('model'),
                    'rate_limit_delay': config.get('rate_limit'),
                    'temperature': config.get('temperature'),
                    'max_output_tokens': config.get('max_output_tokens')
                }
                # Filter out None values
                return {k: v for k, v in mapped_config.items() if v is not None}
            except Exception as e:
                logger.warning(f"Failed to load config for {provider_name}: {e}")
        
        # Use fallback defaults
        fallback = cls.FALLBACK_DEFAULTS.get(provider_name.lower(), {})
        return {
            'model_name': fallback.get('model'),
            'rate_limit_delay': fallback.get('rate_limit'),
            'temperature': fallback.get('temperature'),
            'max_output_tokens': fallback.get('max_output_tokens')
        }

    @classmethod
    def create_provider(
        cls,
        provider_name: str,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        rate_limit_delay: Optional[float] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        **kwargs
    ) -> BaseLLMProvider:
        """
        Create an LLM provider instance with configuration support.

        Args:
            provider_name (str): Name of the provider ('groq' or 'gemini')
            api_key (Optional[str]): API key, will try to get from env if not provided
            model_name (Optional[str]): Model name, uses config/default if not provided
            rate_limit_delay (Optional[float]): Rate limit delay, uses config/default if not provided
            temperature (Optional[float]): Temperature setting, uses config/default if not provided
            max_output_tokens (Optional[int]): Max output tokens, uses config/default if not provided
            **kwargs: Additional provider-specific arguments

        Returns:
            BaseLLMProvider: Configured provider instance

        Raises:
            ValueError: If provider is not supported or API key is missing
        """
        provider_name = provider_name.lower()
        
        if provider_name not in cls.SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unsupported provider: {provider_name}. "
                f"Supported providers: {list(cls.SUPPORTED_PROVIDERS.keys())}"
            )
        
        # Get configuration from config file
        config = cls._get_provider_config(provider_name)
        
        # Get API key from parameter or environment
        if api_key is None:
            env_key = f"{provider_name.upper()}_API_KEY"
            api_key = os.getenv(env_key)
            
            if not api_key:
                raise ValueError(
                    f"API key not provided. Please set {env_key} environment variable "
                    f"or pass api_key parameter."
                )
        
        # Use provided parameters or fall back to config/defaults
        final_model_name = model_name or config.get('model_name')
        final_rate_limit = rate_limit_delay or config.get('rate_limit_delay')
        final_temperature = temperature or config.get('temperature')
        final_max_tokens = max_output_tokens or config.get('max_output_tokens')
        
        # Store additional config for providers that support it
        provider_config = kwargs.copy()
        if final_temperature is not None:
            provider_config['temperature'] = final_temperature
        if final_max_tokens is not None:
            provider_config['max_output_tokens'] = final_max_tokens
        
        # Create provider instance
        provider_class = cls.SUPPORTED_PROVIDERS[provider_name]
        
        try:
            provider = provider_class(
                api_key=api_key,
                model_name=final_model_name,
                rate_limit_delay=final_rate_limit,
                **provider_config
            )
            
            logger.info(f"Successfully created {provider_name} provider with model {final_model_name}")
            return provider
            
        except Exception as e:
            logger.error(f"Failed to create {provider_name} provider: {e}")
            raise

    @classmethod
    def create_groq_provider(
        cls,
        api_key: Optional[str] = None,
        **kwargs
    ) -> GroqProvider:
        """
        Convenience method to create a Groq provider with config defaults.

        Args:
            api_key (Optional[str]): Groq API key
            **kwargs: Additional configuration parameters

        Returns:
            GroqProvider: Configured Groq provider
        """
        return cls.create_provider('groq', api_key=api_key, **kwargs)

    @classmethod
    def create_gemini_provider(
        cls,
        api_key: Optional[str] = None,
        **kwargs
    ) -> GeminiProvider:
        """
        Convenience method to create a Gemini provider with config defaults.

        Args:
            api_key (Optional[str]): Gemini API key
            **kwargs: Additional configuration parameters

        Returns:
            GeminiProvider: Configured Gemini provider
        """
        return cls.create_provider('gemini', api_key=api_key, **kwargs)

    @classmethod
    def get_available_providers(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get information about available providers including their configurations.

        Returns:
            Dict[str, Dict[str, Any]]: Provider information including supported modalities
        """
        providers_info = {}
        
        for name, provider_class in cls.SUPPORTED_PROVIDERS.items():
            config = cls._get_provider_config(name)
            
            # Create a temporary instance to get supported modalities
            try:
                temp_provider = provider_class("dummy_key", config.get('model_name', 'dummy_model'))
                supported_modalities = temp_provider._get_supported_modalities()
            except:
                supported_modalities = ["unknown"]
            
            providers_info[name] = {
                "class": provider_class.__name__,
                "config": config,
                "supported_modalities": supported_modalities,
                "env_key": f"{name.upper()}_API_KEY"
            }
        
        return providers_info

    @classmethod
    def check_environment(cls) -> Dict[str, bool]:
        """
        Check if environment variables are properly set for each provider.

        Returns:
            Dict[str, bool]: Status of each provider's environment setup
        """
        status = {}
        
        for provider_name in cls.SUPPORTED_PROVIDERS.keys():
            env_key = f"{provider_name.upper()}_API_KEY"
            api_key = os.getenv(env_key)
            status[provider_name] = bool(api_key)
        
        return status


def create_provider_from_config(config: Dict[str, Any]) -> BaseLLMProvider:
    """
    Create a provider from a configuration dictionary.

    Args:
        config (Dict[str, Any]): Configuration dictionary with provider settings

    Returns:
        BaseLLMProvider: Configured provider instance
    """
    provider_name = config.get('provider')
    if not provider_name:
        raise ValueError("Provider name not specified in config")
    
    return LLMProviderFactory.create_provider(
        provider_name=provider_name,
        api_key=config.get('api_key'),
        model_name=config.get('model_name'),
        rate_limit_delay=config.get('rate_limit_delay'),
        temperature=config.get('temperature'),
        max_output_tokens=config.get('max_output_tokens')
    )