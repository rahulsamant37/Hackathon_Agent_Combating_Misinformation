"""
LLM Router for managing multiple LLM providers with fallback logic.
"""

import asyncio
from typing import Dict, Any, List, Optional, Union
from enum import Enum
from langchain.schema import BaseMessage

from .providers.base_provider import BaseLLMProvider, LLMResponse, LLMProviderConfig
from .providers.gemini_provider import GeminiProvider
from .providers.groq_provider import GroqProvider
from config.settings import get_config
from utils.logger import get_logger
from utils.exceptions import LLMProviderError

logger = get_logger(__name__)


class ProviderType(Enum):
    """Supported LLM provider types."""
    GEMINI = "gemini"
    GROQ = "groq"


class LLMRouter:
    """
    Router for managing multiple LLM providers with automatic fallback.
    
    Handles provider selection, health monitoring, and graceful degradation
    when primary providers fail.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_config()
        self.providers: Dict[str, BaseLLMProvider] = {}
        self.provider_health: Dict[str, bool] = {}
        
        # Initialize providers based on configuration
        self._initialize_providers()
        
        # Set default and fallback providers
        self.default_provider = self.config.get("llm", {}).get("default_provider", "gemini")
        self.fallback_provider = self.config.get("llm", {}).get("fallback_provider", "groq")
        
        logger.info(f"LLM Router initialized with providers: {list(self.providers.keys())}")
        logger.info(f"Default provider: {self.default_provider}, Fallback: {self.fallback_provider}")
    
    def _initialize_providers(self):
        """Initialize all configured LLM providers."""
        llm_config = self.config.get("llm", {}).get("providers", {})
        
        for provider_name, provider_config in llm_config.items():
            try:
                config = LLMProviderConfig(**provider_config)
                
                if provider_name == "gemini":
                    provider = GeminiProvider(config)
                elif provider_name == "groq":
                    provider = GroqProvider(config)
                else:
                    logger.warning(f"Unknown provider type: {provider_name}")
                    continue
                
                if provider.is_available():
                    self.providers[provider_name] = provider
                    self.provider_health[provider_name] = True
                    logger.info(f"Initialized {provider_name} provider")
                else:
                    logger.warning(f"Provider {provider_name} is not available (missing API key)")
                    
            except Exception as e:
                logger.error(f"Failed to initialize {provider_name} provider: {str(e)}")
    
    async def generate_response(
        self,
        messages: List[BaseMessage],
        provider: Optional[str] = None,
        use_fallback: bool = True,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a response using the specified or default provider.
        
        Args:
            messages: List of messages to send to the LLM
            provider: Specific provider to use (optional)
            use_fallback: Whether to use fallback provider if primary fails
            **kwargs: Additional arguments to pass to the provider
            
        Returns:
            LLMResponse: The generated response
            
        Raises:
            LLMProviderError: If all providers fail
        """
        # Determine which provider to use
        target_provider = provider or self.default_provider
        
        # Try primary provider
        if target_provider in self.providers:
            try:
                response = await self._generate_with_provider(
                    target_provider, messages, **kwargs
                )
                self.provider_health[target_provider] = True
                return response
                
            except Exception as e:
                logger.warning(f"Primary provider {target_provider} failed: {str(e)}")
                self.provider_health[target_provider] = False
                
                if not use_fallback:
                    raise LLMProviderError(f"Provider {target_provider} failed: {str(e)}")
        
        # Try fallback provider if enabled and different from primary
        if use_fallback and self.fallback_provider != target_provider:
            if self.fallback_provider in self.providers:
                try:
                    logger.info(f"Falling back to {self.fallback_provider} provider")
                    response = await self._generate_with_provider(
                        self.fallback_provider, messages, **kwargs
                    )
                    self.provider_health[self.fallback_provider] = True
                    return response
                    
                except Exception as e:
                    logger.error(f"Fallback provider {self.fallback_provider} failed: {str(e)}")
                    self.provider_health[self.fallback_provider] = False
        
        # Try any remaining healthy providers
        for provider_name, provider in self.providers.items():
            if (provider_name != target_provider and 
                provider_name != self.fallback_provider and
                self.provider_health.get(provider_name, True)):
                
                try:
                    logger.info(f"Trying alternative provider: {provider_name}")
                    response = await self._generate_with_provider(
                        provider_name, messages, **kwargs
                    )
                    self.provider_health[provider_name] = True
                    return response
                    
                except Exception as e:
                    logger.warning(f"Alternative provider {provider_name} failed: {str(e)}")
                    self.provider_health[provider_name] = False
        
        raise LLMProviderError("All LLM providers failed")
    
    async def generate_structured_response(
        self,
        messages: List[BaseMessage],
        response_schema: Dict[str, Any],
        provider: Optional[str] = None,
        use_fallback: bool = True,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a structured response with schema validation.
        
        Args:
            messages: List of messages to send to the LLM
            response_schema: Expected response schema
            provider: Specific provider to use (optional)
            use_fallback: Whether to use fallback provider if primary fails
            **kwargs: Additional arguments to pass to the provider
            
        Returns:
            LLMResponse: The generated structured response
        """
        target_provider = provider or self.default_provider
        
        # Try primary provider
        if target_provider in self.providers:
            try:
                response = await self.providers[target_provider].generate_structured_response(
                    messages, response_schema, **kwargs
                )
                self.provider_health[target_provider] = True
                return response
                
            except Exception as e:
                logger.warning(f"Primary provider {target_provider} failed for structured response: {str(e)}")
                self.provider_health[target_provider] = False
                
                if not use_fallback:
                    raise LLMProviderError(f"Provider {target_provider} failed: {str(e)}")
        
        # Try fallback provider
        if use_fallback and self.fallback_provider != target_provider:
            if self.fallback_provider in self.providers:
                try:
                    logger.info(f"Falling back to {self.fallback_provider} for structured response")
                    response = await self.providers[self.fallback_provider].generate_structured_response(
                        messages, response_schema, **kwargs
                    )
                    self.provider_health[self.fallback_provider] = True
                    return response
                    
                except Exception as e:
                    logger.error(f"Fallback provider {self.fallback_provider} failed: {str(e)}")
                    self.provider_health[self.fallback_provider] = False
        
        raise LLMProviderError("All LLM providers failed for structured response")
    
    async def _generate_with_provider(
        self, 
        provider_name: str, 
        messages: List[BaseMessage], 
        **kwargs
    ) -> LLMResponse:
        """Generate response with a specific provider."""
        provider = self.providers[provider_name]
        return await provider.generate_response(messages, **kwargs)
    
    async def health_check_all(self) -> Dict[str, bool]:
        """Perform health checks on all providers."""
        health_results = {}
        
        for provider_name, provider in self.providers.items():
            try:
                is_healthy = await provider.health_check()
                health_results[provider_name] = is_healthy
                self.provider_health[provider_name] = is_healthy
                
            except Exception as e:
                logger.error(f"Health check failed for {provider_name}: {str(e)}")
                health_results[provider_name] = False
                self.provider_health[provider_name] = False
        
        return health_results
    
    def get_provider_status(self) -> Dict[str, Any]:
        """Get status information for all providers."""
        status = {
            "providers": {},
            "default_provider": self.default_provider,
            "fallback_provider": self.fallback_provider,
            "healthy_providers": []
        }
        
        for provider_name, provider in self.providers.items():
            is_healthy = self.provider_health.get(provider_name, False)
            status["providers"][provider_name] = {
                "available": provider.is_available(),
                "healthy": is_healthy,
                "info": provider.get_provider_info()
            }
            
            if is_healthy:
                status["healthy_providers"].append(provider_name)
        
        return status
    
    def get_available_providers(self) -> List[str]:
        """Get list of available provider names."""
        return list(self.providers.keys())
    
    def set_default_provider(self, provider_name: str):
        """Set the default provider."""
        if provider_name in self.providers:
            self.default_provider = provider_name
            logger.info(f"Default provider set to: {provider_name}")
        else:
            raise ValueError(f"Provider {provider_name} not available")
    
    def set_fallback_provider(self, provider_name: str):
        """Set the fallback provider."""
        if provider_name in self.providers:
            self.fallback_provider = provider_name
            logger.info(f"Fallback provider set to: {provider_name}")
        else:
            raise ValueError(f"Provider {provider_name} not available")