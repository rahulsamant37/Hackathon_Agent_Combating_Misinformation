"""
Base LLM provider interface for abstracting different AI services.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
from langchain.schema import BaseMessage


class LLMResponse(BaseModel):
    """Standardized response from LLM providers."""
    content: str
    provider: str
    model: str
    tokens_used: Optional[int] = None
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = {}


class LLMProviderConfig(BaseModel):
    """Configuration for LLM providers."""
    model: str
    max_tokens: int = 4096
    temperature: float = 0.1
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: int = 1
    api_key: Optional[str] = None


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, config: LLMProviderConfig):
        self.config = config
        self.provider_name = self.__class__.__name__.replace("Provider", "").lower()
    
    @abstractmethod
    async def generate_response(
        self, 
        messages: List[BaseMessage],
        **kwargs
    ) -> LLMResponse:
        """Generate a response from the LLM provider."""
        pass
    
    @abstractmethod
    async def generate_structured_response(
        self,
        messages: List[BaseMessage], 
        response_schema: Dict[str, Any],
        **kwargs
    ) -> LLMResponse:
        """Generate a structured response with schema validation."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available and configured."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Perform a health check on the provider."""
        pass
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the provider."""
        return {
            "name": self.provider_name,
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature
        }