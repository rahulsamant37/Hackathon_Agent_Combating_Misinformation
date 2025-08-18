"""
Google Gemini LLM provider implementation using langchain-google-genai.
"""

import os
import asyncio
import json
from typing import Dict, Any, List, Optional
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import PydanticOutputParser
from pydantic import ValidationError

from .base_provider import BaseLLMProvider, LLMResponse, LLMProviderConfig
from utils.logger import get_logger
from utils.exceptions import LLMProviderError

logger = get_logger(__name__)


class GeminiProvider(BaseLLMProvider):
    """Google Gemini provider implementation."""
    
    def __init__(self, config: LLMProviderConfig):
        super().__init__(config)
        self.api_key = config.api_key or os.getenv("GEMINI_API_KEY")
        
        if not self.api_key:
            raise LLMProviderError("Gemini API key not provided")
        
        self.llm = ChatGoogleGenerativeAI(
            model=config.model,
            google_api_key=self.api_key,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout=config.timeout
        )
    
    async def generate_response(
        self, 
        messages: List[BaseMessage],
        **kwargs
    ) -> LLMResponse:
        """Generate a response from Gemini."""
        try:
            logger.info(f"Generating response with Gemini model: {self.config.model}")
            
            # Convert messages to format expected by Gemini
            formatted_messages = self._format_messages(messages)
            
            # Generate response with retry logic
            response = await self._generate_with_retry(formatted_messages, **kwargs)
            
            return LLMResponse(
                content=response.content,
                provider=self.provider_name,
                model=self.config.model,
                tokens_used=getattr(response, 'usage', {}).get('total_tokens'),
                metadata={
                    "response_metadata": getattr(response, 'response_metadata', {}),
                    **kwargs
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating response with Gemini: {str(e)}")
            raise LLMProviderError(f"Gemini generation failed: {str(e)}")
    
    async def generate_structured_response(
        self,
        messages: List[BaseMessage], 
        response_schema: Dict[str, Any],
        **kwargs
    ) -> LLMResponse:
        """Generate a structured response with schema validation."""
        try:
            logger.info(f"Generating structured response with Gemini")
            
            # Add schema instruction to the last message
            schema_instruction = f"\n\nPlease respond with a valid JSON object that matches this schema: {json.dumps(response_schema, indent=2)}"
            
            if messages and isinstance(messages[-1], HumanMessage):
                messages[-1].content += schema_instruction
            else:
                messages.append(HumanMessage(content=schema_instruction))
            
            response = await self.generate_response(messages, **kwargs)
            
            # Validate response against schema
            try:
                parsed_content = json.loads(response.content)
                # Basic schema validation could be enhanced with jsonschema
                response.metadata["parsed_content"] = parsed_content
                response.metadata["schema_valid"] = True
            except (json.JSONDecodeError, ValidationError) as e:
                logger.warning(f"Response doesn't match expected schema: {str(e)}")
                response.metadata["schema_valid"] = False
                response.metadata["schema_error"] = str(e)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating structured response with Gemini: {str(e)}")
            raise LLMProviderError(f"Gemini structured generation failed: {str(e)}")
    
    def is_available(self) -> bool:
        """Check if Gemini provider is available."""
        return bool(self.api_key)
    
    async def health_check(self) -> bool:
        """Perform a health check on Gemini."""
        try:
            test_message = [HumanMessage(content="Hello, respond with 'OK' if you can hear me.")]
            response = await self.generate_response(test_message)
            return "OK" in response.content.upper()
        except Exception as e:
            logger.error(f"Gemini health check failed: {str(e)}")
            return False
    
    def _format_messages(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """Format messages for Gemini API."""
        formatted = []
        for message in messages:
            if isinstance(message, (HumanMessage, SystemMessage)):
                formatted.append(message)
            else:
                # Convert other message types to HumanMessage
                formatted.append(HumanMessage(content=str(message.content)))
        return formatted
    
    async def _generate_with_retry(self, messages: List[BaseMessage], **kwargs):
        """Generate response with retry logic."""
        last_exception = None
        
        for attempt in range(self.config.retry_attempts):
            try:
                if attempt > 0:
                    await asyncio.sleep(self.config.retry_delay * attempt)
                
                response = await self.llm.ainvoke(messages)
                return response
                
            except Exception as e:
                last_exception = e
                logger.warning(f"Gemini attempt {attempt + 1} failed: {str(e)}")
                
                if attempt == self.config.retry_attempts - 1:
                    break
        
        raise LLMProviderError(f"Gemini failed after {self.config.retry_attempts} attempts: {str(last_exception)}")