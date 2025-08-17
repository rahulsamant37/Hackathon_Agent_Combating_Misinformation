"""
Custom exception classes for the misinformation detection tool.

This module defines a hierarchy of exceptions used throughout the application
for proper error handling and user feedback.
"""

from typing import Any, Dict, Optional
from datetime import datetime


class MisinformationToolException(Exception):
    """Base exception for the misinformation detection tool.
    
    All custom exceptions in the application should inherit from this class.
    """
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        """Initialize the exception.
        
        Args:
            message: Human-readable error message
            error_code: Machine-readable error code
            details: Additional error details
            cause: The underlying exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        self.cause = cause
        self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses.
        
        Returns:
            Dictionary representation of the exception
        """
        result = {
            "error_code": self.error_code,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
        }
        
        if self.details:
            result["details"] = self.details
        
        if self.cause:
            result["cause"] = str(self.cause)
        
        return result


class ValidationError(MisinformationToolException):
    """Raised when input validation fails."""
    
    def __init__(
        self, 
        message: str, 
        field: Optional[str] = None,
        value: Optional[Any] = None,
        **kwargs
    ):
        """Initialize validation error.
        
        Args:
            message: Error message
            field: Name of the field that failed validation
            value: The invalid value
            **kwargs: Additional arguments for base class
        """
        details = kwargs.get("details", {})
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = str(value)
        
        kwargs["details"] = details
        super().__init__(message, **kwargs)
        self.field = field
        self.value = value


class ContentProcessingError(MisinformationToolException):
    """Raised when content processing fails."""
    
    def __init__(
        self, 
        message: str, 
        content_type: Optional[str] = None,
        processing_stage: Optional[str] = None,
        **kwargs
    ):
        """Initialize content processing error.
        
        Args:
            message: Error message
            content_type: Type of content being processed
            processing_stage: Stage where processing failed
            **kwargs: Additional arguments for base class
        """
        details = kwargs.get("details", {})
        if content_type:
            details["content_type"] = content_type
        if processing_stage:
            details["processing_stage"] = processing_stage
        
        kwargs["details"] = details
        super().__init__(message, **kwargs)
        self.content_type = content_type
        self.processing_stage = processing_stage


class LLMProviderError(MisinformationToolException):
    """Raised when LLM provider calls fail."""
    
    def __init__(
        self, 
        message: str, 
        provider: Optional[str] = None,
        model: Optional[str] = None,
        status_code: Optional[int] = None,
        **kwargs
    ):
        """Initialize LLM provider error.
        
        Args:
            message: Error message
            provider: Name of the LLM provider
            model: Model name that failed
            status_code: HTTP status code if applicable
            **kwargs: Additional arguments for base class
        """
        details = kwargs.get("details", {})
        if provider:
            details["provider"] = provider
        if model:
            details["model"] = model
        if status_code:
            details["status_code"] = status_code
        
        kwargs["details"] = details
        super().__init__(message, **kwargs)
        self.provider = provider
        self.model = model
        self.status_code = status_code


class AnalysisError(MisinformationToolException):
    """Raised when content analysis fails."""
    
    def __init__(
        self, 
        message: str, 
        analysis_type: Optional[str] = None,
        analysis_id: Optional[str] = None,
        **kwargs
    ):
        """Initialize analysis error.
        
        Args:
            message: Error message
            analysis_type: Type of analysis that failed
            analysis_id: ID of the failed analysis
            **kwargs: Additional arguments for base class
        """
        details = kwargs.get("details", {})
        if analysis_type:
            details["analysis_type"] = analysis_type
        if analysis_id:
            details["analysis_id"] = analysis_id
        
        kwargs["details"] = details
        super().__init__(message, **kwargs)
        self.analysis_type = analysis_type
        self.analysis_id = analysis_id


class ConfigurationError(MisinformationToolException):
    """Raised when configuration is invalid or missing."""
    
    def __init__(
        self, 
        message: str, 
        config_key: Optional[str] = None,
        config_file: Optional[str] = None,
        **kwargs
    ):
        """Initialize configuration error.
        
        Args:
            message: Error message
            config_key: Configuration key that is invalid
            config_file: Configuration file with the error
            **kwargs: Additional arguments for base class
        """
        details = kwargs.get("details", {})
        if config_key:
            details["config_key"] = config_key
        if config_file:
            details["config_file"] = config_file
        
        kwargs["details"] = details
        super().__init__(message, **kwargs)
        self.config_key = config_key
        self.config_file = config_file


class AuthenticationError(MisinformationToolException):
    """Raised when authentication fails."""
    
    def __init__(
        self, 
        message: str = "Authentication failed", 
        auth_type: Optional[str] = None,
        **kwargs
    ):
        """Initialize authentication error.
        
        Args:
            message: Error message
            auth_type: Type of authentication that failed
            **kwargs: Additional arguments for base class
        """
        details = kwargs.get("details", {})
        if auth_type:
            details["auth_type"] = auth_type
        
        kwargs["details"] = details
        super().__init__(message, **kwargs)
        self.auth_type = auth_type


class AuthorizationError(MisinformationToolException):
    """Raised when authorization fails."""
    
    def __init__(
        self, 
        message: str = "Access denied", 
        required_permission: Optional[str] = None,
        **kwargs
    ):
        """Initialize authorization error.
        
        Args:
            message: Error message
            required_permission: Permission that was required
            **kwargs: Additional arguments for base class
        """
        details = kwargs.get("details", {})
        if required_permission:
            details["required_permission"] = required_permission
        
        kwargs["details"] = details
        super().__init__(message, **kwargs)
        self.required_permission = required_permission


class RateLimitError(MisinformationToolException):
    """Raised when rate limits are exceeded."""
    
    def __init__(
        self, 
        message: str = "Rate limit exceeded", 
        limit: Optional[int] = None,
        reset_time: Optional[datetime] = None,
        **kwargs
    ):
        """Initialize rate limit error.
        
        Args:
            message: Error message
            limit: The rate limit that was exceeded
            reset_time: When the rate limit resets
            **kwargs: Additional arguments for base class
        """
        details = kwargs.get("details", {})
        if limit:
            details["limit"] = limit
        if reset_time:
            details["reset_time"] = reset_time.isoformat()
        
        kwargs["details"] = details
        super().__init__(message, **kwargs)
        self.limit = limit
        self.reset_time = reset_time


class TimeoutError(MisinformationToolException):
    """Raised when operations timeout."""
    
    def __init__(
        self, 
        message: str = "Operation timed out", 
        timeout_seconds: Optional[float] = None,
        operation: Optional[str] = None,
        **kwargs
    ):
        """Initialize timeout error.
        
        Args:
            message: Error message
            timeout_seconds: The timeout that was exceeded
            operation: Name of the operation that timed out
            **kwargs: Additional arguments for base class
        """
        details = kwargs.get("details", {})
        if timeout_seconds:
            details["timeout_seconds"] = timeout_seconds
        if operation:
            details["operation"] = operation
        
        kwargs["details"] = details
        super().__init__(message, **kwargs)
        self.timeout_seconds = timeout_seconds
        self.operation = operation


class ExternalServiceError(MisinformationToolException):
    """Raised when external service calls fail."""
    
    def __init__(
        self, 
        message: str, 
        service_name: Optional[str] = None,
        status_code: Optional[int] = None,
        response_body: Optional[str] = None,
        **kwargs
    ):
        """Initialize external service error.
        
        Args:
            message: Error message
            service_name: Name of the external service
            status_code: HTTP status code
            response_body: Response body from the service
            **kwargs: Additional arguments for base class
        """
        details = kwargs.get("details", {})
        if service_name:
            details["service_name"] = service_name
        if status_code:
            details["status_code"] = status_code
        if response_body:
            details["response_body"] = response_body[:1000]  # Truncate long responses
        
        kwargs["details"] = details
        super().__init__(message, **kwargs)
        self.service_name = service_name
        self.status_code = status_code
        self.response_body = response_body


class FileProcessingError(ContentProcessingError):
    """Raised when file processing fails."""
    
    def __init__(
        self, 
        message: str, 
        filename: Optional[str] = None,
        file_size: Optional[int] = None,
        **kwargs
    ):
        """Initialize file processing error.
        
        Args:
            message: Error message
            filename: Name of the file being processed
            file_size: Size of the file in bytes
            **kwargs: Additional arguments for base class
        """
        details = kwargs.get("details", {})
        if filename:
            details["filename"] = filename
        if file_size:
            details["file_size"] = file_size
        
        kwargs["details"] = details
        super().__init__(message, **kwargs)
        self.filename = filename
        self.file_size = file_size


# Convenience functions for creating common exceptions

def validation_error(message: str, field: Optional[str] = None, value: Optional[Any] = None) -> ValidationError:
    """Create a validation error."""
    return ValidationError(message, field=field, value=value)


def content_processing_error(
    message: str, 
    content_type: Optional[str] = None, 
    stage: Optional[str] = None
) -> ContentProcessingError:
    """Create a content processing error."""
    return ContentProcessingError(message, content_type=content_type, processing_stage=stage)


def llm_provider_error(
    message: str, 
    provider: Optional[str] = None, 
    model: Optional[str] = None
) -> LLMProviderError:
    """Create an LLM provider error."""
    return LLMProviderError(message, provider=provider, model=model)


def analysis_error(
    message: str, 
    analysis_type: Optional[str] = None, 
    analysis_id: Optional[str] = None
) -> AnalysisError:
    """Create an analysis error."""
    return AnalysisError(message, analysis_type=analysis_type, analysis_id=analysis_id)


def configuration_error(
    message: str, 
    config_key: Optional[str] = None
) -> ConfigurationError:
    """Create a configuration error."""
    return ConfigurationError(message, config_key=config_key)