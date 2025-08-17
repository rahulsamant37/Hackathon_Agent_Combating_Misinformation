"""
Custom logging configuration for the misinformation detection tool.

This module provides structured logging with support for different output formats,
log rotation, and correlation IDs for request tracking.
"""

import logging
import logging.handlers
import sys
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from contextvars import ContextVar

import structlog
from rich.console import Console
from rich.logging import RichHandler

from config.settings import get_config


# Context variable for correlation ID tracking
correlation_id: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)


class CorrelationIDProcessor:
    """Structlog processor to add correlation IDs to log records."""
    
    def __call__(self, logger: Any, method_name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Add correlation ID to the event dictionary."""
        cid = correlation_id.get()
        if cid:
            event_dict['correlation_id'] = cid
        return event_dict


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add correlation ID if available
        cid = correlation_id.get()
        if cid:
            log_entry['correlation_id'] = cid
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields from the record
        for key, value in record.__dict__.items():
            if key not in ('name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created',
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'getMessage', 'exc_info',
                          'exc_text', 'stack_info'):
                log_entry[key] = value
        
        return json.dumps(log_entry, default=str)


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        color = self.COLORS.get(record.levelname, '')
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        
        # Add correlation ID to message if available
        cid = correlation_id.get()
        if cid:
            record.msg = f"[{cid[:8]}] {record.msg}"
        
        return super().format(record)


class LoggerManager:
    """Manages logger configuration and setup."""
    
    def __init__(self):
        """Initialize the logger manager."""
        self.config = get_config()
        self.console = Console()
        self._loggers: Dict[str, logging.Logger] = {}
        self._setup_structlog()
    
    def _setup_structlog(self) -> None:
        """Configure structlog for structured logging."""
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                CorrelationIDProcessor(),
                structlog.processors.TimeStamper(fmt="ISO"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
    
    def _create_file_handler(self, log_file: str, level: str = "INFO") -> logging.Handler:
        """Create a file handler with rotation."""
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use rotating file handler
        handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        
        handler.setLevel(getattr(logging, level.upper()))
        handler.setFormatter(JSONFormatter())
        
        return handler
    
    def _create_console_handler(self, level: str = "INFO", use_rich: bool = True) -> logging.Handler:
        """Create a console handler."""
        if use_rich:
            handler = RichHandler(
                console=self.console,
                show_time=True,
                show_path=True,
                markup=True,
                rich_tracebacks=True
            )
        else:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(ColoredFormatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
        
        handler.setLevel(getattr(logging, level.upper()))
        return handler
    
    def get_logger(self, name: str, level: Optional[str] = None) -> logging.Logger:
        """Get or create a logger with the specified name.
        
        Args:
            name: Logger name
            level: Log level (optional, uses config default)
            
        Returns:
            Configured logger instance
        """
        if name in self._loggers:
            return self._loggers[name]
        
        logger = logging.getLogger(name)
        
        # Set log level
        log_level = level or self.config.settings.log_level
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Add console handler
        console_handler = self._create_console_handler(log_level)
        logger.addHandler(console_handler)
        
        # Add file handler if log file is configured
        if self.config.settings.log_file:
            file_handler = self._create_file_handler(
                self.config.settings.log_file, 
                log_level
            )
            logger.addHandler(file_handler)
        
        # Prevent propagation to root logger
        logger.propagate = False
        
        self._loggers[name] = logger
        return logger
    
    def get_structured_logger(self, name: str) -> structlog.BoundLogger:
        """Get a structured logger using structlog.
        
        Args:
            name: Logger name
            
        Returns:
            Structured logger instance
        """
        # Ensure the underlying logger is configured
        self.get_logger(name)
        return structlog.get_logger(name)
    
    def set_correlation_id(self, cid: Optional[str] = None) -> str:
        """Set correlation ID for request tracking.
        
        Args:
            cid: Correlation ID (generates new one if None)
            
        Returns:
            The correlation ID that was set
        """
        if cid is None:
            cid = str(uuid.uuid4())
        
        correlation_id.set(cid)
        return cid
    
    def clear_correlation_id(self) -> None:
        """Clear the current correlation ID."""
        correlation_id.set(None)
    
    def get_correlation_id(self) -> Optional[str]:
        """Get the current correlation ID."""
        return correlation_id.get()


# Global logger manager instance
_logger_manager = LoggerManager()


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """Get a configured logger instance.
    
    Args:
        name: Logger name (typically __name__)
        level: Optional log level override
        
    Returns:
        Configured logger instance
    """
    return _logger_manager.get_logger(name, level)


def get_structured_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger instance.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Structured logger instance
    """
    return _logger_manager.get_structured_logger(name)


def set_correlation_id(cid: Optional[str] = None) -> str:
    """Set correlation ID for request tracking.
    
    Args:
        cid: Correlation ID (generates new one if None)
        
    Returns:
        The correlation ID that was set
    """
    return _logger_manager.set_correlation_id(cid)


def clear_correlation_id() -> None:
    """Clear the current correlation ID."""
    _logger_manager.clear_correlation_id()


def get_correlation_id() -> Optional[str]:
    """Get the current correlation ID."""
    return _logger_manager.get_correlation_id()


# Convenience function for getting the main application logger
def get_app_logger() -> logging.Logger:
    """Get the main application logger."""
    return get_logger("misinformation_tool")


# Create some common loggers
api_logger = get_logger("misinformation_tool.api")
llm_logger = get_logger("misinformation_tool.llm")
analysis_logger = get_logger("misinformation_tool.analysis")
streamlit_logger = get_logger("misinformation_tool.streamlit")