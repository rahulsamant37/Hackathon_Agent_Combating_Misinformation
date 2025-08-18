"""
FastAPI main application for the misinformation detection tool.

This module sets up the FastAPI application with middleware, routes,
and error handling for the misinformation detection backend.
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware

from config.settings import get_config
from utils.logger import get_logger
from utils.exceptions import MisinformationToolException
from api.routes import health, analysis


# Initialize logger
logger = get_logger(__name__)
config = get_config()


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging HTTP requests and responses."""
    
    async def dispatch(self, request: Request, call_next):
        """Process request and log details."""
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Log request
        start_time = time.time()
        logger.info(
            f"Request started",
            extra={
                "request_id": request_id,
                "method": request.method,
                "url": str(request.url),
                "client_ip": request.client.host if request.client else None,
                "user_agent": request.headers.get("user-agent"),
            }
        )
        
        try:
            response = await call_next(request)
            
            # Log response
            process_time = time.time() - start_time
            logger.info(
                f"Request completed",
                extra={
                    "request_id": request_id,
                    "status_code": response.status_code,
                    "process_time": round(process_time, 4),
                }
            )
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                f"Request failed",
                extra={
                    "request_id": request_id,
                    "error": str(e),
                    "process_time": round(process_time, 4),
                },
                exc_info=True
            )
            raise


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for handling application errors."""
    
    async def dispatch(self, request: Request, call_next):
        """Process request with error handling."""
        try:
            return await call_next(request)
        except MisinformationToolException as e:
            logger.error(f"Application error: {e}", exc_info=True)
            return JSONResponse(
                status_code=400,
                content={
                    "error_code": e.__class__.__name__,
                    "message": str(e),
                    "request_id": getattr(request.state, "request_id", None),
                    "timestamp": time.time()
                }
            )
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={
                    "error_code": "InternalServerError",
                    "message": "An unexpected error occurred",
                    "request_id": getattr(request.state, "request_id", None),
                    "timestamp": time.time()
                }
            )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting misinformation detection API")
    
    # Validate configuration
    validation_errors = config.validate_configuration()
    if validation_errors:
        logger.error(f"Configuration validation failed: {validation_errors}")
        raise RuntimeError(f"Configuration errors: {', '.join(validation_errors)}")
    
    logger.info("Configuration validated successfully")
    
    # Initialize any required services here
    # (e.g., database connections, cache, etc.)
    
    yield
    
    # Shutdown
    logger.info("Shutting down misinformation detection API")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title="Misinformation Detection API",
        description="AI-powered tool for detecting misinformation in text, images, and URLs",
        version="1.0.0",
        docs_url="/docs" if config.settings.debug else None,
        redoc_url="/redoc" if config.settings.debug else None,
        lifespan=lifespan
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )
    
    # Add trusted host middleware for security
    if not config.settings.debug:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["localhost", "127.0.0.1", config.settings.api_host]
        )
    
    # Add custom middleware
    app.add_middleware(ErrorHandlingMiddleware)
    app.add_middleware(RequestLoggingMiddleware)
    
    # Exception handlers
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle request validation errors."""
        logger.warning(f"Validation error: {exc.errors()}")
        return JSONResponse(
            status_code=422,
            content={
                "error_code": "ValidationError",
                "message": "Request validation failed",
                "details": exc.errors(),
                "request_id": getattr(request.state, "request_id", None),
                "timestamp": time.time()
            }
        )
    
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        """Handle HTTP exceptions."""
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error_code": "HTTPException",
                "message": exc.detail,
                "request_id": getattr(request.state, "request_id", None),
                "timestamp": time.time()
            }
        )
    
    # Include routers
    app.include_router(health.router, prefix="/api/v1", tags=["health"])
    app.include_router(analysis.router, prefix="/api/v1", tags=["analysis"])
    
    return app


# Create the application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api.main:app",
        host=config.settings.api_host,
        port=config.settings.api_port,
        reload=config.settings.debug,
        log_level=config.settings.log_level.lower()
    )