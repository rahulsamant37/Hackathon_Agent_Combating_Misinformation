"""
Health check endpoints for the misinformation detection API.

This module provides health check and readiness endpoints for monitoring
and deployment purposes.
"""

import time
import psutil
from typing import Dict, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config.settings import get_config
from utils.logger import get_logger


logger = get_logger(__name__)
config = get_config()

router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: datetime
    version: str = "1.0.0"
    uptime: float
    request_id: str


class ReadinessResponse(BaseModel):
    """Readiness check response model."""
    status: str
    timestamp: datetime
    checks: Dict[str, Any]
    request_id: str


class SystemInfo(BaseModel):
    """System information response model."""
    status: str
    timestamp: datetime
    system: Dict[str, Any]
    request_id: str


# Track application start time
_start_time = time.time()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Basic health check endpoint.
    
    Returns basic application status and uptime information.
    This endpoint should always return 200 OK if the application is running.
    """
    try:
        uptime = time.time() - _start_time
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.utcnow(),
            uptime=uptime,
            request_id="health-check"
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")


@router.get("/ready", response_model=ReadinessResponse)
async def readiness_check():
    """
    Readiness check endpoint.
    
    Performs comprehensive checks to determine if the application
    is ready to serve requests. Checks configuration, dependencies,
    and external services.
    """
    checks = {}
    overall_status = "ready"
    
    try:
        # Check configuration
        validation_errors = config.validate_configuration()
        checks["configuration"] = {
            "status": "pass" if not validation_errors else "fail",
            "errors": validation_errors
        }
        if validation_errors:
            overall_status = "not_ready"
        
        # Check API keys
        gemini_key = config.get_api_key("gemini")
        groq_key = config.get_api_key("groq")
        
        checks["api_keys"] = {
            "status": "pass" if (gemini_key or groq_key) else "fail",
            "gemini_configured": bool(gemini_key),
            "groq_configured": bool(groq_key)
        }
        if not (gemini_key or groq_key):
            overall_status = "not_ready"
        
        # Check file system
        try:
            import tempfile
            import os
            
            # Test write access to upload directory
            upload_dir = config.settings.upload_dir
            os.makedirs(upload_dir, exist_ok=True)
            
            test_file = os.path.join(upload_dir, "test_write.tmp")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            
            checks["filesystem"] = {
                "status": "pass",
                "upload_dir_writable": True
            }
        except Exception as e:
            checks["filesystem"] = {
                "status": "fail",
                "error": str(e)
            }
            overall_status = "not_ready"
        
        # Check memory usage
        memory = psutil.virtual_memory()
        memory_usage_percent = memory.percent
        
        checks["memory"] = {
            "status": "pass" if memory_usage_percent < 90 else "warn",
            "usage_percent": memory_usage_percent,
            "available_gb": round(memory.available / (1024**3), 2)
        }
        
        # Check disk space
        disk = psutil.disk_usage('.')
        disk_usage_percent = (disk.used / disk.total) * 100
        
        checks["disk"] = {
            "status": "pass" if disk_usage_percent < 90 else "warn",
            "usage_percent": round(disk_usage_percent, 2),
            "free_gb": round(disk.free / (1024**3), 2)
        }
        
        return ReadinessResponse(
            status=overall_status,
            timestamp=datetime.utcnow(),
            checks=checks,
            request_id="readiness-check"
        )
        
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(status_code=503, detail="Service not ready")


@router.get("/system", response_model=SystemInfo)
async def system_info():
    """
    System information endpoint.
    
    Returns detailed system information for monitoring and debugging.
    Only available in debug mode for security reasons.
    """
    if not config.settings.debug:
        raise HTTPException(status_code=404, detail="Not found")
    
    try:
        # Get system information
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('.')
        
        system_info = {
            "cpu": {
                "usage_percent": cpu_percent,
                "count": psutil.cpu_count(),
                "count_logical": psutil.cpu_count(logical=True)
            },
            "memory": {
                "total_gb": round(memory.total / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "used_gb": round(memory.used / (1024**3), 2),
                "usage_percent": memory.percent
            },
            "disk": {
                "total_gb": round(disk.total / (1024**3), 2),
                "free_gb": round(disk.free / (1024**3), 2),
                "used_gb": round(disk.used / (1024**3), 2),
                "usage_percent": round((disk.used / disk.total) * 100, 2)
            },
            "uptime": time.time() - _start_time,
            "python_version": f"{psutil.version_info}",
            "process_id": psutil.Process().pid
        }
        
        return SystemInfo(
            status="ok",
            timestamp=datetime.utcnow(),
            system=system_info,
            request_id="system-info"
        )
        
    except Exception as e:
        logger.error(f"System info check failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system information")


@router.get("/ping")
async def ping():
    """
    Simple ping endpoint for basic connectivity testing.
    
    Returns a simple pong response with timestamp.
    """
    return {
        "message": "pong",
        "timestamp": datetime.utcnow().isoformat(),
        "status": "ok"
    }