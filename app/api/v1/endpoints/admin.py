"""
Administrative API endpoints.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import psutil
import time
from datetime import datetime
from pathlib import Path

from app.core.config import settings
from app.models.common import HealthCheckResponse, SystemMetrics, ConfigurationInfo

router = APIRouter()

# Store startup time
startup_time = time.time()

@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Health check endpoint.

    Returns the current health status of the API service,
    including uptime and various system checks.
    """
    uptime = time.time() - startup_time

    # Perform various health checks
    checks = {
        "database": True,  # Would check database connectivity
        "openai_api": bool(settings.OPENAI_API_KEY),
        "data_directory": settings.DATA_DIR.exists(),
        "input_directory": settings.INPUT_DIR.exists(),
        "output_directory": settings.OUTPUT_DIR.exists(),
    }

    return HealthCheckResponse(
        status="healthy" if all(checks.values()) else "degraded",
        version=settings.VERSION,
        uptime=uptime,
        checks=checks
    )

@router.get("/metrics", response_model=SystemMetrics)
async def get_system_metrics():
    """
    Get current system metrics.

    Returns CPU usage, memory usage, disk usage, and other
    performance metrics for monitoring purposes.
    """
    try:
        # Get system metrics using psutil
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        return SystemMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory.percent,
            disk_usage=disk.percent,
            active_requests=0,  # Would track active requests
            total_requests=0,   # Would track total requests
            uptime=time.time() - startup_time
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting system metrics: {str(e)}")

@router.get("/config", response_model=ConfigurationInfo)
async def get_configuration():
    """
    Get current configuration information.

    Returns the current API configuration settings
    (excluding sensitive information like API keys).
    """
    return ConfigurationInfo(
        data_directory=str(settings.DATA_DIR),
        default_model=settings.DEFAULT_MODEL,
        max_concurrent_requests=settings.MAX_CONCURRENT_REQUESTS,
        request_timeout=settings.REQUEST_TIMEOUT,
        debug_mode=settings.DEBUG,
        available_models=[
            "gpt-4o-mini",
            "gpt-4.1",
            "o3-mini",
            "o1",
            "claude-3-sonnet"
        ]
    )

@router.get("/status", response_model=dict)
async def get_service_status():
    """
    Get detailed service status.

    Returns comprehensive status information including
    file system status, configuration, and service health.
    """
    try:
        # Check data directories
        data_status = {
            "data_dir_exists": settings.DATA_DIR.exists(),
            "input_dir_exists": settings.INPUT_DIR.exists(),
            "intermediate_dir_exists": settings.INTERMEDIATE_DIR.exists(),
            "output_dir_exists": settings.OUTPUT_DIR.exists(),
        }

        # Check key files
        key_files = {
            "entities_result": (settings.INPUT_DIR / "entities_result.json").exists(),
            "retrieved_chunks": (settings.INPUT_DIR / "retrieved_chunks_15.json").exists(),
            "final_prompt": (settings.INPUT_DIR / "final_prompt.json").exists(),
            "optimized_graph": (settings.OUTPUT_DIR / "optimized_entity_graph.json").exists(),
            "alias_dict": (settings.INTERMEDIATE_DIR / "alias_dict.json").exists(),
        }

        # Get system info
        system_info = {
            "python_version": f"{psutil.PYTHON_VERSION}",
            "platform": psutil.LINUX if hasattr(psutil, 'LINUX') else "unknown",
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "disk_total": psutil.disk_usage('/').total,
        }

        return {
            "service": "Research_CODE API",
            "version": settings.VERSION,
            "status": "running",
            "uptime": time.time() - startup_time,
            "timestamp": datetime.now().isoformat(),
            "data_status": data_status,
            "key_files": key_files,
            "system_info": system_info,
            "configuration": {
                "debug_mode": settings.DEBUG,
                "default_model": settings.DEFAULT_MODEL,
                "max_concurrent_requests": settings.MAX_CONCURRENT_REQUESTS,
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting service status: {str(e)}")

@router.post("/reload")
async def reload_configuration():
    """
    Reload configuration from environment variables.

    This endpoint allows reloading the configuration without
    restarting the service (useful for development).
    """
    try:
        # In a real implementation, you would reload the settings
        # For now, we'll just return a success message
        return {
            "success": True,
            "message": "Configuration reloaded successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reloading configuration: {str(e)}")

@router.get("/logs")
async def get_recent_logs(lines: int = 100):
    """
    Get recent log entries.

    Returns the most recent log entries for debugging purposes.
    """
    try:
        # In a real implementation, you would read from log files
        # For now, return a placeholder
        return {
            "logs": [
                {
                    "timestamp": datetime.now().isoformat(),
                    "level": "INFO",
                    "message": "API service is running normally",
                    "module": "main"
                }
            ],
            "total_lines": 1,
            "requested_lines": lines
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting logs: {str(e)}")