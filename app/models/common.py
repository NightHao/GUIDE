"""
Common Pydantic models used across the API.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any, Union
from datetime import datetime
from enum import Enum

class OperationStatus(str, Enum):
    """Enumeration of operation statuses."""
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    PENDING = "pending"
    RUNNING = "running"

class APIResponse(BaseModel):
    """Base API response model."""
    status: OperationStatus = Field(description="Operation status")
    message: str = Field(description="Response message")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    data: Optional[Any] = Field(description="Response data")
    errors: Optional[List[str]] = Field(description="List of errors if any")

class HealthCheckResponse(BaseModel):
    """Health check response model."""
    status: str = Field(description="Service status")
    version: str = Field(description="API version")
    uptime: float = Field(description="Service uptime in seconds")
    checks: Dict[str, bool] = Field(description="Individual health checks")

class FileStatus(BaseModel):
    """Model for file status information."""
    exists: bool = Field(description="Whether the file exists")
    path: str = Field(description="File path")
    size: Optional[int] = Field(description="File size in bytes")
    last_modified: Optional[datetime] = Field(description="Last modification time")
    is_readable: bool = Field(description="Whether the file is readable")

class ProcessingProgress(BaseModel):
    """Model for processing progress."""
    total_items: int = Field(description="Total number of items to process")
    processed_items: int = Field(description="Number of items processed")
    remaining_items: int = Field(description="Number of items remaining")
    progress_percentage: float = Field(description="Progress percentage")
    estimated_time_remaining: Optional[float] = Field(description="Estimated time remaining in seconds")
    current_item: Optional[str] = Field(description="Currently processing item")

class TaskStatus(BaseModel):
    """Model for async task status."""
    task_id: str = Field(description="Unique task identifier")
    status: OperationStatus = Field(description="Task status")
    created_at: datetime = Field(description="Task creation time")
    started_at: Optional[datetime] = Field(description="Task start time")
    completed_at: Optional[datetime] = Field(description="Task completion time")
    progress: Optional[ProcessingProgress] = Field(description="Task progress")
    result: Optional[Any] = Field(description="Task result if completed")
    error: Optional[str] = Field(description="Error message if failed")

class SystemMetrics(BaseModel):
    """Model for system metrics."""
    cpu_usage: float = Field(description="CPU usage percentage")
    memory_usage: float = Field(description="Memory usage percentage")
    disk_usage: float = Field(description="Disk usage percentage")
    active_requests: int = Field(description="Number of active requests")
    total_requests: int = Field(description="Total requests processed")
    uptime: float = Field(description="System uptime in seconds")

class ErrorDetail(BaseModel):
    """Model for detailed error information."""
    error_code: str = Field(description="Error code")
    error_message: str = Field(description="Error message")
    error_type: str = Field(description="Error type")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
    context: Optional[Dict[str, Any]] = Field(description="Additional error context")

class ValidationError(BaseModel):
    """Model for validation errors."""
    field: str = Field(description="Field that failed validation")
    message: str = Field(description="Validation error message")
    rejected_value: Any = Field(description="Value that was rejected")

class ConfigurationInfo(BaseModel):
    """Model for configuration information."""
    data_directory: str = Field(description="Data directory path")
    default_model: str = Field(description="Default LLM model")
    max_concurrent_requests: int = Field(description="Maximum concurrent requests")
    request_timeout: int = Field(description="Request timeout in seconds")
    debug_mode: bool = Field(description="Whether debug mode is enabled")
    available_models: List[str] = Field(description="List of available models")