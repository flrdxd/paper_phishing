"""Resource monitoring for training and inference operations."""

from __future__ import annotations

import time
import psutil
import logging
from datetime import datetime
from typing import Any, Callable
from functools import wraps
from contextlib import contextmanager

logger = logging.getLogger(__name__)


def get_system_resources() -> dict[str, Any]:
    """Get current system resource usage."""
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()

        resources = {
            "cpu_percent": cpu_percent,
            "memory_used_gb": memory.used / (1024**3),
            "memory_available_gb": memory.available / (1024**3),
            "memory_percent": memory.percent,
            "memory_total_gb": memory.total / (1024**3),
        }

        # GPU information if available
        try:
            import torch
            if torch.cuda.is_available():
                gpu_memory_allocated = torch.cuda.memory_allocated() / (1024**3)
                gpu_memory_reserved = torch.cuda.memory_reserved() / (1024**3)
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)

                resources.update({
                    "gpu_used_gb": gpu_memory_allocated,
                    "gpu_reserved_gb": gpu_memory_reserved,
                    "gpu_total_gb": gpu_memory_total,
                    "gpu_percent": (gpu_memory_allocated / gpu_memory_total) * 100,
                })
        except Exception:
            pass

        return resources
    except Exception as e:
        logger.warning(f"Failed to get system resources: {e}")
        return {}


@contextmanager
def monitor_resources(operation_name: str):
    """Context manager to monitor resources during an operation.

    Args:
        operation_name: Name of the operation being monitored

    Yields:
        dict: Resource usage metrics
    """
    logger.info(f"Starting resource monitoring for: {operation_name}")
    start_time = time.time()
    start_resources = get_system_resources()

    resources_log = []

    try:
        yield {
            "operation": operation_name,
            "start_time": datetime.now().isoformat(),
            "start_resources": start_resources,
            "log": resources_log,
        }
    finally:
        end_time = time.time()
        end_resources = get_system_resources()
        elapsed = end_time - start_time

        summary = {
            "operation": operation_name,
            "start_time": datetime.fromtimestamp(start_time).isoformat(),
            "end_time": datetime.fromtimestamp(end_time).isoformat(),
            "elapsed_seconds": elapsed,
            "start_resources": start_resources,
            "end_resources": end_resources,
            "resource_delta": {
                k: end_resources.get(k, 0) - v for k, v in start_resources.items()
            } if start_resources and end_resources else {},
        }

        logger.info(f"Resource monitoring complete for: {operation_name}")
        logger.info(f"  Elapsed time: {elapsed:.2f} seconds")
        if start_resources and end_resources:
            logger.info(f"  CPU: {start_resources.get('cpu_percent', 0):.1f}% -> {end_resources.get('cpu_percent', 0):.1f}%")
            logger.info(f"  Memory: {start_resources.get('memory_used_gb', 0):.2f}GB -> {end_resources.get('memory_used_gb', 0):.2f}GB")
            if "gpu_used_gb" in end_resources:
                logger.info(f"  GPU: {start_resources.get('gpu_used_gb', 0):.2f}GB -> {end_resources.get('gpu_used_gb', 0):.2f}GB")


def timed_operation(operation_name: str):
    """Decorator to time and monitor resource usage of a function.

    Args:
        operation_name: Name of the operation for logging

    Example:
        @timed_operation("model_training")
        def train_model():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            with monitor_resources(operation_name) as monitoring:
                result = func(*args, **kwargs)
                return result, monitoring
        return wrapper
    return decorator


def format_resource_summary(resource_data: dict[str, Any]) -> str:
    """Format resource monitoring data for human reading.

    Args:
        resource_data: Resource monitoring data from monitor_resources

    Returns:
        Formatted string with key metrics
    """
    if not resource_data:
        return "No resource data available"

    lines = [
        f"Operation: {resource_data.get('operation', 'unknown')}",
        f"Duration: {resource_data.get('elapsed_seconds', 0):.2f} seconds",
    ]

    start_res = resource_data.get('start_resources', {})
    end_res = resource_data.get('end_resources', {})

    if start_res:
        lines.append(f"CPU: {start_res.get('cpu_percent', 0):.1f}% -> {end_res.get('cpu_percent', 0):.1f}%")
        lines.append(f"Memory: {start_res.get('memory_used_gb', 0):.2f}GB -> {end_res.get('memory_used_gb', 0):.2f}GB")

    if start_res and 'gpu_used_gb' in end_res:
        lines.append(f"GPU Memory: {start_res.get('gpu_used_gb', 0):.2f}GB -> {end_res.get('gpu_used_gb', 0):.2f}GB")

    return "\n".join(lines)