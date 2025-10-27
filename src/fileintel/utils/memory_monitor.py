"""
Memory monitoring utilities for Celery workers.

Provides tools to track memory usage, detect memory leaks, and
manage garbage collection in long-running worker processes.
"""

import gc
import logging
import os
import psutil
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class MemoryMonitor:
    """Monitor and log memory usage for Celery tasks."""

    def __init__(self, task_name: str = "unknown", enable_gc: bool = True):
        """
        Initialize memory monitor.

        Args:
            task_name: Name of the task being monitored
            enable_gc: Whether to enable aggressive garbage collection
        """
        self.task_name = task_name
        self.enable_gc = enable_gc
        self.process = psutil.Process(os.getpid())
        self.start_memory = None
        self.peak_memory = 0

    def get_memory_info(self) -> Dict[str, Any]:
        """
        Get detailed memory information for the current process.

        Returns:
            Dict containing memory metrics in MB and percentages
        """
        try:
            # Process memory info
            mem_info = self.process.memory_info()
            mem_percent = self.process.memory_percent()

            # System memory info
            virtual_mem = psutil.virtual_memory()

            # Convert bytes to MB for readability
            rss_mb = mem_info.rss / (1024 * 1024)
            vms_mb = mem_info.vms / (1024 * 1024)

            # Track peak memory
            if rss_mb > self.peak_memory:
                self.peak_memory = rss_mb

            return {
                "timestamp": datetime.now().isoformat(),
                "task_name": self.task_name,
                "process": {
                    "pid": self.process.pid,
                    "rss_mb": round(rss_mb, 2),  # Resident Set Size (actual RAM)
                    "vms_mb": round(vms_mb, 2),  # Virtual Memory Size
                    "percent": round(mem_percent, 2),
                    "peak_mb": round(self.peak_memory, 2),
                },
                "system": {
                    "total_mb": round(virtual_mem.total / (1024 * 1024), 2),
                    "available_mb": round(virtual_mem.available / (1024 * 1024), 2),
                    "used_percent": virtual_mem.percent,
                },
                "gc": {
                    "collections": gc.get_count(),
                    "objects": len(gc.get_objects()),
                },
            }
        except Exception as e:
            logger.error(f"Failed to get memory info: {e}")
            return {"error": str(e)}

    def log_memory(self, context: str = "") -> None:
        """
        Log current memory usage.

        Args:
            context: Additional context for the log message
        """
        mem_info = self.get_memory_info()

        if "error" in mem_info:
            logger.error(f"Memory monitoring error: {mem_info['error']}")
            return

        process_info = mem_info["process"]
        system_info = mem_info["system"]
        gc_info = mem_info["gc"]

        log_msg = (
            f"Memory [{self.task_name}] {context}: "
            f"RSS={process_info['rss_mb']:.1f}MB "
            f"(peak={process_info['peak_mb']:.1f}MB), "
            f"Process%={process_info['percent']:.1f}%, "
            f"System={system_info['used_percent']:.1f}%, "
            f"GC objects={gc_info['objects']:,}"
        )

        # Warn if memory usage is high
        if process_info["percent"] > 80:
            logger.warning(f"HIGH MEMORY USAGE: {log_msg}")
        else:
            logger.info(log_msg)

    def start_monitoring(self) -> None:
        """Start memory monitoring (capture baseline)."""
        mem_info = self.get_memory_info()
        self.start_memory = mem_info["process"]["rss_mb"]
        self.log_memory("START")

    def checkpoint(self, label: str) -> None:
        """
        Log memory at a checkpoint.

        Args:
            label: Label for this checkpoint
        """
        self.log_memory(f"checkpoint:{label}")

    def finish_monitoring(self) -> Dict[str, Any]:
        """
        Finish monitoring and return summary.

        Returns:
            Summary of memory usage during task execution
        """
        mem_info = self.get_memory_info()
        current_memory = mem_info["process"]["rss_mb"]

        summary = {
            "task_name": self.task_name,
            "start_mb": round(self.start_memory, 2) if self.start_memory else 0,
            "finish_mb": round(current_memory, 2),
            "peak_mb": round(self.peak_memory, 2),
            "delta_mb": round(current_memory - self.start_memory, 2)
            if self.start_memory
            else 0,
        }

        # Log completion
        self.log_memory(
            f"FINISH (delta={summary['delta_mb']:+.1f}MB, peak={summary['peak_mb']:.1f}MB)"
        )

        # Aggressive garbage collection if enabled
        if self.enable_gc:
            self.force_garbage_collection()

        return summary

    def force_garbage_collection(self) -> Dict[str, Any]:
        """
        Force garbage collection and log results.

        Returns:
            Dictionary with garbage collection stats
        """
        before_mem = self.get_memory_info()["process"]["rss_mb"]

        # Force full garbage collection across all generations
        collected = {
            "gen0": gc.collect(0),  # Young objects
            "gen1": gc.collect(1),  # Middle-aged objects
            "gen2": gc.collect(2),  # Old objects (full collection)
        }

        after_mem = self.get_memory_info()["process"]["rss_mb"]
        freed_mb = before_mem - after_mem

        result = {
            "before_mb": round(before_mem, 2),
            "after_mb": round(after_mem, 2),
            "freed_mb": round(freed_mb, 2),
            "objects_collected": sum(collected.values()),
            "collections_by_generation": collected,
        }

        if freed_mb > 10:  # Only log if significant memory freed
            logger.info(
                f"Garbage collection [{self.task_name}]: "
                f"freed {freed_mb:.1f}MB, "
                f"collected {result['objects_collected']} objects"
            )

        return result


def log_worker_memory_summary():
    """
    Log memory summary for the entire worker process.

    Useful for periodic monitoring or worker startup/shutdown.
    """
    try:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        mem_percent = process.memory_percent()
        virtual_mem = psutil.virtual_memory()

        logger.info(
            f"Worker memory summary: "
            f"PID={process.pid}, "
            f"RSS={mem_info.rss / (1024 * 1024):.1f}MB, "
            f"Process%={mem_percent:.1f}%, "
            f"System={virtual_mem.percent:.1f}% "
            f"(available={virtual_mem.available / (1024 * 1024):.1f}MB)"
        )
    except Exception as e:
        logger.error(f"Failed to log worker memory summary: {e}")


def check_memory_threshold(threshold_mb: float = 3000.0) -> bool:
    """
    Check if process memory exceeds threshold.

    Args:
        threshold_mb: Memory threshold in MB

    Returns:
        True if memory usage is below threshold, False otherwise
    """
    try:
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / (1024 * 1024)

        if mem_mb > threshold_mb:
            logger.warning(
                f"Memory threshold exceeded: {mem_mb:.1f}MB > {threshold_mb:.1f}MB"
            )
            return False
        return True
    except Exception as e:
        logger.error(f"Failed to check memory threshold: {e}")
        return True  # Fail open


def get_system_memory_summary() -> Dict[str, Any]:
    """
    Get comprehensive system memory summary.

    Returns:
        Dictionary with system-wide memory information
    """
    try:
        virtual_mem = psutil.virtual_memory()
        swap_mem = psutil.swap_memory()

        return {
            "virtual_memory": {
                "total_gb": round(virtual_mem.total / (1024**3), 2),
                "available_gb": round(virtual_mem.available / (1024**3), 2),
                "used_gb": round(virtual_mem.used / (1024**3), 2),
                "percent": virtual_mem.percent,
            },
            "swap_memory": {
                "total_gb": round(swap_mem.total / (1024**3), 2),
                "used_gb": round(swap_mem.used / (1024**3), 2),
                "percent": swap_mem.percent,
            },
        }
    except Exception as e:
        logger.error(f"Failed to get system memory summary: {e}")
        return {"error": str(e)}
