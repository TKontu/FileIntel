"""Comprehensive unit tests for VRAM monitoring and batch processing systems."""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import threading

from src.fileintel.rag.graph_rag.services.vram_monitor import (
    VRAMMonitor,
    VRAMStats,
    BatchSizeOptimizer,
    MemoryPressureLevel,
    VRAMAlert,
)
from src.fileintel.rag.graph_rag.services.batch_processor import (
    AsyncBatchProcessor,
    BatchConfig,
    BatchJob,
    BatchResult,
    BatchStatus,
)


@pytest.fixture
def mock_nvidia_smi():
    """Mock nvidia-smi command output."""
    return """
<?xml version="1.0" ?>
<!DOCTYPE nvidia_smi_log SYSTEM "nvsmi_log.dtd">
<nvidia_smi_log>
    <gpu>
        <memory_usage>
            <total>24576 MiB</total>
            <used>8192 MiB</used>
            <free>16384 MiB</free>
        </memory_usage>
        <temperature>
            <gpu_temp>45 C</gpu_temp>
        </temperature>
        <utilization>
            <gpu_util>65 %</gpu_util>
            <memory_util>33 %</memory_util>
        </utilization>
    </gpu>
</nvidia_smi_log>
    """.strip()


@pytest.fixture
def mock_psutil():
    """Mock psutil for system memory monitoring."""
    mock_psutil = Mock()
    mock_psutil.virtual_memory.return_value = Mock(
        total=64 * 1024**3,  # 64 GB
        used=32 * 1024**3,  # 32 GB used
        available=32 * 1024**3,  # 32 GB available
        percent=50.0,
    )
    return mock_psutil


@pytest.fixture
def vram_monitor():
    """Create VRAM monitor for testing."""
    return VRAMMonitor(
        monitoring_interval=0.1,  # Fast for testing
        memory_threshold_mb=20480,  # 20GB threshold
        enable_optimization=True,
    )


@pytest.fixture
def batch_config():
    """Create batch configuration for testing."""
    return BatchConfig(
        initial_batch_size=4,
        min_batch_size=1,
        max_batch_size=16,
        batch_timeout=30.0,
        max_retries=3,
        adaptive_sizing=True,
        memory_threshold_mb=18000,  # 18GB threshold
    )


@pytest.fixture
def batch_processor(batch_config):
    """Create async batch processor for testing."""
    return AsyncBatchProcessor(batch_config)


class TestVRAMStats:
    """Test cases for VRAM statistics data structure."""

    def test_vram_stats_creation(self):
        """Test VRAM stats creation and properties."""
        stats = VRAMStats(
            total_mb=24576,
            used_mb=8192,
            free_mb=16384,
            utilization_percent=33.3,
            temperature_celsius=45,
            timestamp=datetime.utcnow(),
        )

        assert stats.total_mb == 24576
        assert stats.used_mb == 8192
        assert stats.free_mb == 16384
        assert stats.utilization_percent == 33.3
        assert stats.temperature_celsius == 45
        assert stats.memory_pressure_level == MemoryPressureLevel.LOW

    def test_memory_pressure_calculation(self):
        """Test memory pressure level calculation."""
        # Low pressure (< 70% usage)
        stats_low = VRAMStats(24576, 8192, 16384, 33.3, 45, datetime.utcnow())
        assert stats_low.memory_pressure_level == MemoryPressureLevel.LOW

        # Medium pressure (70-85% usage)
        stats_medium = VRAMStats(24576, 18432, 6144, 75.0, 55, datetime.utcnow())
        assert stats_medium.memory_pressure_level == MemoryPressureLevel.MEDIUM

        # High pressure (85-95% usage)
        stats_high = VRAMStats(24576, 22118, 2458, 90.0, 65, datetime.utcnow())
        assert stats_high.memory_pressure_level == MemoryPressureLevel.HIGH

        # Critical pressure (> 95% usage)
        stats_critical = VRAMStats(24576, 23347, 1229, 95.0, 75, datetime.utcnow())
        assert stats_critical.memory_pressure_level == MemoryPressureLevel.CRITICAL

    def test_vram_stats_serialization(self):
        """Test VRAM stats to dictionary conversion."""
        stats = VRAMStats(24576, 8192, 16384, 33.3, 45, datetime.utcnow())
        stats_dict = stats.to_dict()

        expected_keys = [
            "total_mb",
            "used_mb",
            "free_mb",
            "utilization_percent",
            "temperature_celsius",
            "memory_pressure_level",
            "timestamp",
        ]

        for key in expected_keys:
            assert key in stats_dict


class TestVRAMMonitor:
    """Test cases for VRAM monitoring functionality."""

    @patch("subprocess.run")
    def test_get_vram_stats_success(
        self, mock_subprocess, vram_monitor, mock_nvidia_smi
    ):
        """Test successful VRAM stats retrieval."""
        mock_subprocess.return_value.stdout = mock_nvidia_smi
        mock_subprocess.return_value.returncode = 0

        stats = vram_monitor.get_vram_stats()

        assert stats is not None
        assert stats.total_mb == 24576
        assert stats.used_mb == 8192
        assert stats.free_mb == 16384
        assert stats.utilization_percent == 33.0
        assert stats.temperature_celsius == 45

    @patch("subprocess.run")
    def test_get_vram_stats_nvidia_smi_failure(self, mock_subprocess, vram_monitor):
        """Test VRAM stats retrieval when nvidia-smi fails."""
        mock_subprocess.side_effect = Exception("nvidia-smi not found")

        stats = vram_monitor.get_vram_stats()
        assert stats is None

    @patch("subprocess.run")
    def test_get_vram_stats_invalid_xml(self, mock_subprocess, vram_monitor):
        """Test VRAM stats retrieval with invalid XML."""
        mock_subprocess.return_value.stdout = "invalid xml"
        mock_subprocess.return_value.returncode = 0

        stats = vram_monitor.get_vram_stats()
        assert stats is None

    @pytest.mark.asyncio
    @patch("subprocess.run")
    async def test_start_stop_monitoring(
        self, mock_subprocess, vram_monitor, mock_nvidia_smi
    ):
        """Test starting and stopping VRAM monitoring."""
        mock_subprocess.return_value.stdout = mock_nvidia_smi
        mock_subprocess.return_value.returncode = 0

        # Start monitoring
        await vram_monitor.start_monitoring()
        assert vram_monitor.is_monitoring is True

        # Wait for some monitoring cycles
        await asyncio.sleep(0.3)

        # Stop monitoring
        await vram_monitor.stop_monitoring()
        assert vram_monitor.is_monitoring is False

        # Should have collected some stats
        history = vram_monitor.get_stats_history(limit=10)
        assert len(history) > 0

    def test_stats_history_management(self, vram_monitor):
        """Test VRAM stats history management."""
        # Add multiple stats entries
        for i in range(15):
            stats = VRAMStats(
                total_mb=24576,
                used_mb=8192 + i * 100,
                free_mb=16384 - i * 100,
                utilization_percent=33.3 + i,
                temperature_celsius=45 + i,
                timestamp=datetime.utcnow(),
            )
            vram_monitor._add_stats_to_history(stats)

        # Should respect max history size (default 100)
        history = vram_monitor.get_stats_history()
        assert len(history) == 15

        # Test limit parameter
        limited_history = vram_monitor.get_stats_history(limit=5)
        assert len(limited_history) == 5

    def test_alert_generation(self, vram_monitor):
        """Test VRAM alert generation."""
        # Create high memory usage stats
        high_usage_stats = VRAMStats(
            total_mb=24576,
            used_mb=22118,  # 90% usage
            free_mb=2458,
            utilization_percent=90.0,
            temperature_celsius=75,
            timestamp=datetime.utcnow(),
        )

        alerts = vram_monitor._check_alert_conditions(high_usage_stats)
        assert len(alerts) > 0

        # Should have memory usage alert
        memory_alerts = [a for a in alerts if a.alert_type == "high_memory_usage"]
        assert len(memory_alerts) > 0

    def test_temperature_alert_generation(self, vram_monitor):
        """Test temperature-based alert generation."""
        high_temp_stats = VRAMStats(
            total_mb=24576,
            used_mb=8192,
            free_mb=16384,
            utilization_percent=33.3,
            temperature_celsius=85,  # High temperature
            timestamp=datetime.utcnow(),
        )

        alerts = vram_monitor._check_alert_conditions(high_temp_stats)
        temp_alerts = [a for a in alerts if a.alert_type == "high_temperature"]
        assert len(temp_alerts) > 0

    def test_get_current_stats(self, vram_monitor):
        """Test getting current VRAM statistics."""
        # Add some stats to history
        stats = VRAMStats(24576, 8192, 16384, 33.3, 45, datetime.utcnow())
        vram_monitor._add_stats_to_history(stats)

        current = vram_monitor.get_current_stats()
        assert current is not None
        assert current.total_mb == 24576


class TestBatchSizeOptimizer:
    """Test cases for batch size optimization."""

    def test_optimizer_initialization(self):
        """Test batch size optimizer initialization."""
        optimizer = BatchSizeOptimizer(
            initial_batch_size=4,
            min_batch_size=1,
            max_batch_size=16,
            memory_threshold_mb=18000,
        )

        assert optimizer.current_batch_size == 4
        assert optimizer.min_batch_size == 1
        assert optimizer.max_batch_size == 16

    def test_adjust_for_memory_pressure(self):
        """Test batch size adjustment for memory pressure."""
        optimizer = BatchSizeOptimizer(4, 1, 16, 18000)

        # High memory pressure should reduce batch size
        high_pressure_stats = VRAMStats(24576, 22118, 2458, 90.0, 65, datetime.utcnow())
        new_size = optimizer.adjust_batch_size(high_pressure_stats)
        assert new_size < 4

        # Low memory pressure should allow increase
        low_pressure_stats = VRAMStats(24576, 6144, 18432, 25.0, 45, datetime.utcnow())
        optimizer.current_batch_size = 2  # Start lower
        new_size = optimizer.adjust_batch_size(low_pressure_stats)
        assert new_size >= 2

    def test_batch_size_bounds_enforcement(self):
        """Test batch size bounds are enforced."""
        optimizer = BatchSizeOptimizer(4, 2, 8, 18000)

        # Should not go below minimum
        critical_stats = VRAMStats(24576, 23347, 1229, 95.0, 75, datetime.utcnow())
        for _ in range(10):  # Multiple adjustments
            new_size = optimizer.adjust_batch_size(critical_stats)
            assert new_size >= 2

        # Should not go above maximum
        low_stats = VRAMStats(24576, 2458, 22118, 10.0, 35, datetime.utcnow())
        optimizer.current_batch_size = 8
        for _ in range(10):  # Multiple adjustments
            new_size = optimizer.adjust_batch_size(low_stats)
            assert new_size <= 8

    def test_performance_tracking(self):
        """Test batch performance tracking."""
        optimizer = BatchSizeOptimizer(4, 1, 16, 18000)

        # Record some performance metrics
        optimizer.record_batch_performance(
            batch_size=4, processing_time=2.5, success=True
        )
        optimizer.record_batch_performance(
            batch_size=4, processing_time=3.0, success=True
        )
        optimizer.record_batch_performance(
            batch_size=8, processing_time=6.0, success=False
        )

        stats = optimizer.get_performance_stats()
        assert stats["total_batches"] == 3
        assert stats["successful_batches"] == 2
        assert stats["average_processing_time"] > 0


class TestBatchConfig:
    """Test cases for batch configuration."""

    def test_batch_config_creation(self):
        """Test batch configuration creation and validation."""
        config = BatchConfig(
            initial_batch_size=4,
            min_batch_size=1,
            max_batch_size=16,
            batch_timeout=30.0,
            max_retries=3,
            adaptive_sizing=True,
            memory_threshold_mb=18000,
        )

        assert config.initial_batch_size == 4
        assert config.adaptive_sizing is True
        assert config.memory_threshold_mb == 18000

    def test_batch_config_validation(self):
        """Test batch configuration validation."""
        # Invalid min/max relationship
        with pytest.raises(ValueError):
            BatchConfig(
                initial_batch_size=4,
                min_batch_size=10,
                max_batch_size=8,  # max < min
                batch_timeout=30.0,
            )

        # Invalid initial size
        with pytest.raises(ValueError):
            BatchConfig(
                initial_batch_size=20,  # > max
                min_batch_size=1,
                max_batch_size=16,
                batch_timeout=30.0,
            )


class TestBatchJob:
    """Test cases for batch job handling."""

    def test_batch_job_creation(self):
        """Test batch job creation."""
        job = BatchJob(
            job_id="test-job-1",
            data={"text": "test document"},
            priority=1,
            created_at=datetime.utcnow(),
        )

        assert job.job_id == "test-job-1"
        assert job.data["text"] == "test document"
        assert job.priority == 1
        assert job.status == BatchStatus.PENDING

    def test_batch_job_status_transitions(self):
        """Test batch job status transitions."""
        job = BatchJob("test-job-1", {"data": "test"}, 1, datetime.utcnow())

        # Pending -> Processing
        job.mark_processing()
        assert job.status == BatchStatus.PROCESSING
        assert job.started_at is not None

        # Processing -> Completed
        job.mark_completed({"result": "success"})
        assert job.status == BatchStatus.COMPLETED
        assert job.completed_at is not None
        assert job.result == {"result": "success"}

    def test_batch_job_failure_handling(self):
        """Test batch job failure handling."""
        job = BatchJob("test-job-1", {"data": "test"}, 1, datetime.utcnow())
        job.mark_processing()

        # Mark as failed
        error = Exception("Processing failed")
        job.mark_failed(error)
        assert job.status == BatchStatus.FAILED
        assert job.error == str(error)

    def test_batch_job_retry_logic(self):
        """Test batch job retry logic."""
        job = BatchJob(
            "test-job-1", {"data": "test"}, 1, datetime.utcnow(), max_retries=3
        )

        # First failure
        job.mark_failed(Exception("Error 1"))
        assert job.can_retry() is True
        assert job.retry_count == 0

        # Retry
        job.retry()
        assert job.status == BatchStatus.PENDING
        assert job.retry_count == 1

        # Max retries reached
        for i in range(3):
            job.mark_failed(Exception(f"Error {i+2}"))
            if job.can_retry():
                job.retry()

        assert job.can_retry() is False


class TestAsyncBatchProcessor:
    """Test cases for async batch processing."""

    @pytest.mark.asyncio
    async def test_batch_processor_initialization(self, batch_processor):
        """Test batch processor initialization."""
        assert batch_processor.config.initial_batch_size == 4
        assert batch_processor.current_batch_size == 4
        assert batch_processor.is_processing is False

    @pytest.mark.asyncio
    async def test_add_job_to_batch(self, batch_processor):
        """Test adding jobs to batch queue."""
        job = BatchJob("test-1", {"text": "test doc 1"}, 1, datetime.utcnow())
        await batch_processor.add_job(job)

        assert batch_processor.pending_jobs.qsize() == 1

    @pytest.mark.asyncio
    async def test_batch_processing_cycle(self, batch_processor):
        """Test complete batch processing cycle."""

        # Mock processing function
        async def mock_process_batch(jobs: List[BatchJob]) -> List[BatchResult]:
            results = []
            for job in jobs:
                result = BatchResult(
                    job_id=job.job_id,
                    success=True,
                    result={"processed": job.data["text"]},
                    processing_time=0.1,
                    timestamp=datetime.utcnow(),
                )
                results.append(result)
            return results

        # Add jobs
        jobs = [
            BatchJob(f"test-{i}", {"text": f"doc {i}"}, 1, datetime.utcnow())
            for i in range(3)
        ]

        for job in jobs:
            await batch_processor.add_job(job)

        # Start processing
        await batch_processor.start_processing(mock_process_batch)

        # Wait for processing
        await asyncio.sleep(0.5)

        # Stop processing
        await batch_processor.stop_processing()

        # Check results
        assert batch_processor.pending_jobs.qsize() == 0

    @pytest.mark.asyncio
    async def test_batch_size_adaptation(self, batch_processor):
        """Test adaptive batch size adjustment."""
        # Mock VRAM monitor with high memory pressure
        mock_vram_monitor = Mock()
        mock_vram_monitor.get_current_stats.return_value = VRAMStats(
            24576, 22118, 2458, 90.0, 65, datetime.utcnow()
        )

        batch_processor.vram_monitor = mock_vram_monitor
        batch_processor.config.adaptive_sizing = True

        # Process should adapt batch size
        await batch_processor._adapt_batch_size()

        # Should have reduced batch size due to high memory pressure
        assert batch_processor.current_batch_size < 4

    @pytest.mark.asyncio
    async def test_batch_timeout_handling(self, batch_processor):
        """Test batch timeout handling."""

        # Mock slow processing function
        async def slow_process_batch(jobs: List[BatchJob]) -> List[BatchResult]:
            await asyncio.sleep(2.0)  # Longer than timeout
            return []

        batch_processor.config.batch_timeout = 0.5

        job = BatchJob("test-timeout", {"text": "slow doc"}, 1, datetime.utcnow())
        await batch_processor.add_job(job)

        # Should handle timeout gracefully
        await batch_processor.start_processing(slow_process_batch)
        await asyncio.sleep(1.0)
        await batch_processor.stop_processing()

        # Job should still be in queue or marked as failed
        stats = batch_processor.get_processing_stats()
        assert stats["failed_batches"] > 0 or batch_processor.pending_jobs.qsize() > 0

    @pytest.mark.asyncio
    async def test_concurrent_batch_processing(self, batch_processor):
        """Test concurrent batch processing."""

        async def mock_process_batch(jobs: List[BatchJob]) -> List[BatchResult]:
            # Simulate some processing time
            await asyncio.sleep(0.1)
            return [
                BatchResult(
                    job.job_id, True, {"result": "processed"}, 0.1, datetime.utcnow()
                )
                for job in jobs
            ]

        # Add many jobs
        jobs = [
            BatchJob(f"test-{i}", {"text": f"doc {i}"}, 1, datetime.utcnow())
            for i in range(20)
        ]

        for job in jobs:
            await batch_processor.add_job(job)

        # Process with concurrent workers
        batch_processor.config.max_concurrent_batches = 3

        await batch_processor.start_processing(mock_process_batch)
        await asyncio.sleep(1.0)
        await batch_processor.stop_processing()

        # All jobs should be processed
        stats = batch_processor.get_processing_stats()
        assert stats["total_jobs_processed"] >= 20

    def test_batch_processor_stats(self, batch_processor):
        """Test batch processor statistics."""
        stats = batch_processor.get_processing_stats()

        expected_keys = [
            "total_jobs_processed",
            "successful_jobs",
            "failed_jobs",
            "current_batch_size",
            "average_batch_time",
            "total_batches_processed",
            "failed_batches",
            "queue_size",
            "is_processing",
        ]

        for key in expected_keys:
            assert key in stats


@pytest.mark.integration
class TestVRAMBatchIntegration:
    """Integration tests for VRAM monitoring and batch processing."""

    @pytest.mark.asyncio
    @patch("subprocess.run")
    async def test_integrated_memory_aware_processing(
        self, mock_subprocess, mock_nvidia_smi
    ):
        """Test integrated memory-aware batch processing."""
        mock_subprocess.return_value.stdout = mock_nvidia_smi
        mock_subprocess.return_value.returncode = 0

        # Create integrated system
        vram_monitor = VRAMMonitor(monitoring_interval=0.1)
        batch_config = BatchConfig(initial_batch_size=8, adaptive_sizing=True)
        batch_processor = AsyncBatchProcessor(batch_config)
        batch_processor.vram_monitor = vram_monitor

        # Start monitoring
        await vram_monitor.start_monitoring()

        # Mock processing function
        async def mock_process_batch(jobs: List[BatchJob]) -> List[BatchResult]:
            # Simulate memory-intensive processing
            await asyncio.sleep(0.1)
            return [
                BatchResult(
                    job.job_id, True, {"result": "processed"}, 0.1, datetime.utcnow()
                )
                for job in jobs
            ]

        # Add jobs
        jobs = [
            BatchJob(f"doc-{i}", {"text": f"document {i}"}, 1, datetime.utcnow())
            for i in range(16)
        ]

        for job in jobs:
            await batch_processor.add_job(job)

        # Start processing
        await batch_processor.start_processing(mock_process_batch)
        await asyncio.sleep(1.0)

        # System should adapt batch size based on memory usage
        final_batch_size = batch_processor.current_batch_size
        assert final_batch_size > 0  # Should still be processing

        # Cleanup
        await batch_processor.stop_processing()
        await vram_monitor.stop_monitoring()

    @pytest.mark.asyncio
    async def test_memory_pressure_response(self):
        """Test system response to memory pressure changes."""
        vram_monitor = VRAMMonitor()
        optimizer = BatchSizeOptimizer(8, 1, 16, 18000)

        # Simulate increasing memory pressure
        memory_scenarios = [
            (8192, 33.3),  # Low pressure
            (15360, 62.5),  # Medium pressure
            (20480, 83.3),  # High pressure
            (23347, 95.0),  # Critical pressure
        ]

        batch_sizes = []
        for used_mb, util_pct in memory_scenarios:
            stats = VRAMStats(
                24576, used_mb, 24576 - used_mb, util_pct, 45, datetime.utcnow()
            )
            new_size = optimizer.adjust_batch_size(stats)
            batch_sizes.append(new_size)

        # Batch sizes should generally decrease as memory pressure increases
        assert batch_sizes[0] >= batch_sizes[-1]  # First should be >= last

    @pytest.mark.asyncio
    async def test_error_recovery_under_memory_pressure(self):
        """Test error recovery under memory pressure conditions."""
        batch_processor = AsyncBatchProcessor(
            BatchConfig(initial_batch_size=8, adaptive_sizing=True, max_retries=2)
        )

        # Mock processing that fails under high memory pressure
        call_count = 0

        async def memory_sensitive_process(jobs: List[BatchJob]) -> List[BatchResult]:
            nonlocal call_count
            call_count += 1

            if len(jobs) > 4:  # Simulate memory pressure failure
                raise Exception("Out of memory error")

            return [
                BatchResult(
                    job.job_id, True, {"result": "processed"}, 0.1, datetime.utcnow()
                )
                for job in jobs
            ]

        # Add jobs
        jobs = [
            BatchJob(f"test-{i}", {"text": f"doc {i}"}, 1, datetime.utcnow())
            for i in range(10)
        ]

        for job in jobs:
            await batch_processor.add_job(job)

        # Should adapt batch size and retry on failures
        await batch_processor.start_processing(memory_sensitive_process)
        await asyncio.sleep(2.0)
        await batch_processor.stop_processing()

        # Should have processed some jobs despite initial failures
        stats = batch_processor.get_processing_stats()
        assert stats["total_jobs_processed"] > 0 or call_count > 1
