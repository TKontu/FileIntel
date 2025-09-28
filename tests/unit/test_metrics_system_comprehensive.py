"""Comprehensive unit tests for metrics collection and monitoring systems."""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import threading
from concurrent.futures import ThreadPoolExecutor

from src.fileintel.worker.metrics import (
    MetricsCollector,
    JobMetrics,
    SystemMetrics,
    PerformanceMetrics,
    MetricsExporter,
    MetricsAggregator,
    AlertThreshold,
)
from src.fileintel.worker.monitoring import (
    SystemMonitor,
    HealthCheck,
    HealthStatus,
    ComponentStatus,
    MonitoringConfig,
)


@pytest.fixture
def metrics_collector():
    """Create metrics collector for testing."""
    return MetricsCollector(
        collection_interval=0.1,  # Fast for testing
        export_interval=1.0,
        enable_prometheus=False,  # Disable for testing
        enable_structured_logging=True,
    )


@pytest.fixture
def monitoring_config():
    """Create monitoring configuration for testing."""
    return MonitoringConfig(
        health_check_interval=0.5,
        component_timeout=10.0,
        enable_system_metrics=True,
        enable_performance_metrics=True,
        alert_thresholds={
            "cpu_usage": AlertThreshold("cpu_usage", 80.0, "percentage"),
            "memory_usage": AlertThreshold("memory_usage", 85.0, "percentage"),
            "disk_usage": AlertThreshold("disk_usage", 90.0, "percentage"),
        },
    )


@pytest.fixture
def system_monitor(monitoring_config):
    """Create system monitor for testing."""
    return SystemMonitor(monitoring_config)


@pytest.fixture
def mock_prometheus_registry():
    """Mock Prometheus registry for testing."""
    registry = Mock()
    registry.register = Mock()
    registry.unregister = Mock()
    return registry


class TestJobMetrics:
    """Test cases for job metrics data structures."""

    def test_job_metrics_creation(self):
        """Test job metrics creation and validation."""
        metrics = JobMetrics(
            job_id="test-job-1",
            job_type="document_analysis",
            status="completed",
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow() + timedelta(seconds=5),
            processing_time=5.0,
            memory_usage_mb=512,
            cpu_usage_percent=45.0,
            retry_count=1,
            error_message=None,
        )

        assert metrics.job_id == "test-job-1"
        assert metrics.job_type == "document_analysis"
        assert metrics.status == "completed"
        assert metrics.processing_time == 5.0
        assert metrics.is_successful is True

    def test_job_metrics_failure_case(self):
        """Test job metrics for failed jobs."""
        metrics = JobMetrics(
            job_id="failed-job-1",
            job_type="text_analysis",
            status="failed",
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow() + timedelta(seconds=2),
            processing_time=2.0,
            memory_usage_mb=256,
            cpu_usage_percent=30.0,
            retry_count=3,
            error_message="Processing timeout",
        )

        assert metrics.is_successful is False
        assert metrics.retry_count == 3
        assert metrics.error_message == "Processing timeout"

    def test_job_metrics_serialization(self):
        """Test job metrics to dictionary conversion."""
        metrics = JobMetrics(
            job_id="test-job-1",
            job_type="document_analysis",
            status="completed",
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow() + timedelta(seconds=5),
            processing_time=5.0,
            memory_usage_mb=512,
            cpu_usage_percent=45.0,
            retry_count=0,
        )

        metrics_dict = metrics.to_dict()

        expected_keys = [
            "job_id",
            "job_type",
            "status",
            "start_time",
            "end_time",
            "processing_time",
            "memory_usage_mb",
            "cpu_usage_percent",
            "retry_count",
            "is_successful",
        ]

        for key in expected_keys:
            assert key in metrics_dict


class TestSystemMetrics:
    """Test cases for system metrics data structures."""

    def test_system_metrics_creation(self):
        """Test system metrics creation."""
        metrics = SystemMetrics(
            timestamp=datetime.utcnow(),
            cpu_usage_percent=65.0,
            memory_usage_mb=8192,
            memory_total_mb=16384,
            disk_usage_percent=45.0,
            network_bytes_sent=1024000,
            network_bytes_received=2048000,
            active_connections=25,
            load_average=2.5,
        )

        assert metrics.cpu_usage_percent == 65.0
        assert metrics.memory_usage_percent == 50.0  # 8192/16384 * 100
        assert metrics.disk_usage_percent == 45.0

    def test_system_metrics_thresholds(self):
        """Test system metrics threshold checking."""
        metrics = SystemMetrics(
            timestamp=datetime.utcnow(),
            cpu_usage_percent=85.0,  # High
            memory_usage_mb=14000,  # High (85.4% of 16384)
            memory_total_mb=16384,
            disk_usage_percent=95.0,  # Critical
            network_bytes_sent=1024000,
            network_bytes_received=2048000,
            active_connections=25,
            load_average=4.0,
        )

        assert metrics.is_cpu_high(80.0) is True
        assert metrics.is_memory_high(85.0) is True
        assert metrics.is_disk_critical(90.0) is True

    def test_system_metrics_health_assessment(self):
        """Test system health assessment."""
        healthy_metrics = SystemMetrics(
            datetime.utcnow(), 30.0, 4096, 16384, 50.0, 1024000, 2048000, 10, 1.5
        )

        unhealthy_metrics = SystemMetrics(
            datetime.utcnow(), 95.0, 15000, 16384, 98.0, 1024000, 2048000, 100, 8.0
        )

        assert healthy_metrics.overall_health_score() > 0.8
        assert unhealthy_metrics.overall_health_score() < 0.3


class TestMetricsCollector:
    """Test cases for metrics collector functionality."""

    def test_metrics_collector_initialization(self, metrics_collector):
        """Test metrics collector initialization."""
        assert metrics_collector.collection_interval == 0.1
        assert metrics_collector.export_interval == 1.0
        assert metrics_collector.is_collecting is False

    @pytest.mark.asyncio
    async def test_start_stop_collection(self, metrics_collector):
        """Test starting and stopping metrics collection."""
        await metrics_collector.start_collection()
        assert metrics_collector.is_collecting is True

        # Let it collect for a short time
        await asyncio.sleep(0.3)

        await metrics_collector.stop_collection()
        assert metrics_collector.is_collecting is False

        # Should have collected some metrics
        system_metrics = metrics_collector.get_system_metrics_history(limit=5)
        assert len(system_metrics) > 0

    def test_record_job_metrics(self, metrics_collector):
        """Test recording job metrics."""
        job_metrics = JobMetrics(
            job_id="test-job-1",
            job_type="document_analysis",
            status="completed",
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow() + timedelta(seconds=3),
            processing_time=3.0,
            memory_usage_mb=512,
            cpu_usage_percent=40.0,
            retry_count=0,
        )

        metrics_collector.record_job_metrics(job_metrics)

        # Should be stored in job metrics history
        job_history = metrics_collector.get_job_metrics_history(limit=10)
        assert len(job_history) == 1
        assert job_history[0].job_id == "test-job-1"

    def test_job_metrics_aggregation(self, metrics_collector):
        """Test job metrics aggregation by type."""
        # Record multiple job metrics
        job_types = ["document_analysis", "text_extraction", "document_analysis"]

        for i, job_type in enumerate(job_types):
            metrics = JobMetrics(
                job_id=f"job-{i}",
                job_type=job_type,
                status="completed" if i % 2 == 0 else "failed",
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow() + timedelta(seconds=2 + i),
                processing_time=2.0 + i,
                memory_usage_mb=512,
                cpu_usage_percent=40.0,
                retry_count=0 if i % 2 == 0 else 1,
            )
            metrics_collector.record_job_metrics(metrics)

        # Get aggregated metrics
        aggregated = metrics_collector.get_aggregated_job_metrics()

        assert "document_analysis" in aggregated
        assert "text_extraction" in aggregated
        assert aggregated["document_analysis"]["total_jobs"] == 2

    @patch("psutil.cpu_percent")
    @patch("psutil.virtual_memory")
    @patch("psutil.disk_usage")
    def test_system_metrics_collection(
        self, mock_disk, mock_memory, mock_cpu, metrics_collector
    ):
        """Test system metrics collection."""
        # Mock system stats
        mock_cpu.return_value = 45.0
        mock_memory.return_value = Mock(
            total=16 * 1024**3, used=8 * 1024**3, percent=50.0  # 16GB  # 8GB used
        )
        mock_disk.return_value = Mock(percent=65.0)

        system_metrics = metrics_collector._collect_system_metrics()

        assert system_metrics.cpu_usage_percent == 45.0
        assert system_metrics.memory_usage_percent == 50.0
        assert system_metrics.disk_usage_percent == 65.0

    def test_metrics_history_management(self, metrics_collector):
        """Test metrics history size management."""
        # Add many job metrics
        for i in range(150):
            metrics = JobMetrics(
                job_id=f"job-{i}",
                job_type="test",
                status="completed",
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow() + timedelta(seconds=1),
                processing_time=1.0,
                memory_usage_mb=256,
                cpu_usage_percent=30.0,
                retry_count=0,
            )
            metrics_collector.record_job_metrics(metrics)

        # Should respect max history size
        history = metrics_collector.get_job_metrics_history()
        assert len(history) <= 100  # Default max size

    def test_performance_metrics_calculation(self, metrics_collector):
        """Test performance metrics calculation."""
        # Record various job metrics
        processing_times = [1.0, 2.0, 3.0, 4.0, 5.0]

        for i, proc_time in enumerate(processing_times):
            metrics = JobMetrics(
                job_id=f"perf-job-{i}",
                job_type="performance_test",
                status="completed",
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow() + timedelta(seconds=proc_time),
                processing_time=proc_time,
                memory_usage_mb=512,
                cpu_usage_percent=50.0,
                retry_count=0,
            )
            metrics_collector.record_job_metrics(metrics)

        perf_metrics = metrics_collector.get_performance_metrics()

        assert perf_metrics.average_processing_time == 3.0  # Mean of 1,2,3,4,5
        assert perf_metrics.total_jobs == 5
        assert perf_metrics.success_rate == 1.0  # All successful

    def test_alert_threshold_evaluation(self, metrics_collector):
        """Test alert threshold evaluation."""
        # Set up alert thresholds
        thresholds = {
            "cpu_usage": AlertThreshold("cpu_usage", 80.0, "percentage"),
            "memory_usage": AlertThreshold("memory_usage", 85.0, "percentage"),
        }
        metrics_collector.alert_thresholds = thresholds

        # Create high-usage system metrics
        high_usage_metrics = SystemMetrics(
            timestamp=datetime.utcnow(),
            cpu_usage_percent=85.0,  # Above threshold
            memory_usage_mb=14000,  # ~85.4% of 16384, above threshold
            memory_total_mb=16384,
            disk_usage_percent=50.0,
            network_bytes_sent=1024000,
            network_bytes_received=2048000,
            active_connections=25,
            load_average=3.0,
        )

        alerts = metrics_collector._evaluate_alert_thresholds(high_usage_metrics)

        assert len(alerts) == 2  # CPU and memory alerts
        alert_types = [alert.metric_name for alert in alerts]
        assert "cpu_usage" in alert_types
        assert "memory_usage" in alert_types


class TestMetricsExporter:
    """Test cases for metrics export functionality."""

    def test_metrics_exporter_initialization(self):
        """Test metrics exporter initialization."""
        exporter = MetricsExporter(
            export_format="json", export_destination="file", export_interval=60.0
        )

        assert exporter.export_format == "json"
        assert exporter.export_destination == "file"

    @pytest.mark.asyncio
    async def test_json_metrics_export(self, tmp_path):
        """Test JSON metrics export."""
        export_file = tmp_path / "metrics.json"

        exporter = MetricsExporter(
            export_format="json",
            export_destination=str(export_file),
            export_interval=60.0,
        )

        # Mock metrics data
        metrics_data = {
            "system_metrics": {
                "cpu_usage_percent": 45.0,
                "memory_usage_percent": 60.0,
                "timestamp": datetime.utcnow().isoformat(),
            },
            "job_metrics": {
                "total_jobs": 100,
                "successful_jobs": 95,
                "average_processing_time": 2.5,
            },
        }

        await exporter.export_metrics(metrics_data)

        # Check that file was created and contains data
        assert export_file.exists()

        import json

        with open(export_file, "r") as f:
            exported_data = json.load(f)

        assert exported_data["system_metrics"]["cpu_usage_percent"] == 45.0
        assert exported_data["job_metrics"]["total_jobs"] == 100

    @pytest.mark.asyncio
    async def test_prometheus_metrics_export(self, mock_prometheus_registry):
        """Test Prometheus metrics export."""
        exporter = MetricsExporter(
            export_format="prometheus",
            export_destination="http://localhost:9090",
            export_interval=60.0,
        )

        with patch(
            "prometheus_client.CollectorRegistry", return_value=mock_prometheus_registry
        ):
            metrics_data = {
                "system_metrics": {
                    "cpu_usage_percent": 45.0,
                    "memory_usage_percent": 60.0,
                },
                "job_metrics": {
                    "total_jobs": 100,
                    "successful_jobs": 95,
                },
            }

            await exporter.export_metrics(metrics_data)

            # Should have registered metrics
            assert mock_prometheus_registry.register.called

    def test_metrics_formatting(self):
        """Test metrics data formatting for export."""
        exporter = MetricsExporter("json", "file", 60.0)

        raw_metrics = {
            "timestamp": datetime.utcnow(),
            "cpu_percent": 45.5,
            "memory_mb": 8192,
            "job_count": 150,
        }

        formatted = exporter._format_metrics(raw_metrics)

        # Timestamp should be formatted as ISO string
        assert isinstance(formatted["timestamp"], str)
        assert formatted["cpu_percent"] == 45.5


class TestSystemMonitor:
    """Test cases for system monitoring functionality."""

    def test_system_monitor_initialization(self, system_monitor):
        """Test system monitor initialization."""
        assert system_monitor.config.health_check_interval == 0.5
        assert system_monitor.is_monitoring is False

    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self, system_monitor):
        """Test starting and stopping system monitoring."""
        await system_monitor.start_monitoring()
        assert system_monitor.is_monitoring is True

        # Let it monitor for a short time
        await asyncio.sleep(1.0)

        await system_monitor.stop_monitoring()
        assert system_monitor.is_monitoring is False

    def test_component_health_check_registration(self, system_monitor):
        """Test registering component health checks."""

        # Mock health check function
        async def database_health_check() -> HealthStatus:
            return HealthStatus(
                component="database",
                status=ComponentStatus.HEALTHY,
                message="Database connection active",
                response_time=0.05,
                timestamp=datetime.utcnow(),
            )

        system_monitor.register_health_check("database", database_health_check)

        assert "database" in system_monitor.health_checks
        assert system_monitor.health_checks["database"] == database_health_check

    @pytest.mark.asyncio
    async def test_health_check_execution(self, system_monitor):
        """Test health check execution."""

        # Register mock health checks
        async def healthy_component() -> HealthStatus:
            return HealthStatus(
                "component1", ComponentStatus.HEALTHY, "OK", 0.01, datetime.utcnow()
            )

        async def unhealthy_component() -> HealthStatus:
            return HealthStatus(
                "component2", ComponentStatus.UNHEALTHY, "Error", 0.1, datetime.utcnow()
            )

        system_monitor.register_health_check("component1", healthy_component)
        system_monitor.register_health_check("component2", unhealthy_component)

        # Execute health checks
        health_results = await system_monitor.execute_health_checks()

        assert len(health_results) == 2
        assert health_results["component1"].status == ComponentStatus.HEALTHY
        assert health_results["component2"].status == ComponentStatus.UNHEALTHY

    def test_overall_system_health_assessment(self, system_monitor):
        """Test overall system health assessment."""
        # Mock health check results
        health_results = {
            "database": HealthStatus(
                "database", ComponentStatus.HEALTHY, "OK", 0.01, datetime.utcnow()
            ),
            "redis": HealthStatus(
                "redis", ComponentStatus.HEALTHY, "OK", 0.02, datetime.utcnow()
            ),
            "worker": HealthStatus(
                "worker", ComponentStatus.DEGRADED, "Slow", 0.5, datetime.utcnow()
            ),
        }

        overall_health = system_monitor._assess_overall_health(health_results)

        # Should be degraded due to worker component
        assert overall_health.status == ComponentStatus.DEGRADED
        assert "worker" in overall_health.message.lower()

    @pytest.mark.asyncio
    async def test_health_check_timeout_handling(self, system_monitor):
        """Test health check timeout handling."""

        # Mock slow health check
        async def slow_health_check() -> HealthStatus:
            await asyncio.sleep(15.0)  # Longer than timeout
            return HealthStatus(
                "slow_component", ComponentStatus.HEALTHY, "OK", 15.0, datetime.utcnow()
            )

        system_monitor.register_health_check("slow_component", slow_health_check)
        system_monitor.config.component_timeout = 1.0  # Short timeout

        health_results = await system_monitor.execute_health_checks()

        # Should timeout and mark as unhealthy
        assert health_results["slow_component"].status == ComponentStatus.UNHEALTHY
        assert "timeout" in health_results["slow_component"].message.lower()

    def test_monitoring_configuration_validation(self):
        """Test monitoring configuration validation."""
        # Valid configuration
        valid_config = MonitoringConfig(
            health_check_interval=1.0,
            component_timeout=10.0,
            enable_system_metrics=True,
        )

        assert valid_config.health_check_interval == 1.0

        # Invalid configuration
        with pytest.raises(ValueError):
            MonitoringConfig(
                health_check_interval=-1.0,  # Invalid negative interval
                component_timeout=5.0,
            )


class TestMetricsAggregator:
    """Test cases for metrics aggregation functionality."""

    def test_metrics_aggregator_initialization(self):
        """Test metrics aggregator initialization."""
        aggregator = MetricsAggregator(
            aggregation_window=timedelta(hours=1),
            aggregation_functions=["avg", "min", "max", "sum"],
        )

        assert aggregator.aggregation_window == timedelta(hours=1)
        assert "avg" in aggregator.aggregation_functions

    def test_job_metrics_aggregation_by_type(self):
        """Test job metrics aggregation by type."""
        aggregator = MetricsAggregator(
            aggregation_window=timedelta(hours=1),
            aggregation_functions=["avg", "count", "sum"],
        )

        # Add job metrics
        job_metrics = [
            JobMetrics(
                "job1",
                "type_a",
                "completed",
                datetime.utcnow(),
                datetime.utcnow(),
                2.0,
                512,
                40.0,
                0,
            ),
            JobMetrics(
                "job2",
                "type_a",
                "completed",
                datetime.utcnow(),
                datetime.utcnow(),
                3.0,
                512,
                45.0,
                0,
            ),
            JobMetrics(
                "job3",
                "type_b",
                "failed",
                datetime.utcnow(),
                datetime.utcnow(),
                1.0,
                256,
                30.0,
                1,
            ),
            JobMetrics(
                "job4",
                "type_a",
                "completed",
                datetime.utcnow(),
                datetime.utcnow(),
                4.0,
                768,
                50.0,
                0,
            ),
        ]

        for metrics in job_metrics:
            aggregator.add_job_metrics(metrics)

        # Get aggregated results
        aggregated = aggregator.aggregate_by_job_type()

        assert "type_a" in aggregated
        assert "type_b" in aggregated

        type_a_stats = aggregated["type_a"]
        assert type_a_stats["count"] == 3
        assert type_a_stats["avg_processing_time"] == 3.0  # (2+3+4)/3
        assert type_a_stats["success_rate"] == 1.0  # All successful

    def test_time_series_aggregation(self):
        """Test time series aggregation of metrics."""
        aggregator = MetricsAggregator(
            aggregation_window=timedelta(minutes=5),
            aggregation_functions=["avg", "max", "min"],
        )

        # Add system metrics across time
        base_time = datetime.utcnow()

        for i in range(10):
            timestamp = base_time + timedelta(minutes=i)
            system_metrics = SystemMetrics(
                timestamp=timestamp,
                cpu_usage_percent=40.0 + i * 2,  # Increasing CPU usage
                memory_usage_mb=8192,
                memory_total_mb=16384,
                disk_usage_percent=50.0,
                network_bytes_sent=1000000,
                network_bytes_received=2000000,
                active_connections=20,
                load_average=2.0,
            )
            aggregator.add_system_metrics(system_metrics)

        # Get time series aggregation
        time_series = aggregator.aggregate_time_series(
            metric_name="cpu_usage_percent", interval=timedelta(minutes=5)
        )

        assert len(time_series) > 0
        # Should have aggregated values for CPU usage
        for bucket in time_series:
            assert "avg" in bucket
            assert "max" in bucket
            assert "min" in bucket

    def test_percentile_calculations(self):
        """Test percentile calculations in aggregation."""
        aggregator = MetricsAggregator(
            aggregation_window=timedelta(hours=1),
            aggregation_functions=["p50", "p95", "p99"],
        )

        # Add job metrics with varying processing times
        processing_times = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

        for i, proc_time in enumerate(processing_times):
            metrics = JobMetrics(
                f"job{i}",
                "test_type",
                "completed",
                datetime.utcnow(),
                datetime.utcnow(),
                proc_time,
                512,
                40.0,
                0,
            )
            aggregator.add_job_metrics(metrics)

        # Get percentile aggregations
        percentiles = aggregator.calculate_percentiles("processing_time")

        assert "p50" in percentiles  # Median
        assert "p95" in percentiles
        assert "p99" in percentiles

        # P50 should be around 5.5 (median of 1-10)
        assert 5.0 <= percentiles["p50"] <= 6.0


@pytest.mark.integration
class TestMetricsIntegration:
    """Integration tests for complete metrics system."""

    @pytest.mark.asyncio
    async def test_end_to_end_metrics_collection(self, tmp_path):
        """Test end-to-end metrics collection and export."""
        export_file = tmp_path / "integration_metrics.json"

        # Set up integrated system
        metrics_collector = MetricsCollector(
            collection_interval=0.1, export_interval=0.5, enable_prometheus=False
        )

        exporter = MetricsExporter(
            export_format="json",
            export_destination=str(export_file),
            export_interval=0.5,
        )

        metrics_collector.add_exporter(exporter)

        # Start collection
        await metrics_collector.start_collection()

        # Record some job metrics
        for i in range(5):
            job_metrics = JobMetrics(
                f"integration-job-{i}",
                "integration_test",
                "completed",
                datetime.utcnow(),
                datetime.utcnow(),
                1.0 + i,
                512,
                40.0,
                0,
            )
            metrics_collector.record_job_metrics(job_metrics)

        # Let system collect and export
        await asyncio.sleep(1.0)

        # Stop collection
        await metrics_collector.stop_collection()

        # Verify export occurred
        assert export_file.exists()

    @pytest.mark.asyncio
    async def test_metrics_with_monitoring_integration(self):
        """Test metrics collection integrated with system monitoring."""
        monitoring_config = MonitoringConfig(
            health_check_interval=0.2,
            enable_system_metrics=True,
            enable_performance_metrics=True,
        )

        system_monitor = SystemMonitor(monitoring_config)
        metrics_collector = MetricsCollector(collection_interval=0.1)

        # Register health checks
        async def mock_database_health():
            return HealthStatus(
                "database",
                ComponentStatus.HEALTHY,
                "Connected",
                0.01,
                datetime.utcnow(),
            )

        async def mock_worker_health():
            return HealthStatus(
                "worker", ComponentStatus.HEALTHY, "Processing", 0.02, datetime.utcnow()
            )

        system_monitor.register_health_check("database", mock_database_health)
        system_monitor.register_health_check("worker", mock_worker_health)

        # Start both systems
        await system_monitor.start_monitoring()
        await metrics_collector.start_collection()

        # Let them run together
        await asyncio.sleep(1.0)

        # Stop both systems
        await metrics_collector.stop_collection()
        await system_monitor.stop_monitoring()

        # Verify integration worked
        system_metrics = metrics_collector.get_system_metrics_history(limit=5)
        assert len(system_metrics) > 0

        # Get final health status
        health_status = await system_monitor.get_overall_health()
        assert health_status.status in [
            ComponentStatus.HEALTHY,
            ComponentStatus.DEGRADED,
        ]

    @pytest.mark.asyncio
    async def test_performance_under_load(self):
        """Test metrics system performance under load."""
        metrics_collector = MetricsCollector(
            collection_interval=0.05, export_interval=1.0  # Very frequent collection
        )

        await metrics_collector.start_collection()

        # Simulate high load with many job metrics
        start_time = time.time()

        tasks = []
        for batch in range(10):  # 10 batches
            task = asyncio.create_task(self._record_job_batch(metrics_collector, batch))
            tasks.append(task)

        await asyncio.gather(*tasks)

        end_time = time.time()

        await metrics_collector.stop_collection()

        # Should complete reasonably quickly
        assert end_time - start_time < 5.0

        # Should have recorded all metrics
        job_history = metrics_collector.get_job_metrics_history()
        assert len(job_history) == 100  # 10 batches * 10 jobs each

    async def _record_job_batch(self, collector: MetricsCollector, batch_id: int):
        """Record a batch of job metrics for load testing."""
        for i in range(10):
            job_metrics = JobMetrics(
                f"load-test-{batch_id}-{i}",
                "load_test",
                "completed",
                datetime.utcnow(),
                datetime.utcnow(),
                0.5,
                256,
                35.0,
                0,
            )
            collector.record_job_metrics(job_metrics)
            await asyncio.sleep(0.01)  # Small delay between jobs
