"""Tests for metrics collection service."""

import pytest
import time
from unittest.mock import patch, Mock

from fileintel.worker.metrics import (
    MetricsCollector,
    JobMetrics,
    JobType,
    metrics_collector,
)


@pytest.fixture
def collector():
    """Create a fresh metrics collector for testing."""
    return MetricsCollector()


def test_metrics_collector_initialization(collector):
    """Test metrics collector initialization."""
    assert collector.config is not None
    assert collector.registry is not None
    assert len(collector.job_history) == 0
    assert collector.max_history_size == 10000

    # Check that all Prometheus metrics are created
    assert collector.jobs_total is not None
    assert collector.job_duration is not None
    assert collector.batch_processing_speedup is not None
    assert collector.llm_request_duration is not None
    assert collector.job_errors is not None


def test_job_metrics_creation():
    """Test JobMetrics creation."""
    start_time = time.time()
    metrics = JobMetrics(job_type="test_job", start_time=start_time)

    assert metrics.job_type == "test_job"
    assert metrics.start_time == start_time
    assert metrics.end_time is None
    assert metrics.success is False
    assert metrics.error_type is None
    assert metrics.processing_time == 0.0
    assert metrics.metadata == {}


def test_record_job_start(collector):
    """Test recording job start."""
    job_metrics = collector.record_job_start("indexing", "job123")

    assert isinstance(job_metrics, JobMetrics)
    assert job_metrics.job_type == "indexing"
    assert job_metrics.start_time > 0
    assert job_metrics.end_time is None


def test_record_job_completion_success(collector):
    """Test recording successful job completion."""
    # Start a job
    job_metrics = collector.record_job_start("indexing", "job123")
    initial_history_size = len(collector.job_history)

    # Complete it successfully
    time.sleep(0.01)  # Small delay to ensure processing time > 0
    collector.record_job_completion(
        job_metrics, success=True, test_metadata="test_value"
    )

    assert job_metrics.success is True
    assert job_metrics.end_time is not None
    assert job_metrics.processing_time > 0
    assert job_metrics.metadata["test_metadata"] == "test_value"
    assert len(collector.job_history) == initial_history_size + 1


def test_record_job_completion_failure(collector):
    """Test recording failed job completion."""
    job_metrics = collector.record_job_start("graphrag", "job456")

    collector.record_job_completion(
        job_metrics, success=False, error_type="llm_timeout"
    )

    assert job_metrics.success is False
    assert job_metrics.error_type == "llm_timeout"
    assert job_metrics.end_time is not None


def test_record_batch_processing(collector):
    """Test recording batch processing metrics."""
    # Test with valid speedup
    collector.record_batch_processing(
        batch_size=4, batch_time=2.0, sequential_time_estimate=8.0
    )

    # Test with zero sequential time (should not crash)
    collector.record_batch_processing(
        batch_size=2, batch_time=1.0, sequential_time_estimate=0.0
    )

    # No exceptions should be raised
    assert True


def test_record_llm_request(collector):
    """Test recording LLM request metrics."""
    # Successful request
    collector.record_llm_request(
        provider="openai", model="gpt-4", duration=1.5, success=True
    )

    # Failed request
    collector.record_llm_request(
        provider="openai", model="gpt-4", duration=0.5, success=False
    )

    # No exceptions should be raised
    assert True


def test_record_chunk_processing(collector):
    """Test recording chunk processing metrics."""
    collector.record_chunk_processing("entity_extraction", 0.8)
    collector.record_chunk_processing("embedding", 0.3)

    # No exceptions should be raised
    assert True


def test_record_job_retry(collector):
    """Test recording job retry metrics."""
    collector.record_job_retry("indexing", 60.0)
    collector.record_job_retry("graphrag", 120.0)

    # No exceptions should be raised
    assert True


def test_update_queue_metrics(collector):
    """Test updating queue metrics."""
    collector.update_queue_metrics(
        pending_jobs=10, processing_jobs=3, failed_jobs=1, active_workers=2
    )

    # No exceptions should be raised
    assert True


def test_update_circuit_breaker_metrics(collector):
    """Test updating circuit breaker metrics."""
    collector.update_circuit_breaker_metrics(
        "openai", 1, 3
    )  # Open state with 3 failures
    collector.update_circuit_breaker_metrics("openai", 0, 0)  # Closed state

    # No exceptions should be raised
    assert True


def test_update_dead_letter_queue_size(collector):
    """Test updating dead letter queue size."""
    collector.update_dead_letter_queue_size(5)
    collector.update_dead_letter_queue_size(0)

    # No exceptions should be raised
    assert True


def test_get_job_success_rates(collector):
    """Test calculating job success rates."""
    # Add some job history
    for i in range(10):
        job_metrics = JobMetrics(
            job_type="indexing",
            start_time=time.time() - 1800,  # 30 minutes ago
            end_time=time.time() - 1700,
            success=i < 8,  # 8 successes, 2 failures
            processing_time=100.0,
        )
        collector.job_history.append(job_metrics)

    # Add some GraphRAG jobs
    for i in range(5):
        job_metrics = JobMetrics(
            job_type="graphrag",
            start_time=time.time() - 1800,
            end_time=time.time() - 1700,
            success=i < 4,  # 4 successes, 1 failure
            processing_time=200.0,
        )
        collector.job_history.append(job_metrics)

    success_rates = collector.get_job_success_rates()

    assert "indexing" in success_rates
    assert "graphrag" in success_rates
    assert success_rates["indexing"] == 0.8  # 8/10
    assert success_rates["graphrag"] == 0.8  # 4/5


def test_get_average_processing_times(collector):
    """Test calculating average processing times."""
    # Add job history with different processing times
    processing_times = [1.0, 2.0, 3.0, 4.0, 5.0]
    for i, proc_time in enumerate(processing_times):
        job_metrics = JobMetrics(
            job_type="indexing",
            start_time=time.time() - 1800,
            end_time=time.time() - 1800 + proc_time,
            success=True,
            processing_time=proc_time,
        )
        collector.job_history.append(job_metrics)

    avg_times = collector.get_average_processing_times()

    assert "indexing" in avg_times
    expected_avg = sum(processing_times) / len(processing_times)
    assert abs(avg_times["indexing"] - expected_avg) < 0.01


def test_get_metrics_summary(collector):
    """Test getting comprehensive metrics summary."""
    # Add some test data
    job_metrics = JobMetrics(
        job_type="test",
        start_time=time.time() - 1800,
        end_time=time.time() - 1700,
        success=True,
        processing_time=100.0,
    )
    collector.job_history.append(job_metrics)

    summary = collector.get_metrics_summary()

    assert "success_rates" in summary
    assert "average_processing_times" in summary
    assert "total_jobs_tracked" in summary
    assert "metrics_timestamp" in summary
    assert summary["total_jobs_tracked"] == 1


def test_job_history_size_limit(collector):
    """Test that job history respects size limit."""
    # Set a small limit for testing
    collector.max_history_size = 5

    # Add more jobs than the limit
    for i in range(10):
        job_metrics = JobMetrics(
            job_type="test",
            start_time=time.time() - i,
            end_time=time.time() - i + 1,
            success=True,
            processing_time=1.0,
        )
        collector._add_to_history(job_metrics)

    # Should only keep the most recent jobs
    assert len(collector.job_history) == 5

    # Should keep the most recent ones (higher start_time values)
    start_times = [job.start_time for job in collector.job_history]
    assert start_times == sorted(start_times, reverse=True)[-5:]


def test_export_prometheus_metrics(collector):
    """Test exporting Prometheus metrics."""
    # Add some metrics
    collector.update_queue_metrics(5, 2, 1, 3)

    metrics_output = collector.export_prometheus_metrics()

    assert isinstance(metrics_output, str)
    assert len(metrics_output) > 0
    # Should contain Prometheus format
    assert "# HELP" in metrics_output or "# TYPE" in metrics_output


def test_job_type_enum():
    """Test JobType enumeration."""
    assert JobType.INDEXING.value == "indexing"
    assert JobType.GRAPHRAG.value == "graphrag"
    assert JobType.ANALYSIS.value == "analysis"
    assert JobType.GLOBAL_QUERY.value == "global_query"
    assert JobType.LOCAL_QUERY.value == "local_query"
    assert JobType.VECTOR_QUERY.value == "vector_query"


def test_global_metrics_collector_instance():
    """Test that global metrics collector instance is available."""
    assert metrics_collector is not None
    assert isinstance(metrics_collector, MetricsCollector)


def test_metrics_with_empty_history(collector):
    """Test metrics calculation with empty history."""
    success_rates = collector.get_job_success_rates()
    avg_times = collector.get_average_processing_times()

    assert success_rates == {}
    assert avg_times == {}


def test_metrics_with_old_jobs(collector):
    """Test that old jobs are excluded from recent calculations."""
    # Add old job (more than 1 hour ago)
    old_job = JobMetrics(
        job_type="indexing",
        start_time=time.time() - 7200,  # 2 hours ago
        end_time=time.time() - 7100,
        success=True,
        processing_time=100.0,
    )
    collector.job_history.append(old_job)

    # Add recent job
    recent_job = JobMetrics(
        job_type="indexing",
        start_time=time.time() - 1800,  # 30 minutes ago
        end_time=time.time() - 1700,
        success=False,
        processing_time=50.0,
    )
    collector.job_history.append(recent_job)

    success_rates = collector.get_job_success_rates()
    avg_times = collector.get_average_processing_times()

    # Should only consider recent job
    assert success_rates["indexing"] == 0.0  # Recent job failed
    assert avg_times["indexing"] == 50.0  # Recent job time


def test_job_metrics_with_batch_metadata(collector):
    """Test job metrics with batch processing metadata."""
    job_metrics = collector.record_job_start("graphrag", "batch_job")

    collector.record_job_completion(
        job_metrics, success=True, batch_size=4, batch_speedup=3.5, chunks_processed=20
    )

    assert job_metrics.metadata["batch_size"] == 4
    assert job_metrics.metadata["batch_speedup"] == 3.5
    assert job_metrics.metadata["chunks_processed"] == 20
