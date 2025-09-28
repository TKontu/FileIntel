#!/usr/bin/env python3
"""
Standalone test for alerting system without requiring full dependencies.
"""
import sys
import os
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))


def test_alerting_imports():
    """Test that alerting system imports work."""
    try:
        from fileintel.worker.alerting import AlertLevel, AlertRule, Alert, AlertManager

        print("✓ Alerting imports successful")

        # Test enum values
        assert AlertLevel.INFO.value == "info"
        assert AlertLevel.CRITICAL.value == "critical"
        print("✓ AlertLevel enum working")

        # Test dataclass creation
        from datetime import datetime

        alert = Alert(
            rule_name="test_rule",
            level=AlertLevel.WARNING,
            message="Test alert",
            timestamp=datetime.utcnow(),
        )
        assert alert.rule_name == "test_rule"
        assert alert.level == AlertLevel.WARNING
        print("✓ Alert dataclass working")

        return True

    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_metrics_imports():
    """Test that metrics system imports work."""
    try:
        from fileintel.worker.metrics import MetricsCollector, JobMetrics

        print("✓ Metrics imports successful")

        # Test that we can create the collector (may fail due to dependencies)
        try:
            collector = MetricsCollector()
            print("✓ MetricsCollector instantiation successful")
        except Exception as e:
            print(f"! MetricsCollector requires dependencies: {e}")

        return True

    except ImportError as e:
        print(f"✗ Metrics import error: {e}")
        return False


def test_vram_monitoring_imports():
    """Test VRAM monitoring imports."""
    try:
        from fileintel.rag.graph_rag.services.vram_monitor import (
            VRAMMonitor,
            VRAMStats,
            BatchSizeOptimizer,
        )

        print("✓ VRAM monitoring imports successful")
        return True

    except ImportError as e:
        print(f"✗ VRAM monitoring import error: {e}")
        return False


def test_job_manager_imports():
    """Test job manager imports."""
    try:
        # Import the specific classes we created tests for
        from fileintel.worker.job_manager import JobManager
        from fileintel.worker.monitoring import SystemMonitor

        print("✓ Job manager imports successful")
        return True

    except ImportError as e:
        print(f"✗ Job manager import error: {e}")
        return False


def test_llm_connection_pool_imports():
    """Test LLM connection pool imports."""
    try:
        from fileintel.llm_integration.connection_pool import (
            ConnectionPool,
            CircuitBreaker,
            PooledConnection,
        )

        print("✓ LLM connection pool imports successful")
        return True

    except ImportError as e:
        print(f"✗ LLM connection pool import error: {e}")
        return False


def main():
    """Run all standalone tests."""
    print("Running standalone tests for pytest script components...\n")

    results = []
    results.append(test_alerting_imports())
    results.append(test_metrics_imports())
    results.append(test_vram_monitoring_imports())
    results.append(test_job_manager_imports())
    results.append(test_llm_connection_pool_imports())

    successful = sum(results)
    total = len(results)

    print(f"\nResults: {successful}/{total} import tests passed")

    if successful == total:
        print("✓ All imports working - pytest scripts should be structurally correct")
        return 0
    else:
        print("! Some imports failed - may need dependency resolution")
        return 1


if __name__ == "__main__":
    sys.exit(main())
