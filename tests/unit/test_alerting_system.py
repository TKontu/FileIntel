"""
Comprehensive tests for AlertManager and alerting system functionality.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from uuid import uuid4

from fileintel.worker.alerting import (
    AlertManager,
    AlertRule,
    Alert,
    AlertLevel,
)
from fileintel.storage.base import StorageInterface


@pytest.fixture
def mock_storage():
    """Mock storage interface for alerting tests."""
    storage = Mock(spec=StorageInterface)
    storage.get_dead_letter_queue_size = Mock(return_value=5)
    storage.get_jobs_older_than = Mock(return_value=[])
    storage.get_workers_without_heartbeat_since = Mock(return_value=[])
    return storage


@pytest.fixture
def alert_manager(mock_storage):
    """AlertManager instance with mock storage."""
    return AlertManager(mock_storage)


@pytest.fixture
def sample_alert():
    """Sample alert for testing."""
    return Alert(
        rule_name="test_rule",
        level=AlertLevel.WARNING,
        message="Test alert message",
        timestamp=datetime.utcnow(),
        metadata={"test_key": "test_value"},
    )


class TestAlertRule:
    """Test AlertRule data structure and functionality."""

    def test_alert_rule_creation(self):
        """Test creating an alert rule."""
        condition_func = Mock(return_value=True)

        rule = AlertRule(
            name="test_rule",
            description="Test alert rule",
            level=AlertLevel.ERROR,
            condition_func=condition_func,
            threshold_value=10.0,
            cooldown_minutes=15,
        )

        assert rule.name == "test_rule"
        assert rule.description == "Test alert rule"
        assert rule.level == AlertLevel.ERROR
        assert rule.condition_func == condition_func
        assert rule.threshold_value == 10.0
        assert rule.cooldown_minutes == 15
        assert rule.enabled is True
        assert rule.last_triggered is None

    def test_alert_rule_defaults(self):
        """Test alert rule default values."""
        condition_func = Mock(return_value=False)

        rule = AlertRule(
            name="minimal_rule",
            description="Minimal rule",
            level=AlertLevel.INFO,
            condition_func=condition_func,
        )

        assert rule.threshold_value is None
        assert rule.cooldown_minutes == 30
        assert rule.enabled is True
        assert rule.metadata == {}


class TestAlert:
    """Test Alert data structure."""

    def test_alert_creation(self):
        """Test creating an alert."""
        timestamp = datetime.utcnow()
        metadata = {"source": "test", "value": 42}

        alert = Alert(
            rule_name="cpu_high",
            level=AlertLevel.CRITICAL,
            message="CPU usage is 95%",
            timestamp=timestamp,
            metadata=metadata,
        )

        assert alert.rule_name == "cpu_high"
        assert alert.level == AlertLevel.CRITICAL
        assert alert.message == "CPU usage is 95%"
        assert alert.timestamp == timestamp
        assert alert.metadata == metadata

    def test_alert_default_metadata(self):
        """Test alert with default metadata."""
        alert = Alert(
            rule_name="simple_alert",
            level=AlertLevel.INFO,
            message="Simple message",
            timestamp=datetime.utcnow(),
        )

        assert alert.metadata == {}


class TestAlertManagerInitialization:
    """Test AlertManager initialization and setup."""

    def test_alert_manager_initialization(self, alert_manager):
        """Test alert manager initialization."""
        assert alert_manager.storage is not None
        assert alert_manager.config is not None
        assert len(alert_manager.rules) == 3  # dead_letter, stuck_jobs, dead_workers
        assert alert_manager.dead_letter_threshold == 10
        assert alert_manager.stuck_job_timeout_minutes == 30
        assert alert_manager.monitoring_enabled is True

    def test_default_alert_rules_setup(self, alert_manager):
        """Test that default alert rules are properly set up."""
        # Check dead letter queue rule
        dlq_rule = alert_manager.rules["dead_letter_queue_size"]
        assert dlq_rule.name == "dead_letter_queue_size"
        assert dlq_rule.level == AlertLevel.ERROR
        assert dlq_rule.threshold_value == 10
        assert dlq_rule.cooldown_minutes == 15

        # Check stuck jobs rule
        stuck_rule = alert_manager.rules["stuck_jobs"]
        assert stuck_rule.name == "stuck_jobs"
        assert stuck_rule.level == AlertLevel.WARNING
        assert stuck_rule.threshold_value == 30
        assert stuck_rule.cooldown_minutes == 10

        # Check dead workers rule
        dead_worker_rule = alert_manager.rules["dead_workers"]
        assert dead_worker_rule.name == "dead_workers"
        assert dead_worker_rule.level == AlertLevel.CRITICAL
        assert dead_worker_rule.threshold_value == 2
        assert dead_worker_rule.cooldown_minutes == 5

    def test_alert_handlers_initialization(self, alert_manager):
        """Test alert handlers are initialized."""
        assert len(alert_manager.alert_handlers) == 1  # Default log handler
        assert alert_manager._log_alert in alert_manager.alert_handlers


class TestAlertConditions:
    """Test alert condition checking functions."""

    def test_check_dead_letter_queue_size_above_threshold(
        self, alert_manager, mock_storage
    ):
        """Test dead letter queue size check when above threshold."""
        mock_storage.get_dead_letter_queue_size.return_value = (
            15  # Above threshold of 10
        )

        result = alert_manager._check_dead_letter_queue_size()

        assert result is True
        mock_storage.get_dead_letter_queue_size.assert_called_once()

    def test_check_dead_letter_queue_size_below_threshold(
        self, alert_manager, mock_storage
    ):
        """Test dead letter queue size check when below threshold."""
        mock_storage.get_dead_letter_queue_size.return_value = (
            5  # Below threshold of 10
        )

        result = alert_manager._check_dead_letter_queue_size()

        assert result is False

    def test_check_dead_letter_queue_size_error_handling(
        self, alert_manager, mock_storage
    ):
        """Test error handling in dead letter queue size check."""
        mock_storage.get_dead_letter_queue_size.side_effect = Exception(
            "Database error"
        )

        result = alert_manager._check_dead_letter_queue_size()

        assert result is False  # Should return False on error

    def test_check_stuck_jobs_found(self, alert_manager, mock_storage):
        """Test stuck jobs check when stuck jobs exist."""
        stuck_jobs = [Mock(id=str(uuid4())), Mock(id=str(uuid4()))]
        mock_storage.get_jobs_older_than.return_value = stuck_jobs

        result = alert_manager._check_stuck_jobs()

        assert result is True
        mock_storage.get_jobs_older_than.assert_called_once()

    def test_check_stuck_jobs_none_found(self, alert_manager, mock_storage):
        """Test stuck jobs check when no stuck jobs exist."""
        mock_storage.get_jobs_older_than.return_value = []

        result = alert_manager._check_stuck_jobs()

        assert result is False

    def test_check_dead_workers_found(self, alert_manager, mock_storage):
        """Test dead workers check when dead workers exist."""
        dead_workers = [Mock(worker_id="worker-1"), Mock(worker_id="worker-2")]
        mock_storage.get_workers_without_heartbeat_since.return_value = dead_workers

        result = alert_manager._check_dead_workers()

        assert result is True

    def test_check_dead_workers_none_found(self, alert_manager, mock_storage):
        """Test dead workers check when no dead workers exist."""
        mock_storage.get_workers_without_heartbeat_since.return_value = []

        result = alert_manager._check_dead_workers()

        assert result is False


class TestAlertTriggering:
    """Test alert triggering and handling."""

    @pytest.mark.asyncio
    async def test_trigger_alert_basic(self, alert_manager):
        """Test basic alert triggering."""
        rule = alert_manager.rules["dead_letter_queue_size"]
        timestamp = datetime.utcnow()

        with patch.object(alert_manager, "_generate_alert_message") as mock_message:
            mock_message.return_value = "Test alert message"

            with patch.object(alert_manager, "_get_alert_metadata") as mock_metadata:
                mock_metadata.return_value = {"test": "data"}

                await alert_manager._trigger_alert(rule, timestamp)

        # Check rule state was updated
        assert rule.last_triggered == timestamp

        # Check alert was added to history
        assert len(alert_manager.alert_history) == 1
        alert = alert_manager.alert_history[0]
        assert alert.rule_name == rule.name
        assert alert.level == rule.level
        assert alert.timestamp == timestamp

    @pytest.mark.asyncio
    async def test_trigger_alert_with_handlers(self, alert_manager):
        """Test alert triggering with custom handlers."""
        # Add custom handler
        custom_handler = Mock()
        alert_manager.add_alert_handler(custom_handler)

        rule = alert_manager.rules["stuck_jobs"]

        with patch.object(alert_manager, "_generate_alert_message") as mock_message:
            mock_message.return_value = "Stuck jobs detected"

            with patch.object(alert_manager, "_get_alert_metadata") as mock_metadata:
                mock_metadata.return_value = {"count": 3}

                await alert_manager._trigger_alert(rule, datetime.utcnow())

        # Custom handler should have been called
        custom_handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_trigger_alert_handler_error(self, alert_manager):
        """Test alert triggering with handler that raises exception."""
        # Add handler that raises exception
        error_handler = Mock(side_effect=Exception("Handler error"))
        alert_manager.add_alert_handler(error_handler)

        rule = alert_manager.rules["dead_workers"]

        with patch.object(alert_manager, "_generate_alert_message") as mock_message:
            mock_message.return_value = "Dead workers detected"

            with patch.object(alert_manager, "_get_alert_metadata") as mock_metadata:
                mock_metadata.return_value = {}

                # Should not raise exception despite handler error
                await alert_manager._trigger_alert(rule, datetime.utcnow())

        # Alert should still be recorded
        assert len(alert_manager.alert_history) == 1

    def test_generate_alert_message_dead_letter_queue(
        self, alert_manager, mock_storage
    ):
        """Test alert message generation for dead letter queue."""
        rule = alert_manager.rules["dead_letter_queue_size"]
        mock_storage.get_dead_letter_queue_size.return_value = 15

        message = alert_manager._generate_alert_message(rule)

        assert "15" in message
        assert "10" in message  # threshold
        assert "Dead letter queue size" in message

    def test_generate_alert_message_stuck_jobs(self, alert_manager, mock_storage):
        """Test alert message generation for stuck jobs."""
        rule = alert_manager.rules["stuck_jobs"]
        stuck_jobs = [Mock(id=str(uuid4())), Mock(id=str(uuid4()))]
        mock_storage.get_jobs_older_than.return_value = stuck_jobs

        message = alert_manager._generate_alert_message(rule)

        assert "2" in message  # job count
        assert "30 minutes" in message  # timeout
        assert "jobs running longer" in message

    def test_get_alert_metadata_dead_letter_queue(self, alert_manager, mock_storage):
        """Test alert metadata generation for dead letter queue."""
        rule = alert_manager.rules["dead_letter_queue_size"]
        mock_storage.get_dead_letter_queue_size.return_value = 12

        metadata = alert_manager._get_alert_metadata(rule)

        assert metadata["rule_threshold"] == 10
        assert metadata["current_dlq_size"] == 12
        assert metadata["rule_cooldown_minutes"] == 15


class TestAlertMonitoring:
    """Test alert monitoring functionality."""

    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self, alert_manager):
        """Test starting and stopping monitoring."""
        # Initially no task
        assert alert_manager.monitoring_task is None

        # Start monitoring
        await alert_manager.start_monitoring()
        assert alert_manager.monitoring_task is not None

        # Stop monitoring
        await alert_manager.stop_monitoring()
        assert alert_manager.monitoring_task is None

    @pytest.mark.asyncio
    async def test_check_all_alerts_no_triggers(self, alert_manager, mock_storage):
        """Test checking all alerts when no conditions are met."""
        # Set up storage to return values below thresholds
        mock_storage.get_dead_letter_queue_size.return_value = 5  # Below threshold
        mock_storage.get_jobs_older_than.return_value = []  # No stuck jobs
        mock_storage.get_workers_without_heartbeat_since.return_value = (
            []
        )  # No dead workers

        await alert_manager.check_all_alerts()

        # No alerts should be triggered
        assert len(alert_manager.alert_history) == 0

    @pytest.mark.asyncio
    async def test_check_all_alerts_with_triggers(self, alert_manager, mock_storage):
        """Test checking all alerts when conditions are met."""
        # Set up storage to return values above thresholds
        mock_storage.get_dead_letter_queue_size.return_value = 15  # Above threshold
        mock_storage.get_jobs_older_than.return_value = [
            Mock(id=str(uuid4()))
        ]  # Stuck job
        mock_storage.get_workers_without_heartbeat_since.return_value = (
            []
        )  # No dead workers

        with patch.object(alert_manager, "_generate_alert_message") as mock_message:
            mock_message.return_value = "Alert triggered"

            with patch.object(alert_manager, "_get_alert_metadata") as mock_metadata:
                mock_metadata.return_value = {}

                await alert_manager.check_all_alerts()

        # Two alerts should be triggered (dead letter queue + stuck jobs)
        assert len(alert_manager.alert_history) == 2

    @pytest.mark.asyncio
    async def test_check_all_alerts_cooldown_period(self, alert_manager, mock_storage):
        """Test alert cooldown period functionality."""
        # Set up condition that would trigger alert
        mock_storage.get_dead_letter_queue_size.return_value = 15

        rule = alert_manager.rules["dead_letter_queue_size"]

        # Set last triggered to recent time (within cooldown)
        rule.last_triggered = datetime.utcnow() - timedelta(
            minutes=5
        )  # 5 min ago, cooldown is 15 min

        with patch.object(alert_manager, "_generate_alert_message") as mock_message:
            mock_message.return_value = "Alert message"

            await alert_manager.check_all_alerts()

        # Should not trigger alert due to cooldown
        assert len(alert_manager.alert_history) == 0

    @pytest.mark.asyncio
    async def test_check_all_alerts_disabled_rule(self, alert_manager, mock_storage):
        """Test that disabled rules don't trigger alerts."""
        # Disable a rule
        rule = alert_manager.rules["dead_letter_queue_size"]
        rule.enabled = False

        # Set up condition that would trigger alert if enabled
        mock_storage.get_dead_letter_queue_size.return_value = 15

        await alert_manager.check_all_alerts()

        # No alerts should be triggered
        assert len(alert_manager.alert_history) == 0


class TestAlertHandlers:
    """Test alert handler management."""

    def test_add_alert_handler(self, alert_manager):
        """Test adding custom alert handler."""
        custom_handler = Mock()
        initial_count = len(alert_manager.alert_handlers)

        alert_manager.add_alert_handler(custom_handler)

        assert len(alert_manager.alert_handlers) == initial_count + 1
        assert custom_handler in alert_manager.alert_handlers

    def test_remove_alert_handler(self, alert_manager):
        """Test removing alert handler."""
        custom_handler = Mock()
        alert_manager.add_alert_handler(custom_handler)
        initial_count = len(alert_manager.alert_handlers)

        alert_manager.remove_alert_handler(custom_handler)

        assert len(alert_manager.alert_handlers) == initial_count - 1
        assert custom_handler not in alert_manager.alert_handlers

    def test_remove_nonexistent_handler(self, alert_manager):
        """Test removing handler that doesn't exist."""
        nonexistent_handler = Mock()
        initial_count = len(alert_manager.alert_handlers)

        # Should not raise exception
        alert_manager.remove_alert_handler(nonexistent_handler)

        assert len(alert_manager.alert_handlers) == initial_count

    def test_log_alert_handler(self, alert_manager, sample_alert):
        """Test default log alert handler."""
        with patch("fileintel.worker.alerting.logger") as mock_logger:
            alert_manager._log_alert(sample_alert)

            # Should log at warning level for WARNING alert
            mock_logger.warning.assert_called_once()

    def test_call_handler_safely_sync(self, alert_manager, sample_alert):
        """Test calling synchronous handler safely."""
        sync_handler = Mock()

        # Should not raise exception
        asyncio.run(alert_manager._call_handler_safely(sync_handler, sample_alert))

        sync_handler.assert_called_once_with(sample_alert)

    @pytest.mark.asyncio
    async def test_call_handler_safely_async(self, alert_manager, sample_alert):
        """Test calling asynchronous handler safely."""
        async_handler = AsyncMock()

        await alert_manager._call_handler_safely(async_handler, sample_alert)

        async_handler.assert_called_once_with(sample_alert)


class TestAlertConfiguration:
    """Test alert rule configuration."""

    def test_configure_rule_valid_attributes(self, alert_manager):
        """Test configuring rule with valid attributes."""
        rule_name = "dead_letter_queue_size"

        alert_manager.configure_rule(
            rule_name, threshold_value=20, cooldown_minutes=5, enabled=False
        )

        rule = alert_manager.rules[rule_name]
        assert rule.threshold_value == 20
        assert rule.cooldown_minutes == 5
        assert rule.enabled is False

    def test_configure_rule_invalid_rule_name(self, alert_manager):
        """Test configuring nonexistent rule."""
        with pytest.raises(ValueError, match="Unknown alert rule"):
            alert_manager.configure_rule("nonexistent_rule", threshold_value=10)

    def test_configure_rule_invalid_attribute(self, alert_manager):
        """Test configuring rule with invalid attribute."""
        rule_name = "stuck_jobs"

        # Should not raise exception, just log warning
        alert_manager.configure_rule(rule_name, invalid_attribute="value")

        # Rule should remain unchanged
        rule = alert_manager.rules[rule_name]
        assert not hasattr(rule, "invalid_attribute")


class TestAlertStatus:
    """Test alert status and history functionality."""

    def test_get_alert_history(self, alert_manager):
        """Test getting alert history."""
        # Add some alerts to history
        alerts = [
            Alert("rule1", AlertLevel.INFO, "Message 1", datetime.utcnow()),
            Alert("rule2", AlertLevel.WARNING, "Message 2", datetime.utcnow()),
            Alert("rule3", AlertLevel.ERROR, "Message 3", datetime.utcnow()),
        ]
        alert_manager.alert_history.extend(alerts)

        history = alert_manager.get_alert_history(limit=2)

        assert len(history) == 2
        assert history == alerts[-2:]  # Should return last 2

    def test_get_alert_history_empty(self, alert_manager):
        """Test getting alert history when empty."""
        history = alert_manager.get_alert_history()

        assert history == []

    def test_get_alert_summary(self, alert_manager):
        """Test getting alert system summary."""
        # Add alerts to history
        alerts = [
            Alert("rule1", AlertLevel.INFO, "Info message", datetime.utcnow()),
            Alert("rule2", AlertLevel.WARNING, "Warning message", datetime.utcnow()),
            Alert("rule3", AlertLevel.ERROR, "Error message", datetime.utcnow()),
            Alert("rule4", AlertLevel.WARNING, "Another warning", datetime.utcnow()),
        ]
        alert_manager.alert_history.extend(alerts)

        summary = alert_manager.get_alert_summary()

        assert summary["monitoring_enabled"] is True
        assert summary["total_rules"] == 3
        assert summary["enabled_rules"] == 3
        assert summary["recent_alerts"] == 4
        assert summary["alerts_by_level"]["info"] == 1
        assert summary["alerts_by_level"]["warning"] == 2
        assert summary["alerts_by_level"]["error"] == 1
        assert summary["alert_handlers_count"] == 1


class TestAlertHistoryManagement:
    """Test alert history size management."""

    def test_alert_history_size_limit(self, alert_manager):
        """Test that alert history respects size limit."""
        # Set a small history size for testing
        alert_manager.max_history_size = 3

        # Add more alerts than the limit
        for i in range(5):
            alert = Alert(
                f"rule_{i}", AlertLevel.INFO, f"Message {i}", datetime.utcnow()
            )
            alert_manager.alert_history.append(alert)

            # Manually trigger size limit (normally done in _trigger_alert)
            if len(alert_manager.alert_history) > alert_manager.max_history_size:
                alert_manager.alert_history.pop(0)

        # Should only keep the last 3 alerts
        assert len(alert_manager.alert_history) == 3
        assert alert_manager.alert_history[0].message == "Message 2"
        assert alert_manager.alert_history[-1].message == "Message 4"
