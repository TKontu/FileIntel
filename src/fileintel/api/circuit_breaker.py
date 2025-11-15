"""
Circuit Breaker pattern implementation for external dependencies.

Prevents cascading failures by temporarily blocking requests to
unhealthy services (Redis, databases, etc.).
"""

import time
import logging
from typing import Callable, Any, Optional
from enum import Enum
from functools import wraps
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Blocking requests due to failures
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 5  # Number of failures before opening
    success_threshold: int = 2  # Number of successes needed to close from half-open
    timeout: float = 60.0  # Seconds to wait before trying again (open -> half-open)
    expected_exception: type = Exception  # Exception type that counts as failure


class CircuitBreaker:
    """
    Circuit breaker for protecting against cascading failures.

    Usage:
        breaker = CircuitBreaker("redis", failure_threshold=3, timeout=30)

        @breaker.protect
        def call_redis():
            return redis_client.get("key")

        # Or manually:
        with breaker:
            result = redis_client.get("key")
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        self.name = name
        self.config = CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            success_threshold=success_threshold,
            timeout=timeout,
            expected_exception=expected_exception
        )

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._last_state_change: datetime = datetime.utcnow()

    @property
    def state(self) -> CircuitState:
        """Get current state, automatically transitioning if needed."""
        if self._state == CircuitState.OPEN:
            # Check if timeout expired -> move to half-open
            if self._last_failure_time:
                time_since_failure = (datetime.utcnow() - self._last_failure_time).total_seconds()
                if time_since_failure >= self.config.timeout:
                    self._transition_to(CircuitState.HALF_OPEN)

        return self._state

    def _transition_to(self, new_state: CircuitState):
        """Transition to a new state with logging."""
        old_state = self._state
        self._state = new_state
        self._last_state_change = datetime.utcnow()

        if new_state == CircuitState.OPEN:
            logger.warning(
                f"Circuit breaker '{self.name}' OPENED after {self._failure_count} failures"
            )
        elif new_state == CircuitState.CLOSED:
            logger.info(
                f"Circuit breaker '{self.name}' CLOSED after {self._success_count} successes"
            )
        elif new_state == CircuitState.HALF_OPEN:
            logger.info(
                f"Circuit breaker '{self.name}' HALF-OPEN, testing service"
            )

    def record_success(self):
        """Record a successful operation."""
        self._failure_count = 0  # Reset failures on success

        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.config.success_threshold:
                self._transition_to(CircuitState.CLOSED)
                self._success_count = 0

    def record_failure(self):
        """Record a failed operation."""
        self._failure_count += 1
        self._last_failure_time = datetime.utcnow()
        self._success_count = 0  # Reset successes on failure

        if self._state == CircuitState.HALF_OPEN:
            # Failed during test -> back to OPEN
            self._transition_to(CircuitState.OPEN)
        elif self._failure_count >= self.config.failure_threshold:
            # Too many failures -> OPEN
            self._transition_to(CircuitState.OPEN)

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function through the circuit breaker.

        Raises:
            CircuitBreakerOpenError: If circuit is open
        """
        current_state = self.state  # This may trigger state transition

        if current_state == CircuitState.OPEN:
            raise CircuitBreakerOpenError(
                f"Circuit breaker '{self.name}' is OPEN. "
                f"Service is unavailable. Retry after {self.config.timeout}s."
            )

        try:
            result = func(*args, **kwargs)
            self.record_success()
            return result
        except self.config.expected_exception as e:
            self.record_failure()
            raise

    def protect(self, func: Callable) -> Callable:
        """Decorator to protect a function with circuit breaker."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper

    def __enter__(self):
        """Context manager entry - check if circuit is open."""
        if self.state == CircuitState.OPEN:
            raise CircuitBreakerOpenError(
                f"Circuit breaker '{self.name}' is OPEN"
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - record success/failure."""
        if exc_type is None:
            self.record_success()
        elif isinstance(exc_val, self.config.expected_exception):
            self.record_failure()
        # Don't suppress exceptions
        return False

    def get_stats(self) -> dict:
        """Get circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "last_failure_time": self._last_failure_time.isoformat() if self._last_failure_time else None,
            "last_state_change": self._last_state_change.isoformat(),
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "success_threshold": self.config.success_threshold,
                "timeout": self.config.timeout,
            }
        }


class CircuitBreakerOpenError(Exception):
    """Raised when attempting to call through an open circuit breaker."""
    pass


# Global circuit breakers for common services
_circuit_breakers = {}


def get_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    timeout: float = 60.0
) -> CircuitBreaker:
    """
    Get or create a circuit breaker for a service.

    Args:
        name: Service name (e.g., "redis", "database")
        failure_threshold: Failures before opening circuit
        timeout: Seconds to wait before retrying

    Returns:
        CircuitBreaker instance
    """
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(
            name=name,
            failure_threshold=failure_threshold,
            timeout=timeout
        )
    return _circuit_breakers[name]


def get_all_circuit_breakers() -> dict:
    """Get statistics for all circuit breakers."""
    return {
        name: breaker.get_stats()
        for name, breaker in _circuit_breakers.items()
    }
