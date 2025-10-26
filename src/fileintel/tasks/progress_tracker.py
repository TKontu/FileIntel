"""
Workflow progress tracking using Redis for distributed task monitoring.

Provides progress tracking for parallel Celery task groups (chord/group patterns)
to enable clean, aggregated INFO-level logging instead of per-task spam.
"""

import logging
import threading
from typing import Optional, Tuple
from redis import Redis
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)

# Module-level shared Redis client to prevent connection leaks
_shared_redis_client: Optional[Redis] = None
_redis_client_lock = threading.Lock()


def _get_shared_redis_client() -> Redis:
    """
    Get or create shared Redis client (thread-safe singleton pattern).

    Uses double-checked locking to minimize lock overhead while ensuring
    thread-safety during initialization.
    """
    global _shared_redis_client
    if _shared_redis_client is None:
        with _redis_client_lock:
            # Double-check inside lock to prevent race condition
            if _shared_redis_client is None:
                from fileintel.core.config import get_config
                config = get_config()
                _shared_redis_client = Redis.from_url(
                    config.celery.broker_url,
                    decode_responses=True,
                    max_connections=50,  # Connection pooling
                    socket_keepalive=True,
                    socket_keepalive_options={},
                    health_check_interval=30
                )
                logger.debug("Created shared Redis client for progress tracking")
    return _shared_redis_client


class WorkflowProgressTracker:
    """
    Track progress of parallel task groups using Redis atomic operations.

    Enables logging like:
    - "Starting processing of 168 documents"
    - "Progress: 10/168 documents (6%)"
    - "Progress: 20/168 documents (12%)"
    - "Completed: 168/168 documents (100%)"

    Instead of logging every single task completion.
    """

    def __init__(self, redis_client: Optional[Redis] = None):
        """
        Initialize progress tracker with Redis client.

        Args:
            redis_client: Optional Redis client. If not provided, uses shared singleton.
        """
        if redis_client is None:
            self.redis = _get_shared_redis_client()
        else:
            self.redis = redis_client

    def _total_key(self, workflow_id: str) -> str:
        """Get Redis key for total count."""
        return f"workflow:{workflow_id}:total"

    def _current_key(self, workflow_id: str) -> str:
        """Get Redis key for current count."""
        return f"workflow:{workflow_id}:current"

    def _last_logged_key(self, workflow_id: str) -> str:
        """Get Redis key for last logged count."""
        return f"workflow:{workflow_id}:last_logged"

    def _seen_tasks_key(self, workflow_id: str) -> str:
        """Get Redis key for seen task IDs set (idempotency)."""
        return f"workflow:{workflow_id}:seen_tasks"

    def _validate_workflow_id(self, workflow_id: str) -> str:
        """
        Validate and sanitize workflow_id for Redis key safety.

        Args:
            workflow_id: Workflow identifier to validate

        Returns:
            Sanitized workflow_id

        Raises:
            ValueError: If workflow_id is invalid
        """
        if not workflow_id:
            raise ValueError("workflow_id cannot be empty")

        # Replace problematic characters that could break Redis keys
        safe_id = workflow_id.replace(':', '_').replace(' ', '_').replace('\n', '_')

        if len(safe_id) > 200:  # Reasonable key length limit
            raise ValueError(f"workflow_id too long: {len(safe_id)} chars (max 200)")

        return safe_id

    def initialize(self, workflow_id: str, total_count: int, ttl: int = 3600) -> None:
        """
        Initialize progress tracking for a workflow.

        Args:
            workflow_id: Unique workflow identifier (usually Celery task ID)
            total_count: Total number of tasks in the workflow
            ttl: Time-to-live for Redis keys in seconds (default: 1 hour)
        """
        try:
            # Validate workflow_id to prevent Redis key corruption
            workflow_id = self._validate_workflow_id(workflow_id)

            pipeline = self.redis.pipeline()
            pipeline.set(self._total_key(workflow_id), total_count, ex=ttl)
            pipeline.set(self._current_key(workflow_id), 0, ex=ttl)
            pipeline.set(self._last_logged_key(workflow_id), 0, ex=ttl)
            pipeline.execute()

            logger.debug(f"Initialized progress tracking for workflow {workflow_id}: 0/{total_count}")
        except RedisError as e:
            logger.warning(f"Failed to initialize progress tracking: {e}")

    def increment(
        self,
        workflow_id: str,
        task_id: Optional[str] = None,
        log_interval: int = 10,
        log_percentage: int = 10,
        ttl: int = 3600
    ) -> Tuple[int, int, bool]:
        """
        Increment progress counter and determine if should log (atomic operation).

        Idempotent: If task_id is provided and has been seen before, returns current
        state without incrementing (handles Celery task retries).

        Logs when either condition is met:
        - Every N tasks completed (log_interval)
        - Every N% progress (log_percentage)

        Uses Lua script for atomic increment + check + update to prevent race conditions.

        Args:
            workflow_id: Unique workflow identifier
            task_id: Optional task ID for idempotency (prevents double-counting on retry)
            log_interval: Log every N tasks (default: 10)
            log_percentage: Log every N% progress (default: 10)
            ttl: Refresh TTL on keys (default: 3600 seconds)

        Returns:
            Tuple of (current_count, total_count, should_log)
        """
        try:
            # Validate workflow_id to prevent Redis key corruption
            workflow_id = self._validate_workflow_id(workflow_id)

            # Lua script for atomic increment, check, and conditional update with idempotency
            # Returns: current, total, should_log (1 or 0)
            lua_script = """
            local current_key = KEYS[1]
            local total_key = KEYS[2]
            local last_logged_key = KEYS[3]
            local seen_tasks_key = KEYS[4]
            local log_interval = tonumber(ARGV[1])
            local log_percentage = tonumber(ARGV[2])
            local ttl = tonumber(ARGV[3])
            local task_id = ARGV[4]

            -- Check if we've already seen this task (idempotency for retries)
            if task_id ~= "" and redis.call('SISMEMBER', seen_tasks_key, task_id) == 1 then
                -- Already counted, return current state without incrementing
                local current = tonumber(redis.call('GET', current_key) or 0)
                local total = tonumber(redis.call('GET', total_key) or 0)
                return {current, total, 0}
            end

            -- Mark task as seen if task_id provided
            if task_id ~= "" then
                redis.call('SADD', seen_tasks_key, task_id)
                redis.call('EXPIRE', seen_tasks_key, ttl)
            end

            -- Atomic increment
            local current = redis.call('INCR', current_key)
            redis.call('EXPIRE', current_key, ttl)

            local total = tonumber(redis.call('GET', total_key) or 0)
            local last_logged = tonumber(redis.call('GET', last_logged_key) or 0)

            if total == 0 then
                return {current, 0, 0}
            end

            local should_log = 0

            -- Always log completion
            if current == total then
                should_log = 1
            -- Log every N tasks
            elseif (current - last_logged) >= log_interval then
                should_log = 1
            else
                -- Log every N% progress
                local current_pct = (current / total) * 100
                local last_logged_pct = (last_logged / total) * 100
                if math.floor(current_pct / log_percentage) > math.floor(last_logged_pct / log_percentage) then
                    should_log = 1
                end
            end

            -- Update last logged count if we're logging (atomic)
            if should_log == 1 then
                redis.call('SET', last_logged_key, current, 'EX', ttl)
            end

            return {current, total, should_log}
            """

            result = self.redis.eval(
                lua_script,
                4,  # Number of keys
                self._current_key(workflow_id),
                self._total_key(workflow_id),
                self._last_logged_key(workflow_id),
                self._seen_tasks_key(workflow_id),
                log_interval,
                log_percentage,
                ttl,
                task_id or ""  # Empty string if no task_id provided
            )

            current, total, should_log = int(result[0]), int(result[1]), bool(result[2])

            if total == 0:
                logger.warning(f"Progress tracking not initialized for workflow {workflow_id}")

            return current, total, should_log

        except RedisError as e:
            logger.warning(f"Failed to increment progress: {e}")
            return 0, 0, False

    def get_progress(self, workflow_id: str) -> Tuple[int, int]:
        """
        Get current progress without incrementing.

        Args:
            workflow_id: Unique workflow identifier

        Returns:
            Tuple of (current_count, total_count)
        """
        try:
            current = int(self.redis.get(self._current_key(workflow_id)) or 0)
            total = int(self.redis.get(self._total_key(workflow_id)) or 0)
            return current, total
        except RedisError as e:
            logger.warning(f"Failed to get progress: {e}")
            return 0, 0

    def cleanup(self, workflow_id: str) -> None:
        """
        Clean up Redis keys for a completed workflow.

        Args:
            workflow_id: Unique workflow identifier
        """
        try:
            self.redis.delete(
                self._total_key(workflow_id),
                self._current_key(workflow_id),
                self._last_logged_key(workflow_id),
                self._seen_tasks_key(workflow_id)
            )
            logger.debug(f"Cleaned up progress tracking for workflow {workflow_id}")
        except RedisError as e:
            logger.warning(f"Failed to cleanup progress tracking: {e}")
