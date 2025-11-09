"""
Synchronous Response Handler for Query API.

Enables synchronous request-response pattern while keeping API non-blocking:
- API submits task to Celery queue
- API waits for Redis pub/sub notification (non-blocking asyncio)
- Worker publishes notification when complete
- API fetches result and returns to client through original HTTP connection
"""

import asyncio
import json
import logging
from typing import Optional, Dict, Any
from redis.asyncio import Redis as AsyncRedis
from celery.result import AsyncResult

logger = logging.getLogger(__name__)


class SyncResponseHandler:
    """
    Handles synchronous query responses using Redis pub/sub notifications.

    Architecture:
    1. API calls wait_for_task_completion() with task_id
    2. Subscribes to Redis channel: task:complete:{task_id}
    3. Worker publishes to channel when task completes
    4. API receives notification, fetches result, returns response
    """

    def __init__(self, redis_url: str):
        """
        Initialize sync response handler.

        Args:
            redis_url: Redis connection URL (same as Celery broker)
        """
        self.redis_url = redis_url

    async def wait_for_task_completion(
        self,
        task_id: str,
        timeout: int = 120
    ) -> Optional[Dict[str, Any]]:
        """
        Wait for task completion notification via Redis pub/sub.

        This is non-blocking - uses asyncio, doesn't block the event loop.
        Multiple API instances can each wait for different tasks concurrently.

        Args:
            task_id: Celery task ID to wait for
            timeout: Maximum seconds to wait (default 120s)

        Returns:
            Task result dict if completed within timeout, None if timeout

        Example:
            result = await handler.wait_for_task_completion("abc-123", timeout=60)
            if result:
                return ApiResponse(data=result)
            else:
                return ApiResponse(error="timeout", task_id="abc-123")
        """
        redis_client = None
        pubsub = None

        try:
            # Connect to Redis (async)
            redis_client = await AsyncRedis.from_url(
                self.redis_url,
                decode_responses=True,
                encoding="utf-8"
            )

            # Subscribe to task completion channel
            channel_name = f"task:complete:{task_id}"
            pubsub = redis_client.pubsub()
            await pubsub.subscribe(channel_name)

            logger.info(f"Waiting for task {task_id} (timeout={timeout}s)")

            # Wait for notification with timeout
            try:
                async with asyncio.timeout(timeout):
                    # Listen for messages on subscribed channel
                    async for message in pubsub.listen():
                        if message["type"] == "message":
                            # Task completed - fetch result
                            logger.info(f"Task {task_id} completed, fetching result")
                            result = await self._fetch_task_result(task_id)
                            return result

            except asyncio.TimeoutError:
                logger.warning(
                    f"Task {task_id} timeout after {timeout}s - "
                    "client can poll /tasks/{task_id} for result"
                )
                return None

        except Exception as e:
            logger.error(f"Error waiting for task {task_id}: {e}")
            return None

        finally:
            # Cleanup: unsubscribe and close connection
            if pubsub:
                try:
                    await pubsub.unsubscribe(channel_name)
                    await pubsub.close()
                except Exception as e:
                    logger.warning(f"Error closing pubsub: {e}")

            if redis_client:
                try:
                    await redis_client.close()
                except Exception as e:
                    logger.warning(f"Error closing redis client: {e}")

    async def _fetch_task_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch task result from Celery result backend.

        Args:
            task_id: Celery task ID

        Returns:
            Task result dict or None if not found/failed
        """
        try:
            # Get result from Celery backend
            task_result = AsyncResult(task_id)

            # Check task state
            if task_result.ready():
                if task_result.successful():
                    return task_result.result
                else:
                    # Task failed
                    logger.error(f"Task {task_id} failed: {task_result.info}")
                    return {
                        "error": "task_failed",
                        "details": str(task_result.info)
                    }
            else:
                # Task not ready yet (shouldn't happen if notification sent)
                logger.warning(f"Task {task_id} notification received but not ready")
                return None

        except Exception as e:
            logger.error(f"Error fetching task result {task_id}: {e}")
            return None


async def notify_task_completion(task_id: str, redis_url: str):
    """
    Notify that a task has completed (called from Celery worker).

    Publishes to Redis pub/sub channel that API is listening to.

    Args:
        task_id: Celery task ID that completed
        redis_url: Redis connection URL

    Example (in Celery task):
        @celery_app.task
        def my_task(...):
            result = do_work()
            # Notify API layer
            asyncio.run(notify_task_completion(self.request.id, redis_url))
            return result
    """
    redis_client = None

    try:
        redis_client = await AsyncRedis.from_url(
            redis_url,
            decode_responses=True,
            encoding="utf-8"
        )

        channel_name = f"task:complete:{task_id}"

        # Publish notification
        await redis_client.publish(channel_name, "ready")

        logger.info(f"Published completion notification for task {task_id}")

    except Exception as e:
        logger.error(f"Error publishing task completion for {task_id}: {e}")

    finally:
        if redis_client:
            try:
                await redis_client.close()
            except Exception as e:
                logger.warning(f"Error closing redis client: {e}")


def notify_task_completion_sync(task_id: str, redis_url: str):
    """
    Synchronous wrapper for notify_task_completion.

    Use this from Celery tasks (which may not have async context).

    Args:
        task_id: Celery task ID that completed
        redis_url: Redis connection URL
    """
    try:
        # Run async function in new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(notify_task_completion(task_id, redis_url))
        finally:
            loop.close()
    except Exception as e:
        logger.error(f"Error in sync notification wrapper: {e}")
