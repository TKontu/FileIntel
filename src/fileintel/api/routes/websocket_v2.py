"""
WebSocket API v2 - Real-time Celery task monitoring.

Provides real-time task status updates, progress monitoring, and worker status.
"""

import logging
import json
import asyncio
from datetime import datetime
from typing import Dict, Set, Optional, Any
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi.websockets import WebSocketState

from ..dependencies import get_api_key
from ..models import WebSocketEventType, WebSocketTaskEvent, TaskState

# Import Celery functions with error handling for worker availability
try:
    from fileintel.celery_config import (
        get_celery_app,
        get_task_status,
        get_active_tasks,
        get_worker_stats,
    )

    CELERY_AVAILABLE = True
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"Celery configuration not available: {e}")
    CELERY_AVAILABLE = False

    # Provide stub functions to prevent import errors
    def get_celery_app():
        return None

    def get_task_status(task_id):
        return None

    def get_active_tasks():
        return {}

    def get_worker_stats():
        return {}


logger = logging.getLogger(__name__)
router = APIRouter()


class ConnectionManager:
    """Manages WebSocket connections for task monitoring."""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.task_subscriptions: Dict[
            str, Set[str]
        ] = {}  # task_id -> set of connection_ids
        self.connection_filters: Dict[
            str, Dict[str, Any]
        ] = {}  # connection_id -> filters

    async def connect(self, websocket: WebSocket, connection_id: str):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections[connection_id] = websocket
        self.connection_filters[connection_id] = {}
        logger.info(f"WebSocket connection established: {connection_id}")

    def disconnect(self, connection_id: str):
        """Remove a WebSocket connection."""
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]

        if connection_id in self.connection_filters:
            del self.connection_filters[connection_id]

        # Remove from task subscriptions
        for task_id, connections in list(self.task_subscriptions.items()):
            connections.discard(connection_id)
            if not connections:
                del self.task_subscriptions[task_id]

        logger.info(f"WebSocket connection closed: {connection_id}")

    async def send_to_connection(self, connection_id: str, message: dict):
        """Send a message to a specific connection."""
        if connection_id in self.active_connections:
            websocket = self.active_connections[connection_id]
            try:
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_text(json.dumps(message))
                else:
                    self.disconnect(connection_id)
            except Exception as e:
                logger.error(f"Error sending message to {connection_id}: {e}")
                self.disconnect(connection_id)

    async def broadcast_event(self, event: WebSocketTaskEvent):
        """Broadcast an event to all relevant connections."""
        message = {
            "event_type": event.event_type.value,
            "task_id": event.task_id,
            "timestamp": event.timestamp.isoformat(),
            "data": event.data,
            "worker_id": event.worker_id,
        }

        # Send to connections subscribed to this specific task
        if event.task_id in self.task_subscriptions:
            for connection_id in list(self.task_subscriptions[event.task_id]):
                await self.send_to_connection(connection_id, message)

        # Send to connections with matching filters
        for connection_id, filters in self.connection_filters.items():
            if self._event_matches_filters(event, filters):
                await self.send_to_connection(connection_id, message)

    def _event_matches_filters(
        self, event: WebSocketTaskEvent, filters: Dict[str, Any]
    ) -> bool:
        """Check if an event matches the connection's filters."""
        if not filters:
            return True  # No filters means receive all events

        # Filter by event types
        if "event_types" in filters:
            if event.event_type.value not in filters["event_types"]:
                return False

        # Filter by worker
        if "worker_id" in filters:
            if event.worker_id != filters["worker_id"]:
                return False

        # Filter by task type (would need to be added to event data)
        if "task_types" in filters:
            task_type = event.data.get("task_type")
            if task_type and task_type not in filters["task_types"]:
                return False

        return True

    def subscribe_to_task(self, connection_id: str, task_id: str):
        """Subscribe a connection to a specific task."""
        if task_id not in self.task_subscriptions:
            self.task_subscriptions[task_id] = set()
        self.task_subscriptions[task_id].add(connection_id)

    def unsubscribe_from_task(self, connection_id: str, task_id: str):
        """Unsubscribe a connection from a specific task."""
        if task_id in self.task_subscriptions:
            self.task_subscriptions[task_id].discard(connection_id)
            if not self.task_subscriptions[task_id]:
                del self.task_subscriptions[task_id]

    def set_filters(self, connection_id: str, filters: Dict[str, Any]):
        """Set filters for a connection."""
        if connection_id in self.connection_filters:
            self.connection_filters[connection_id] = filters


# Global connection manager
manager = ConnectionManager()


def _map_celery_state_to_event_type(celery_state: str) -> WebSocketEventType:
    """Map Celery task states to WebSocket event types."""
    mapping = {
        "STARTED": WebSocketEventType.TASK_STARTED,
        "PROGRESS": WebSocketEventType.TASK_PROGRESS,
        "SUCCESS": WebSocketEventType.TASK_COMPLETED,
        "FAILURE": WebSocketEventType.TASK_FAILED,
        "RETRY": WebSocketEventType.TASK_RETRY,
        "REVOKED": WebSocketEventType.TASK_COMPLETED,  # Treat revoked as completed
    }
    return mapping.get(celery_state, WebSocketEventType.TASK_PROGRESS)


async def _create_task_event(
    task_id: str, task_info: Dict[str, Any], worker_id: str = None
) -> WebSocketTaskEvent:
    """Create a WebSocket event from Celery task info."""
    event_type = _map_celery_state_to_event_type(task_info.get("state", "PENDING"))

    # Extract relevant data based on task state
    event_data = {
        "status": task_info.get("state", "PENDING"),
        "task_name": task_info.get("name", "unknown"),
    }

    # Add progress information if available
    if task_info.get("state") == "PROGRESS" and "result" in task_info:
        progress = task_info["result"]
        if isinstance(progress, dict):
            event_data.update(
                {
                    "progress": {
                        "current": progress.get("current", 0),
                        "total": progress.get("total", 1),
                        "percentage": progress.get("percentage", 0.0),
                        "message": progress.get("message", ""),
                    }
                }
            )

    # Add result or error information
    if task_info.get("state") == "SUCCESS":
        event_data["result"] = task_info.get("result")
    elif task_info.get("state") == "FAILURE":
        event_data["error"] = str(task_info.get("result", "Unknown error"))

    return WebSocketTaskEvent(
        event_type=event_type,
        task_id=task_id,
        timestamp=datetime.utcnow(),
        data=event_data,
        worker_id=worker_id,
    )


async def _monitor_celery_events():
    """Background task to monitor Celery events and broadcast updates."""
    # This is a simplified monitoring approach
    # In production, you'd use Celery's event monitoring system
    last_active_tasks = {}

    while True:
        try:
            # Check if Celery is available
            if not CELERY_AVAILABLE:
                await asyncio.sleep(5)  # Wait before retrying
                continue

            # Get current active tasks
            active_tasks_data = get_active_tasks()
            current_active_tasks = {}

            # Process active tasks from all workers
            for worker_name, worker_tasks in active_tasks_data.items():
                if not worker_tasks:
                    continue

                for task_info in worker_tasks:
                    task_id = task_info.get("id", "unknown")
                    current_active_tasks[task_id] = {
                        "state": "STARTED",
                        "name": task_info.get("name", "unknown"),
                        "worker": worker_name,
                    }

                    # Check if this is a new task
                    if task_id not in last_active_tasks:
                        event = await _create_task_event(
                            task_id, current_active_tasks[task_id], worker_name
                        )
                        await manager.broadcast_event(event)

            # Check for completed tasks (no longer in active list)
            for task_id in last_active_tasks:
                if task_id not in current_active_tasks:
                    # Task is no longer active, check its final status
                    task_status = get_task_status(task_id)
                    if task_status:
                        event = await _create_task_event(task_id, task_status)
                        await manager.broadcast_event(event)

            last_active_tasks = current_active_tasks

        except Exception as e:
            logger.error(f"Error monitoring Celery events: {e}")

        # Wait before next check
        await asyncio.sleep(2)


# Start background monitoring
_monitoring_task = None


@router.websocket("/tasks/monitor")
async def websocket_task_monitor(
    websocket: WebSocket, api_key: str = Depends(get_api_key)
):
    """
    WebSocket endpoint for real-time task monitoring.

    Provides real-time updates for task status changes, progress, and completion.
    """
    global _monitoring_task

    connection_id = f"conn_{datetime.utcnow().timestamp()}"

    try:
        await manager.connect(websocket, connection_id)

        # Start monitoring if not already running
        if _monitoring_task is None or _monitoring_task.done():
            _monitoring_task = asyncio.create_task(_monitor_celery_events())

        # Send initial connection confirmation
        await manager.send_to_connection(
            connection_id,
            {
                "event_type": "connection_established",
                "connection_id": connection_id,
                "timestamp": datetime.utcnow().isoformat(),
                "message": "WebSocket connection established for task monitoring",
            },
        )

        # Listen for client messages
        while True:
            try:
                message = await websocket.receive_text()
                data = json.loads(message)

                # Handle different message types
                if data.get("type") == "subscribe_task":
                    task_id = data.get("task_id")
                    if task_id:
                        manager.subscribe_to_task(connection_id, task_id)
                        await manager.send_to_connection(
                            connection_id,
                            {
                                "type": "subscription_confirmed",
                                "task_id": task_id,
                                "timestamp": datetime.utcnow().isoformat(),
                            },
                        )

                elif data.get("type") == "unsubscribe_task":
                    task_id = data.get("task_id")
                    if task_id:
                        manager.unsubscribe_from_task(connection_id, task_id)
                        await manager.send_to_connection(
                            connection_id,
                            {
                                "type": "unsubscription_confirmed",
                                "task_id": task_id,
                                "timestamp": datetime.utcnow().isoformat(),
                            },
                        )

                elif data.get("type") == "set_filters":
                    filters = data.get("filters", {})
                    manager.set_filters(connection_id, filters)
                    await manager.send_to_connection(
                        connection_id,
                        {
                            "type": "filters_updated",
                            "filters": filters,
                            "timestamp": datetime.utcnow().isoformat(),
                        },
                    )

                elif data.get("type") == "get_active_tasks":
                    # Send current active tasks
                    if CELERY_AVAILABLE:
                        active_tasks_data = get_active_tasks()
                    else:
                        active_tasks_data = {}
                    await manager.send_to_connection(
                        connection_id,
                        {
                            "type": "active_tasks",
                            "data": active_tasks_data,
                            "timestamp": datetime.utcnow().isoformat(),
                        },
                    )

                elif data.get("type") == "get_worker_stats":
                    # Send worker statistics
                    if CELERY_AVAILABLE:
                        worker_stats = get_worker_stats()
                    else:
                        worker_stats = {}
                    await manager.send_to_connection(
                        connection_id,
                        {
                            "type": "worker_stats",
                            "data": worker_stats,
                            "timestamp": datetime.utcnow().isoformat(),
                        },
                    )

            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                await manager.send_to_connection(
                    connection_id,
                    {
                        "type": "error",
                        "message": "Invalid JSON in message",
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                )
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {e}")
                await manager.send_to_connection(
                    connection_id,
                    {
                        "type": "error",
                        "message": str(e),
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                )

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        manager.disconnect(connection_id)


@router.websocket("/tasks/{task_id}/monitor")
async def websocket_single_task_monitor(
    websocket: WebSocket, task_id: str, api_key: str = Depends(get_api_key)
):
    """
    WebSocket endpoint for monitoring a specific task.

    Provides real-time updates for a single task's status and progress.
    """
    connection_id = f"task_{task_id}_{datetime.utcnow().timestamp()}"

    try:
        await manager.connect(websocket, connection_id)

        # Automatically subscribe to the specific task
        manager.subscribe_to_task(connection_id, task_id)

        # Send initial task status
        if CELERY_AVAILABLE:
            task_status = get_task_status(task_id)
        else:
            task_status = None
        if task_status:
            initial_event = await _create_task_event(task_id, task_status)
            await manager.send_to_connection(
                connection_id,
                {
                    "event_type": initial_event.event_type.value,
                    "task_id": initial_event.task_id,
                    "timestamp": initial_event.timestamp.isoformat(),
                    "data": initial_event.data,
                    "worker_id": initial_event.worker_id,
                },
            )
        else:
            await manager.send_to_connection(
                connection_id,
                {
                    "event_type": "error",
                    "task_id": task_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "message": f"Task {task_id} not found",
                },
            )

        # Keep connection alive and listen for disconnect
        while True:
            try:
                await websocket.receive_text()
            except WebSocketDisconnect:
                break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Single task WebSocket error: {e}")
    finally:
        manager.disconnect(connection_id)
