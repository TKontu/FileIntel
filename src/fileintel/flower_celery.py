"""
Minimal Celery configuration for Flower monitoring.

This completely avoids any imports that could trigger database connections,
providing only basic Celery task monitoring capabilities.
"""

import os
from celery import Celery

# Create minimal Celery app for monitoring only
app = Celery("fileintel")

# Get broker URL from environment
broker_url = os.environ.get('CELERY_BROKER_URL', 'redis://redis:6379/1')
result_backend = os.environ.get('CELERY_RESULT_BACKEND', 'redis://redis:6379/1')

# Minimal configuration for monitoring only
app.conf.update(
    # Use environment variables for broker configuration
    broker_url=broker_url,
    result_backend=result_backend,

    # Basic serialization settings
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',

    # Timezone configuration
    timezone='UTC',
    enable_utc=True,

    # Disable any database-related features
    task_ignore_result=False,  # Allow result monitoring
    task_store_eager_result=False,  # Don't store results immediately

    # Worker control and monitoring settings (required for Flower inspection)
    worker_send_task_events=True,  # Send task events to broker
    worker_enable_remote_control=True,  # Allow remote worker control/inspection
)

# Do NOT auto-discover tasks to avoid importing modules that connect to database
# Flower will still be able to monitor tasks as they appear in the broker