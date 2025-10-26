#!/usr/bin/env python3
"""Test script to verify Celery configuration."""

try:
    from fileintel.celery_config import app

    print("✓ Celery app created successfully:", app.main)

    from fileintel.tasks import BaseFileIntelTask

    print("✓ BaseFileIntelTask imported successfully")

    from fileintel.core.config import get_config

    config = get_config()
    print("✓ Configuration loaded, Celery broker:", config.celery.broker_url)

    print("\n✓ All Celery components imported and configured successfully!")

except ImportError as e:
    print("✗ Import error:", e)
except Exception as e:
    print("✗ Configuration error:", e)
