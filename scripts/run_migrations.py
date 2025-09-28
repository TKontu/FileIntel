#!/usr/bin/env python3
"""Simple migration runner for production deployments."""

import sys
import logging
from pathlib import Path

# Add src and scripts to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from migration_manager import MigrationManager
from fileintel.storage.postgresql_storage import PostgreSQLStorage
from fileintel.storage.models import SessionLocal
from fileintel.core.config import get_config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def apply_migrations():
    """Apply all pending migrations."""
    db_session = None
    try:
        # Create database session
        db_session = SessionLocal()
        storage = PostgreSQLStorage(db_session)
        manager = MigrationManager(storage)

        logger.info("Starting migration process")

        # Get status
        try:
            status = manager.get_migration_status()
            logger.info(f"Current migration status: {status}")

            if status.get("total_pending", 0) == 0:
                logger.info("No pending migrations")
                return True

            # Apply migrations
            success = manager.apply_migrations(dry_run=False)

            if success:
                logger.info("All migrations applied successfully")
                return True
            else:
                logger.error("Some migrations failed")
                return False

        except Exception as migration_error:
            logger.error(f"Migration execution failed: {migration_error}")
            return False

    except Exception as e:
        logger.error(f"Migration process failed: {e}")
        return False
    finally:
        # Close database session
        if db_session:
            try:
                db_session.close()
            except Exception as close_error:
                logger.warning(f"Error closing database session: {close_error}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run database migrations")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without applying changes",
    )
    args = parser.parse_args()

    db_session = None
    try:
        # Create database session
        db_session = SessionLocal()
        storage = PostgreSQLStorage(db_session)
        manager = MigrationManager(storage)

        success = manager.apply_migrations(dry_run=args.dry_run)
        sys.exit(0 if success else 1)

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
    finally:
        # Close database session
        if db_session:
            try:
                db_session.close()
            except Exception as close_error:
                logger.warning(f"Error closing database session: {close_error}")


if __name__ == "__main__":
    main()
