#!/usr/bin/env python3
"""Migration rollback tool."""

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


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Rollback database migrations")
    parser.add_argument("target_version", help="Target version to rollback to")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without applying changes",
    )
    parser.add_argument(
        "--confirm", action="store_true", help="Confirm rollback action"
    )

    args = parser.parse_args()

    if not args.confirm and not args.dry_run:
        print("WARNING: This will rollback database migrations!")
        print("Use --confirm to proceed or --dry-run to see what would happen")
        sys.exit(1)

    try:
        # Create database session
        db_session = SessionLocal()
        storage = PostgreSQLStorage(db_session)
        manager = MigrationManager(storage)

        # Show current status
        status = manager.get_migration_status()
        print(f"Current version: {status.get('current_version')}")
        print(f"Target version: {args.target_version}")

        if args.dry_run:
            print("\n[DRY RUN] The following would be rolled back:")

        success = manager.rollback_to_version(args.target_version, dry_run=args.dry_run)
        sys.exit(0 if success else 1)

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
    finally:
        # Close database session
        if "db_session" in locals():
            db_session.close()


if __name__ == "__main__":
    main()
