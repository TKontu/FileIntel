"""Database migration management system."""

import os
import sys
import hashlib
import importlib.util
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

from fileintel.storage.postgresql_storage import PostgreSQLStorage
from fileintel.core.config import get_config
from sqlalchemy import text

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MigrationManager:
    """Manages database schema migrations with versioning."""

    def __init__(self, storage: PostgreSQLStorage):
        self.storage = storage

        # Check if database is disabled
        if os.environ.get('DISABLE_DATABASE', 'false').lower() == 'true':
            logger.info("Migration manager disabled - database disabled")
            self.disabled = True
            return

        self.disabled = False
        # Adjust path to point to migrations directory from new location
        self.migrations_dir = (
            Path(__file__).parent.parent.parent / "migrations" / "versions"
        )
        self.migrations_dir.mkdir(parents=True, exist_ok=True)

    def create_schema_versions_table(self):
        """Create the schema_versions table if it doesn't exist."""
        # First check if table already exists
        check_table_sql = """
        SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_name = 'schema_versions'
        );
        """

        try:
            result = self.storage.db.execute(text(check_table_sql)).fetchone()
            if result and result[0]:
                logger.info("Schema versions table already exists")
                return

            # Table doesn't exist, create it
            create_table_sql = """
            CREATE TABLE schema_versions (
                version VARCHAR(50) PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                migration_file VARCHAR(255) NOT NULL,
                checksum VARCHAR(64) NOT NULL,
                description TEXT
            );
            """

            self.storage.db.execute(text(create_table_sql))
            self.storage.db.commit()
            logger.info("Schema versions table created successfully")

        except Exception as e:
            logger.error(f"Failed to create schema versions table: {e}")
            self.storage.db.rollback()

            # If we get a type conflict, try to clean it up
            if "duplicate key value violates unique constraint" in str(
                e
            ) and "schema_versions" in str(e):
                logger.info("Attempting to clean up schema_versions type conflict...")
                try:
                    # Drop the composite type if it exists without a table
                    cleanup_sql = """
                    DO $$
                    BEGIN
                        IF NOT EXISTS (
                            SELECT FROM information_schema.tables
                            WHERE table_schema = 'public' AND table_name = 'schema_versions'
                        ) AND EXISTS (
                            SELECT FROM pg_type
                            WHERE typname = 'schema_versions' AND typnamespace = 2200
                        ) THEN
                            DROP TYPE IF EXISTS schema_versions CASCADE;
                        END IF;
                    END
                    $$;
                    """
                    self.storage.db.execute(text(cleanup_sql))
                    self.storage.db.commit()
                    logger.info(
                        "Cleaned up orphaned schema_versions type, retrying table creation..."
                    )

                    # Retry table creation
                    create_table_sql = """
                    CREATE TABLE schema_versions (
                        version VARCHAR(50) PRIMARY KEY,
                        applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        migration_file VARCHAR(255) NOT NULL,
                        checksum VARCHAR(64) NOT NULL,
                        description TEXT
                    );
                    """
                    self.storage.db.execute(text(create_table_sql))
                    self.storage.db.commit()
                    logger.info(
                        "Schema versions table created successfully after cleanup"
                    )

                except Exception as cleanup_error:
                    logger.error(f"Failed to clean up and retry: {cleanup_error}")
                    self.storage.db.rollback()
                    # If cleanup fails, just proceed - the table might exist from the API container
                    logger.warning(
                        "Proceeding despite schema_versions table creation issue - checking if table is functional..."
                    )
                    try:
                        # Test if we can query the table
                        test_sql = "SELECT COUNT(*) FROM schema_versions;"
                        self.storage.db.execute(text(test_sql))
                        logger.info(
                            "Schema versions table is functional despite creation error"
                        )
                        return
                    except Exception as test_error:
                        logger.error(
                            f"Schema versions table is not functional: {test_error}"
                        )
                        raise
            else:
                raise

    def get_current_schema_version(self) -> Optional[str]:
        """Get the current schema version."""
        try:
            result = self.storage.db.execute(
                text(
                    """
                SELECT version FROM schema_versions
                ORDER BY applied_at DESC
                LIMIT 1
            """
                )
            ).fetchone()
            return result[0] if result else None
        except Exception as e:
            logger.warning(f"Could not get current schema version: {e}")
            return None

    def record_migration(
        self, version: str, filename: str, checksum: str, description: str = ""
    ):
        """Record a migration in the schema_versions table."""
        try:
            self.storage.db.execute(
                text(
                    """
                INSERT INTO schema_versions (version, migration_file, checksum, description)
                VALUES (:version, :filename, :checksum, :description)
            """
                ),
                {
                    "version": version,
                    "filename": filename,
                    "checksum": checksum,
                    "description": description,
                },
            )
            self.storage.db.commit()
        except Exception as e:
            logger.error(f"Failed to record migration: {e}")
            self.storage.db.rollback()
            raise

    def calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate MD5 checksum of a file."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def get_migration_files(self) -> List[Dict[str, Any]]:
        """Get all migration files sorted by version."""
        migration_files = []

        for file_path in self.migrations_dir.glob("*.py"):
            if file_path.name.startswith("__"):
                continue

            # Extract version from filename (format: v001_description.py)
            filename = file_path.name
            if not filename.startswith("v"):
                continue

            try:
                version_part = filename.split("_")[0]
                version = version_part[1:]  # Remove 'v' prefix

                migration_files.append(
                    {
                        "version": version,
                        "filename": filename,
                        "path": file_path,
                        "checksum": self.calculate_file_checksum(file_path),
                    }
                )
            except (IndexError, ValueError):
                logger.warning(
                    f"Skipping migration file with invalid format: {filename}"
                )

        # Sort by version number
        migration_files.sort(key=lambda x: int(x["version"]))
        return migration_files

    def get_applied_migrations(self) -> List[Dict[str, Any]]:
        """Get list of applied migrations."""
        try:
            results = self.storage.db.execute(
                text(
                    """
                SELECT version, migration_file, checksum, applied_at, description
                FROM schema_versions
                ORDER BY version
            """
                )
            ).fetchall()

            return [
                {
                    "version": row[0],
                    "filename": row[1],
                    "checksum": row[2],
                    "applied_at": row[3],
                    "description": row[4],
                }
                for row in results
            ]
        except Exception as e:
            logger.warning(f"Could not get applied migrations: {e}")
            return []

    def get_pending_migrations(self) -> List[Dict[str, Any]]:
        """Get migrations that haven't been applied yet."""
        all_migrations = self.get_migration_files()
        applied_migrations = self.get_applied_migrations()
        applied_versions = {m["version"] for m in applied_migrations}

        pending = []
        for migration in all_migrations:
            if migration["version"] not in applied_versions:
                pending.append(migration)

        return pending

    def load_migration_module(self, file_path: Path):
        """Load a migration module dynamically."""
        spec = importlib.util.spec_from_file_location("migration", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def apply_migration(self, migration: Dict[str, Any], dry_run: bool = False) -> bool:
        """Apply a single migration."""
        logger.info(
            f"{'[DRY RUN] ' if dry_run else ''}Applying migration {migration['version']}: {migration['filename']}"
        )

        try:
            # Load migration module
            migration_module = self.load_migration_module(migration["path"])

            if not hasattr(migration_module, "up"):
                logger.error(f"Migration {migration['filename']} missing 'up' function")
                return False

            if dry_run:
                logger.info(f"[DRY RUN] Would apply migration {migration['version']}")
                return True

            # Apply migration in transaction
            try:
                # Execute migration
                migration_module.up(self.storage.db)

                # Record migration
                self.storage.db.execute(
                    text(
                        """
                    INSERT INTO schema_versions (version, migration_file, checksum, description)
                    VALUES (:version, :filename, :checksum, :description)
                """
                    ),
                    {
                        "version": migration["version"],
                        "filename": migration["filename"],
                        "checksum": migration["checksum"],
                        "description": getattr(
                            migration_module,
                            "description",
                            f"Migration {migration['version']}",
                        ),
                    },
                )

                self.storage.db.commit()
                logger.info(f"Successfully applied migration {migration['version']}")
                return True

            except Exception as e:
                self.storage.db.rollback()
                logger.error(f"Failed to apply migration {migration['version']}: {e}")
                raise

        except Exception as e:
            logger.error(f"Error applying migration {migration['version']}: {e}")
            return False

    def rollback_migration(
        self, migration: Dict[str, Any], dry_run: bool = False
    ) -> bool:
        """Rollback a single migration."""
        logger.info(
            f"{'[DRY RUN] ' if dry_run else ''}Rolling back migration {migration['version']}: {migration['filename']}"
        )

        try:
            # Load migration module
            migration_path = self.migrations_dir / migration["filename"]
            migration_module = self.load_migration_module(migration_path)

            if not hasattr(migration_module, "down"):
                logger.error(
                    f"Migration {migration['filename']} missing 'down' function"
                )
                return False

            if dry_run:
                logger.info(
                    f"[DRY RUN] Would rollback migration {migration['version']}"
                )
                return True

            # Rollback migration in transaction
            try:
                # Execute rollback
                migration_module.down(self.storage.db)

                # Remove migration record
                self.storage.db.execute(
                    text(
                        """
                    DELETE FROM schema_versions WHERE version = :version
                """
                    ),
                    {"version": migration["version"]},
                )

                self.storage.db.commit()
                logger.info(
                    f"Successfully rolled back migration {migration['version']}"
                )
                return True

            except Exception as e:
                self.storage.db.rollback()
                logger.error(
                    f"Failed to rollback migration {migration['version']}: {e}"
                )
                raise

        except Exception as e:
            logger.error(f"Error rolling back migration {migration['version']}: {e}")
            return False

    def apply_migrations(self, dry_run: bool = False) -> bool:
        """Apply all pending migrations."""
        self.create_schema_versions_table()

        pending_migrations = self.get_pending_migrations()

        if not pending_migrations:
            logger.info("No pending migrations to apply")
            return True

        logger.info(f"Found {len(pending_migrations)} pending migrations")

        success_count = 0
        for migration in pending_migrations:
            if self.apply_migration(migration, dry_run):
                success_count += 1
            else:
                logger.error(f"Migration failed, stopping at {migration['version']}")
                break

        if dry_run:
            logger.info(
                f"[DRY RUN] Would apply {success_count}/{len(pending_migrations)} migrations"
            )
        else:
            logger.info(f"Applied {success_count}/{len(pending_migrations)} migrations")

        return success_count == len(pending_migrations)

    def rollback_to_version(self, target_version: str, dry_run: bool = False) -> bool:
        """Rollback to a specific version."""
        applied_migrations = self.get_applied_migrations()

        # Find migrations to rollback (in reverse order)
        migrations_to_rollback = []
        for migration in reversed(applied_migrations):
            if migration["version"] > target_version:
                migrations_to_rollback.append(migration)
            else:
                break

        if not migrations_to_rollback:
            logger.info(f"Already at or before version {target_version}")
            return True

        logger.info(
            f"Rolling back {len(migrations_to_rollback)} migrations to version {target_version}"
        )

        success_count = 0
        for migration in migrations_to_rollback:
            if self.rollback_migration(migration, dry_run):
                success_count += 1
            else:
                logger.error(f"Rollback failed, stopping at {migration['version']}")
                break

        if dry_run:
            logger.info(
                f"[DRY RUN] Would rollback {success_count}/{len(migrations_to_rollback)} migrations"
            )
        else:
            logger.info(
                f"Rolled back {success_count}/{len(migrations_to_rollback)} migrations"
            )

        return success_count == len(migrations_to_rollback)

    def get_migration_status(self) -> Dict[str, Any]:
        """Get overall migration status."""
        try:
            self.create_schema_versions_table()

            current_version = self.get_current_schema_version()
            applied_migrations = self.get_applied_migrations()
            pending_migrations = self.get_pending_migrations()

            return {
                "current_version": current_version,
                "total_applied": len(applied_migrations),
                "total_pending": len(pending_migrations),
                "latest_available_version": max(
                    [m["version"] for m in self.get_migration_files()]
                )
                if self.get_migration_files()
                else None,
                "status": "up_to_date"
                if not pending_migrations
                else "migrations_pending",
            }
        except Exception as e:
            return {"error": str(e), "status": "error"}


def main():
    """Main CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Database Migration Manager")
    parser.add_argument(
        "action",
        choices=["status", "apply", "rollback", "create"],
        help="Action to perform",
    )
    parser.add_argument("--version", help="Target version for rollback")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without applying changes",
    )
    parser.add_argument("--name", help="Migration name for create action")

    args = parser.parse_args()

    # Initialize storage
    from fileintel.storage.models import SessionLocal

    db_session = SessionLocal()
    storage = PostgreSQLStorage(db_session)
    manager = MigrationManager(storage)

    try:
        if args.action == "status":
            status = manager.get_migration_status()
            print("Migration Status:")
            print(f"  Current Version: {status.get('current_version', 'None')}")
            print(f"  Applied Migrations: {status.get('total_applied', 0)}")
            print(f"  Pending Migrations: {status.get('total_pending', 0)}")
            print(f"  Status: {status.get('status', 'unknown')}")

            if status.get("total_pending", 0) > 0:
                print("\nPending migrations:")
                for migration in manager.get_pending_migrations():
                    print(f"  - {migration['version']}: {migration['filename']}")

        elif args.action == "apply":
            success = manager.apply_migrations(dry_run=args.dry_run)
            sys.exit(0 if success else 1)

        elif args.action == "rollback":
            if not args.version:
                print("Error: --version required for rollback")
                sys.exit(1)

            success = manager.rollback_to_version(args.version, dry_run=args.dry_run)
            sys.exit(0 if success else 1)

        elif args.action == "create":
            if not args.name:
                print("Error: --name required for create")
                sys.exit(1)

            # Get next version number
            existing_migrations = manager.get_migration_files()
            next_version = 1
            if existing_migrations:
                max_version = max(int(m["version"]) for m in existing_migrations)
                next_version = max_version + 1

            # Create migration file
            filename = f"v{next_version:03d}_{args.name.replace(' ', '_').lower()}.py"
            migration_path = manager.migrations_dir / filename

            template = f'''"""Migration {next_version:03d}: {args.name}

Created: {datetime.now().isoformat()}
"""
from sqlalchemy import text

description = "{args.name}"


def up(session):
    """Apply migration."""
    # Add your migration SQL here
    # Example:
    # session.execute(text("""
    #     CREATE TABLE example (
    #         id SERIAL PRIMARY KEY,
    #         name VARCHAR(255) NOT NULL
    #     );
    # """))
    pass


def down(session):
    """Rollback migration."""
    # Add your rollback SQL here
    # Example:
    # session.execute(text("DROP TABLE IF EXISTS example;"))
    pass
'''

            with open(migration_path, "w") as f:
                f.write(template)

            print(f"Created migration: {migration_path}")

    except KeyboardInterrupt:
        print("\nAborted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
    finally:
        # Close database session
        if "db_session" in locals():
            db_session.close()


if __name__ == "__main__":
    main()
