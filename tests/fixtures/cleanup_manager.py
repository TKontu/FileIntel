"""Test cleanup manager for FileIntel testing."""

import os
import shutil
import tempfile
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timedelta
import psycopg2
import redis
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class TestCleanupManager:
    """Manages cleanup of test resources across database, filesystem, and cache."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize cleanup manager with configuration."""
        self.config = config or self._load_config()
        self.temp_dirs: Set[Path] = set()
        self.temp_files: Set[Path] = set()
        self.db_records_to_cleanup: Dict[str, List[str]] = {
            "collections": [],
            "documents": [],
            "jobs": [],
            "results": [],
            "dead_letter_jobs": [],
        }
        self.redis_keys_to_cleanup: Set[str] = set()
        self.cleanup_enabled = self.config.get("cleanup_enabled", True)

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from environment."""
        return {
            "db_host": os.environ.get("DB_HOST", "localhost"),
            "db_port": int(os.environ.get("DB_PORT", "5432")),
            "db_name": os.environ.get("DB_NAME", "fileintel_test"),
            "db_user": os.environ.get("DB_USER", "test"),
            "db_password": os.environ.get("DB_PASSWORD", "test"),
            "redis_host": os.environ.get("REDIS_HOST", "localhost"),
            "redis_port": int(os.environ.get("REDIS_PORT", "6379")),
            "cleanup_enabled": os.environ.get("TEST_CLEANUP", "true").lower() == "true",
            "max_cleanup_age_hours": int(os.environ.get("MAX_CLEANUP_AGE_HOURS", "24")),
        }

    def register_temp_directory(self, path: Path) -> Path:
        """Register a temporary directory for cleanup."""
        self.temp_dirs.add(path)
        return path

    def register_temp_file(self, path: Path) -> Path:
        """Register a temporary file for cleanup."""
        self.temp_files.add(path)
        return path

    def register_db_record(self, table: str, record_id: str):
        """Register a database record for cleanup."""
        if table in self.db_records_to_cleanup:
            self.db_records_to_cleanup[table].append(record_id)

    def register_redis_key(self, key: str):
        """Register a Redis key for cleanup."""
        self.redis_keys_to_cleanup.add(key)

    def create_temp_directory(self, prefix: str = "fileintel_test_") -> Path:
        """Create and register a temporary directory."""
        temp_dir = Path(tempfile.mkdtemp(prefix=prefix))
        return self.register_temp_directory(temp_dir)

    def create_temp_file(self, suffix: str = ".tmp", prefix: str = "test_") -> Path:
        """Create and register a temporary file."""
        fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix=prefix)
        os.close(fd)
        temp_file = Path(temp_path)
        return self.register_temp_file(temp_file)

    @contextmanager
    def db_connection(self):
        """Context manager for database connections."""
        conn = None
        try:
            conn = psycopg2.connect(
                host=self.config["db_host"],
                port=self.config["db_port"],
                database=self.config["db_name"],
                user=self.config["db_user"],
                password=self.config["db_password"],
            )
            yield conn
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()

    @contextmanager
    def redis_connection(self):
        """Context manager for Redis connections."""
        client = None
        try:
            client = redis.Redis(
                host=self.config["redis_host"],
                port=self.config["redis_port"],
                decode_responses=True,
            )
            # Test connection
            client.ping()
            yield client
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            yield None

    def cleanup_filesystem(self) -> bool:
        """Clean up temporary files and directories."""
        if not self.cleanup_enabled:
            logger.info("Cleanup disabled, skipping filesystem cleanup")
            return True

        success = True

        # Clean up temporary files
        for temp_file in list(self.temp_files):
            try:
                if temp_file.exists():
                    temp_file.unlink()
                    logger.debug(f"Cleaned up temp file: {temp_file}")
                self.temp_files.remove(temp_file)
            except Exception as e:
                logger.error(f"Failed to cleanup temp file {temp_file}: {e}")
                success = False

        # Clean up temporary directories
        for temp_dir in list(self.temp_dirs):
            try:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                    logger.debug(f"Cleaned up temp directory: {temp_dir}")
                self.temp_dirs.remove(temp_dir)
            except Exception as e:
                logger.error(f"Failed to cleanup temp directory {temp_dir}: {e}")
                success = False

        return success

    def cleanup_database(self) -> bool:
        """Clean up database test records."""
        if not self.cleanup_enabled:
            logger.info("Cleanup disabled, skipping database cleanup")
            return True

        success = True

        try:
            with self.db_connection() as conn:
                cursor = conn.cursor()

                # Clean up in reverse dependency order
                table_order = [
                    "dead_letter_jobs",
                    "results",
                    "jobs",
                    "documents",
                    "collections",
                ]

                for table in table_order:
                    if (
                        table in self.db_records_to_cleanup
                        and self.db_records_to_cleanup[table]
                    ):
                        record_ids = self.db_records_to_cleanup[table]

                        try:
                            # Use parameterized query for safety
                            placeholders = ",".join(["%s"] * len(record_ids))
                            cursor.execute(
                                f"DELETE FROM {table} WHERE id IN ({placeholders})",
                                record_ids,
                            )
                            deleted_count = cursor.rowcount

                            if deleted_count > 0:
                                logger.debug(
                                    f"Cleaned up {deleted_count} records from {table}"
                                )

                            self.db_records_to_cleanup[table].clear()

                        except Exception as e:
                            logger.error(f"Failed to cleanup {table}: {e}")
                            success = False

                # Clean up old test data (older than max age)
                if self.config.get("max_cleanup_age_hours"):
                    cutoff_time = datetime.utcnow() - timedelta(
                        hours=self.config["max_cleanup_age_hours"]
                    )

                    cleanup_queries = [
                        ("test_config WHERE key LIKE 'test_%'", []),
                        (
                            "jobs WHERE created_at < %s AND job_type LIKE 'test_%'",
                            [cutoff_time],
                        ),
                        (
                            "collections WHERE name LIKE 'Test %' AND created_at < %s",
                            [cutoff_time],
                        ),
                    ]

                    for query, params in cleanup_queries:
                        try:
                            cursor.execute(f"DELETE FROM {query}", params)
                            if cursor.rowcount > 0:
                                logger.debug(
                                    f"Cleaned up {cursor.rowcount} old records: {query}"
                                )
                        except Exception as e:
                            logger.warning(f"Failed to cleanup old data: {e}")

                conn.commit()

        except Exception as e:
            logger.error(f"Database cleanup failed: {e}")
            success = False

        return success

    def cleanup_redis(self) -> bool:
        """Clean up Redis test keys."""
        if not self.cleanup_enabled:
            logger.info("Cleanup disabled, skipping Redis cleanup")
            return True

        success = True

        try:
            with self.redis_connection() as redis_client:
                if redis_client is None:
                    logger.warning("Redis not available, skipping Redis cleanup")
                    return True

                # Clean up registered keys
                if self.redis_keys_to_cleanup:
                    deleted_count = redis_client.delete(*self.redis_keys_to_cleanup)
                    if deleted_count > 0:
                        logger.debug(f"Cleaned up {deleted_count} Redis keys")
                    self.redis_keys_to_cleanup.clear()

                # Clean up test patterns
                test_patterns = [
                    "test:*",
                    "fileintel:test:*",
                    "cache:test:*",
                    "session:test:*",
                ]

                for pattern in test_patterns:
                    try:
                        keys = redis_client.keys(pattern)
                        if keys:
                            deleted = redis_client.delete(*keys)
                            if deleted > 0:
                                logger.debug(
                                    f"Cleaned up {deleted} Redis keys matching {pattern}"
                                )
                    except Exception as e:
                        logger.warning(
                            f"Failed to cleanup Redis pattern {pattern}: {e}"
                        )

        except Exception as e:
            logger.error(f"Redis cleanup failed: {e}")
            success = False

        return success

    def cleanup_all(self) -> bool:
        """Clean up all registered resources."""
        if not self.cleanup_enabled:
            logger.info("Cleanup disabled globally")
            return True

        logger.info("Starting comprehensive test cleanup...")

        results = []
        results.append(self.cleanup_filesystem())
        results.append(self.cleanup_database())
        results.append(self.cleanup_redis())

        success = all(results)

        if success:
            logger.info("✓ Test cleanup completed successfully")
        else:
            logger.warning("⚠ Test cleanup completed with some errors")

        return success

    def emergency_cleanup(self) -> bool:
        """Emergency cleanup for when tests are aborted."""
        logger.warning("Performing emergency cleanup...")

        # Force cleanup even if disabled
        original_enabled = self.cleanup_enabled
        self.cleanup_enabled = True

        try:
            return self.cleanup_all()
        finally:
            self.cleanup_enabled = original_enabled

    def get_cleanup_summary(self) -> Dict[str, Any]:
        """Get summary of resources scheduled for cleanup."""
        db_total = sum(len(records) for records in self.db_records_to_cleanup.values())

        return {
            "cleanup_enabled": self.cleanup_enabled,
            "temp_directories": len(self.temp_dirs),
            "temp_files": len(self.temp_files),
            "db_records": db_total,
            "db_tables": {
                table: len(records)
                for table, records in self.db_records_to_cleanup.items()
                if records
            },
            "redis_keys": len(self.redis_keys_to_cleanup),
            "config": {
                "max_cleanup_age_hours": self.config.get("max_cleanup_age_hours"),
                "db_host": self.config.get("db_host"),
                "redis_host": self.config.get("redis_host"),
            },
        }

    def setup_test_isolation(self, test_name: str) -> Dict[str, Any]:
        """Set up isolated test environment."""
        test_id = f"test_{test_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        # Create isolated directories
        test_dir = self.create_temp_directory(f"fileintel_{test_id}_")
        data_dir = test_dir / "data"
        logs_dir = test_dir / "logs"
        data_dir.mkdir()
        logs_dir.mkdir()

        # Create test-specific Redis prefix
        redis_prefix = f"fileintel:test:{test_id}"
        self.register_redis_key(f"{redis_prefix}:*")

        isolation_config = {
            "test_id": test_id,
            "test_dir": test_dir,
            "data_dir": data_dir,
            "logs_dir": logs_dir,
            "redis_prefix": redis_prefix,
            "db_prefix": f"test_{test_id}",
        }

        logger.info(f"Set up test isolation for {test_name}: {test_id}")
        return isolation_config


# Pytest fixtures and decorators
import pytest
import functools
import atexit

# Global cleanup manager for session-level cleanup
_global_cleanup_manager = None


def get_global_cleanup_manager() -> TestCleanupManager:
    """Get or create global cleanup manager."""
    global _global_cleanup_manager
    if _global_cleanup_manager is None:
        _global_cleanup_manager = TestCleanupManager()
        # Register emergency cleanup on exit
        atexit.register(_global_cleanup_manager.emergency_cleanup)
    return _global_cleanup_manager


@pytest.fixture(scope="session")
def global_cleanup():
    """Session-level cleanup fixture."""
    cleanup_manager = get_global_cleanup_manager()
    yield cleanup_manager
    cleanup_manager.cleanup_all()


@pytest.fixture(scope="function")
def test_cleanup():
    """Function-level cleanup fixture."""
    cleanup_manager = TestCleanupManager()
    yield cleanup_manager
    cleanup_manager.cleanup_all()


@pytest.fixture(scope="function")
def isolated_test_env(test_cleanup, request):
    """Fixture for isolated test environment."""
    test_name = request.node.name
    return test_cleanup.setup_test_isolation(test_name)


def cleanup_after_test(cleanup_manager: TestCleanupManager = None):
    """Decorator to automatically cleanup after test function."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            manager = cleanup_manager or TestCleanupManager()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                manager.cleanup_all()

        return wrapper

    return decorator


# Context managers for specific cleanup scenarios
@contextmanager
def temporary_test_collection(
    cleanup_manager: TestCleanupManager, collection_data: Dict[str, Any]
):
    """Context manager for temporary test collection."""
    collection_id = None
    try:
        # Create collection (would normally use FileIntel API)
        collection_id = collection_data["id"]
        cleanup_manager.register_db_record("collections", collection_id)
        yield collection_id
    finally:
        if collection_id and cleanup_manager.cleanup_enabled:
            # Cleanup will be handled by cleanup_manager.cleanup_all()
            pass


@contextmanager
def temporary_test_documents(
    cleanup_manager: TestCleanupManager, document_count: int = 3
):
    """Context manager for temporary test documents."""
    from .test_documents import TestDocumentFixtures

    fixtures = TestDocumentFixtures()
    temp_dir = None
    document_ids = []

    try:
        temp_dir = fixtures.write_fixtures_to_disk()
        cleanup_manager.register_temp_directory(temp_dir)

        # Register document IDs for cleanup (would be created by FileIntel)
        for i in range(document_count):
            doc_id = f"test-doc-{i}"
            document_ids.append(doc_id)
            cleanup_manager.register_db_record("documents", doc_id)

        yield {
            "temp_dir": temp_dir,
            "document_ids": document_ids,
            "fixtures_path": temp_dir,
        }

    finally:
        if temp_dir:
            # Cleanup will be handled by cleanup_manager
            pass
