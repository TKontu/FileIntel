"""
Comprehensive tests for database schema migration system including
migration execution, rollback, versioning, and integrity checks.
"""

import pytest
import os
import tempfile
import hashlib
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from src.fileintel.scripts.migration_manager import MigrationManager
from src.fileintel.storage.postgresql_storage import PostgreSQLStorage


class TestMigrationManager:
    """Test core migration manager functionality."""

    @pytest.fixture
    def mock_session(self):
        return Mock(spec=Session)

    @pytest.fixture
    def mock_storage(self, mock_session):
        return Mock(spec=PostgreSQLStorage)

    @pytest.fixture
    def migration_manager(self, mock_storage):
        manager = MigrationManager(mock_storage)
        # Create temporary migrations directory
        manager.migrations_dir = Path(tempfile.mkdtemp()) / "migrations"
        manager.migrations_dir.mkdir(parents=True, exist_ok=True)
        return manager

    def test_create_schema_versions_table(self, migration_manager, mock_storage):
        """Test creating schema versions table."""
        mock_storage.db.execute.return_value = None
        mock_storage.db.commit.return_value = None

        migration_manager.create_schema_versions_table()

        mock_storage.db.execute.assert_called_once()
        mock_storage.db.commit.assert_called_once()
        # Verify SQL contains CREATE TABLE statement
        call_args = mock_storage.db.execute.call_args[0][0]
        assert "CREATE TABLE IF NOT EXISTS schema_versions" in str(call_args)

    def test_create_schema_versions_table_error_handling(
        self, migration_manager, mock_storage
    ):
        """Test error handling in schema versions table creation."""
        mock_storage.db.execute.side_effect = SQLAlchemyError("Connection failed")
        mock_storage.db.rollback.return_value = None

        with pytest.raises(SQLAlchemyError):
            migration_manager.create_schema_versions_table()

        mock_storage.db.rollback.assert_called_once()

    def test_get_current_schema_version_exists(self, migration_manager, mock_storage):
        """Test getting current schema version when versions exist."""
        mock_result = Mock()
        mock_result.fetchone.return_value = ("003",)
        mock_storage.db.execute.return_value = mock_result

        version = migration_manager.get_current_schema_version()

        assert version == "003"
        mock_storage.db.execute.assert_called_once()

    def test_get_current_schema_version_none(self, migration_manager, mock_storage):
        """Test getting current schema version when no versions exist."""
        mock_result = Mock()
        mock_result.fetchone.return_value = None
        mock_storage.db.execute.return_value = mock_result

        version = migration_manager.get_current_schema_version()

        assert version is None

    def test_get_current_schema_version_error(self, migration_manager, mock_storage):
        """Test error handling in get current schema version."""
        mock_storage.db.execute.side_effect = SQLAlchemyError("Query failed")

        version = migration_manager.get_current_schema_version()

        assert version is None  # Should return None on error

    def test_record_migration(self, migration_manager, mock_storage):
        """Test recording a migration in schema versions table."""
        version = "004"
        filename = "v004_add_user_table.py"
        checksum = "abc123def456"
        description = "Add user table"

        mock_storage.db.execute.return_value = None
        mock_storage.db.commit.return_value = None

        migration_manager.record_migration(version, filename, checksum, description)

        mock_storage.db.execute.assert_called_once()
        mock_storage.db.commit.assert_called_once()
        # Verify parameters were passed correctly
        call_args = mock_storage.db.execute.call_args
        assert version in str(call_args)
        assert filename in str(call_args)

    def test_record_migration_error_handling(self, migration_manager, mock_storage):
        """Test error handling in record migration."""
        mock_storage.db.execute.side_effect = SQLAlchemyError("Insert failed")
        mock_storage.db.rollback.return_value = None

        with pytest.raises(SQLAlchemyError):
            migration_manager.record_migration("004", "test.py", "checksum", "desc")

        mock_storage.db.rollback.assert_called_once()

    def test_calculate_file_checksum(self, migration_manager):
        """Test calculating file checksum."""
        # Create temporary migration file
        migration_content = """
def up(session):
    session.execute("CREATE TABLE test (id SERIAL PRIMARY KEY);")

def down(session):
    session.execute("DROP TABLE test;")
"""
        migration_file = migration_manager.migrations_dir / "test_migration.py"
        migration_file.write_text(migration_content)

        checksum = migration_manager.calculate_file_checksum(migration_file)

        # Verify checksum is MD5 hash
        expected_checksum = hashlib.md5(migration_content.encode()).hexdigest()
        assert checksum == expected_checksum

    def test_get_migration_files(self, migration_manager):
        """Test getting list of migration files."""
        # Create test migration files
        files = [
            "v001_initial_schema.py",
            "v002_add_collections.py",
            "v003_add_jobs.py",
            "__init__.py",  # Should be ignored
            "__pycache__",  # Should be ignored
        ]

        for filename in files:
            if not filename.startswith("__"):
                content = f'"""Migration {filename}"""\ndef up(session): pass\ndef down(session): pass'
                (migration_manager.migrations_dir / filename).write_text(content)
            else:
                (migration_manager.migrations_dir / filename).touch()

        migration_files = migration_manager.get_migration_files()

        # Should return 3 valid migration files, sorted by version
        assert len(migration_files) == 3
        assert migration_files[0]["version"] == "001"
        assert migration_files[1]["version"] == "002"
        assert migration_files[2]["version"] == "003"

        # Verify each file has required fields
        for migration in migration_files:
            assert "version" in migration
            assert "filename" in migration
            assert "path" in migration
            assert "checksum" in migration

    def test_get_migration_files_invalid_format(self, migration_manager):
        """Test handling of invalid migration file names."""
        # Create files with invalid naming
        invalid_files = ["invalid_name.py", "v_missing_number.py", "123_no_v_prefix.py"]

        for filename in invalid_files:
            (migration_manager.migrations_dir / filename).write_text("def up(): pass")

        migration_files = migration_manager.get_migration_files()

        # Should return empty list as no valid files
        assert len(migration_files) == 0


class TestMigrationExecution:
    """Test migration execution and application."""

    @pytest.fixture
    def mock_session(self):
        return Mock(spec=Session)

    @pytest.fixture
    def mock_storage(self, mock_session):
        storage = Mock(spec=PostgreSQLStorage)
        storage.db = mock_session
        return storage

    @pytest.fixture
    def migration_manager(self, mock_storage):
        manager = MigrationManager(mock_storage)
        manager.migrations_dir = Path(tempfile.mkdtemp()) / "migrations"
        manager.migrations_dir.mkdir(parents=True, exist_ok=True)
        return manager

    def test_apply_single_migration_success(self, migration_manager, mock_storage):
        """Test successful application of single migration."""
        # Create test migration
        migration_content = '''"""Test migration"""
from sqlalchemy import text

description = "Add test table"

def up(session):
    session.execute(text("CREATE TABLE test_table (id SERIAL PRIMARY KEY);"))

def down(session):
    session.execute(text("DROP TABLE test_table;"))
'''
        migration_file = migration_manager.migrations_dir / "v001_test_migration.py"
        migration_file.write_text(migration_content)

        migration = {
            "version": "001",
            "filename": "v001_test_migration.py",
            "path": migration_file,
            "checksum": "test_checksum",
        }

        mock_storage.db.execute.return_value = None
        mock_storage.db.commit.return_value = None

        result = migration_manager.apply_migration(migration)

        assert result is True
        mock_storage.db.execute.assert_called()
        mock_storage.db.commit.assert_called()

    def test_apply_single_migration_dry_run(self, migration_manager, mock_storage):
        """Test dry run mode for migration application."""
        migration_file = migration_manager.migrations_dir / "v001_test.py"
        migration_file.write_text("def up(session): pass")

        migration = {
            "version": "001",
            "filename": "v001_test.py",
            "path": migration_file,
            "checksum": "test",
        }

        result = migration_manager.apply_migration(migration, dry_run=True)

        assert result is True
        # Should not execute any database operations
        mock_storage.db.execute.assert_not_called()
        mock_storage.db.commit.assert_not_called()

    def test_apply_migration_missing_up_function(self, migration_manager, mock_storage):
        """Test migration without 'up' function fails."""
        migration_content = '''"""Invalid migration"""
def down(session):
    pass
'''
        migration_file = migration_manager.migrations_dir / "v001_invalid.py"
        migration_file.write_text(migration_content)

        migration = {
            "version": "001",
            "filename": "v001_invalid.py",
            "path": migration_file,
            "checksum": "test",
        }

        result = migration_manager.apply_migration(migration)

        assert result is False

    def test_apply_migration_execution_error(self, migration_manager, mock_storage):
        """Test migration execution error handling."""
        migration_content = '''"""Error migration"""
from sqlalchemy import text

def up(session):
    raise Exception("Migration execution failed")

def down(session):
    pass
'''
        migration_file = migration_manager.migrations_dir / "v001_error.py"
        migration_file.write_text(migration_content)

        migration = {
            "version": "001",
            "filename": "v001_error.py",
            "path": migration_file,
            "checksum": "test",
        }

        mock_storage.db.rollback.return_value = None

        result = migration_manager.apply_migration(migration)

        assert result is False
        mock_storage.db.rollback.assert_called()

    def test_apply_migrations_multiple(self, migration_manager, mock_storage):
        """Test applying multiple migrations in sequence."""
        # Create multiple migration files
        migrations = []
        for i in range(1, 4):
            content = f'''"""Migration {i:03d}"""
from sqlalchemy import text

def up(session):
    session.execute(text("CREATE TABLE table_{i:03d} (id SERIAL);"))

def down(session):
    session.execute(text("DROP TABLE table_{i:03d};"))
'''
            filename = f"v{i:03d}_migration.py"
            migration_file = migration_manager.migrations_dir / filename
            migration_file.write_text(content)

            migrations.append(
                {
                    "version": f"{i:03d}",
                    "filename": filename,
                    "path": migration_file,
                    "checksum": f"checksum_{i}",
                }
            )

        # Mock pending migrations
        with patch.object(
            migration_manager, "get_pending_migrations", return_value=migrations
        ):
            with patch.object(migration_manager, "create_schema_versions_table"):
                mock_storage.db.execute.return_value = None
                mock_storage.db.commit.return_value = None

                result = migration_manager.apply_migrations()

                assert result is True
                # Should have executed each migration
                assert mock_storage.db.execute.call_count >= len(migrations)

    def test_apply_migrations_partial_failure(self, migration_manager, mock_storage):
        """Test handling of partial migration failure."""
        # Create migrations where second one fails
        migration1_content = '''"""Migration 1"""
from sqlalchemy import text

def up(session):
    session.execute(text("CREATE TABLE success (id SERIAL);"))

def down(session):
    pass
'''
        migration2_content = '''"""Migration 2"""
from sqlalchemy import text

def up(session):
    raise Exception("Forced failure")

def down(session):
    pass
'''

        migration1_file = migration_manager.migrations_dir / "v001_success.py"
        migration1_file.write_text(migration1_content)

        migration2_file = migration_manager.migrations_dir / "v002_failure.py"
        migration2_file.write_text(migration2_content)

        migrations = [
            {
                "version": "001",
                "filename": "v001_success.py",
                "path": migration1_file,
                "checksum": "checksum1",
            },
            {
                "version": "002",
                "filename": "v002_failure.py",
                "path": migration2_file,
                "checksum": "checksum2",
            },
        ]

        with patch.object(
            migration_manager, "get_pending_migrations", return_value=migrations
        ):
            with patch.object(migration_manager, "create_schema_versions_table"):
                # First migration succeeds, second fails
                mock_storage.db.execute.side_effect = [
                    None,
                    Exception("Forced failure"),
                ]
                mock_storage.db.commit.side_effect = [None, Exception("Forced failure")]
                mock_storage.db.rollback.return_value = None

                result = migration_manager.apply_migrations()

                # Should fail overall due to second migration
                assert result is False


class TestMigrationRollback:
    """Test migration rollback functionality."""

    @pytest.fixture
    def mock_session(self):
        return Mock(spec=Session)

    @pytest.fixture
    def mock_storage(self, mock_session):
        storage = Mock(spec=PostgreSQLStorage)
        storage.db = mock_session
        return storage

    @pytest.fixture
    def migration_manager(self, mock_storage):
        manager = MigrationManager(mock_storage)
        manager.migrations_dir = Path(tempfile.mkdtemp()) / "migrations"
        manager.migrations_dir.mkdir(parents=True, exist_ok=True)
        return manager

    def test_rollback_single_migration_success(self, migration_manager, mock_storage):
        """Test successful rollback of single migration."""
        migration_content = '''"""Test migration"""
from sqlalchemy import text

def up(session):
    session.execute(text("CREATE TABLE test (id SERIAL);"))

def down(session):
    session.execute(text("DROP TABLE test;"))
'''
        migration_file = migration_manager.migrations_dir / "v002_test.py"
        migration_file.write_text(migration_content)

        migration = {
            "version": "002",
            "filename": "v002_test.py",
            "path": migration_file,
            "checksum": "test",
        }

        mock_storage.db.execute.return_value = None
        mock_storage.db.commit.return_value = None

        result = migration_manager.rollback_migration(migration)

        assert result is True
        mock_storage.db.execute.assert_called()
        mock_storage.db.commit.assert_called()

    def test_rollback_migration_missing_down_function(
        self, migration_manager, mock_storage
    ):
        """Test rollback fails when 'down' function is missing."""
        migration_content = '''"""Invalid migration"""
def up(session):
    pass
'''
        migration_file = migration_manager.migrations_dir / "v002_invalid.py"
        migration_file.write_text(migration_content)

        migration = {
            "version": "002",
            "filename": "v002_invalid.py",
            "path": migration_file,
            "checksum": "test",
        }

        result = migration_manager.rollback_migration(migration)

        assert result is False

    def test_rollback_migration_execution_error(self, migration_manager, mock_storage):
        """Test rollback execution error handling."""
        migration_content = '''"""Error rollback"""
from sqlalchemy import text

def up(session):
    pass

def down(session):
    raise Exception("Rollback failed")
'''
        migration_file = migration_manager.migrations_dir / "v002_error.py"
        migration_file.write_text(migration_content)

        migration = {
            "version": "002",
            "filename": "v002_error.py",
            "path": migration_file,
            "checksum": "test",
        }

        mock_storage.db.rollback.return_value = None

        result = migration_manager.rollback_migration(migration)

        assert result is False
        mock_storage.db.rollback.assert_called()

    def test_rollback_to_version(self, migration_manager, mock_storage):
        """Test rolling back to specific version."""
        # Create applied migrations
        applied_migrations = [
            {"version": "003", "filename": "v003_latest.py"},
            {"version": "002", "filename": "v002_middle.py"},
            {"version": "001", "filename": "v001_initial.py"},
        ]

        target_version = "001"

        with patch.object(
            migration_manager, "get_applied_migrations", return_value=applied_migrations
        ):
            with patch.object(
                migration_manager, "rollback_migration", return_value=True
            ) as mock_rollback:
                result = migration_manager.rollback_to_version(target_version)

                assert result is True
                # Should rollback 003 and 002, but not 001
                assert mock_rollback.call_count == 2

    def test_rollback_dry_run(self, migration_manager, mock_storage):
        """Test rollback dry run mode."""
        migration_file = migration_manager.migrations_dir / "v002_test.py"
        migration_file.write_text("def up(session): pass\ndef down(session): pass")

        migration = {
            "version": "002",
            "filename": "v002_test.py",
            "path": migration_file,
            "checksum": "test",
        }

        result = migration_manager.rollback_migration(migration, dry_run=True)

        assert result is True
        # Should not execute database operations
        mock_storage.db.execute.assert_not_called()


class TestMigrationVersioning:
    """Test migration versioning and tracking."""

    @pytest.fixture
    def mock_session(self):
        return Mock(spec=Session)

    @pytest.fixture
    def mock_storage(self, mock_session):
        storage = Mock(spec=PostgreSQLStorage)
        storage.db = mock_session
        return storage

    @pytest.fixture
    def migration_manager(self, mock_storage):
        return MigrationManager(mock_storage)

    def test_get_applied_migrations(self, migration_manager, mock_storage):
        """Test getting list of applied migrations."""
        mock_result = Mock()
        mock_result.fetchall.return_value = [
            (
                "001",
                "v001_initial.py",
                "checksum1",
                datetime(2024, 1, 1),
                "Initial schema",
            ),
            ("002", "v002_users.py", "checksum2", datetime(2024, 1, 2), "Add users"),
            ("003", "v003_jobs.py", "checksum3", datetime(2024, 1, 3), "Add jobs"),
        ]
        mock_storage.db.execute.return_value = mock_result

        applied_migrations = migration_manager.get_applied_migrations()

        assert len(applied_migrations) == 3
        assert applied_migrations[0]["version"] == "001"
        assert applied_migrations[1]["version"] == "002"
        assert applied_migrations[2]["version"] == "003"

        # Verify all required fields are present
        for migration in applied_migrations:
            assert "version" in migration
            assert "filename" in migration
            assert "checksum" in migration
            assert "applied_at" in migration
            assert "description" in migration

    def test_get_applied_migrations_error(self, migration_manager, mock_storage):
        """Test error handling in get applied migrations."""
        mock_storage.db.execute.side_effect = SQLAlchemyError("Query failed")

        applied_migrations = migration_manager.get_applied_migrations()

        assert applied_migrations == []

    def test_get_pending_migrations(self, migration_manager, mock_storage):
        """Test getting list of pending migrations."""
        # Mock all migrations
        all_migrations = [
            {"version": "001", "filename": "v001_initial.py"},
            {"version": "002", "filename": "v002_users.py"},
            {"version": "003", "filename": "v003_jobs.py"},
            {"version": "004", "filename": "v004_new.py"},
        ]

        # Mock applied migrations (missing 004)
        applied_migrations = [
            {"version": "001"},
            {"version": "002"},
            {"version": "003"},
        ]

        with patch.object(
            migration_manager, "get_migration_files", return_value=all_migrations
        ):
            with patch.object(
                migration_manager,
                "get_applied_migrations",
                return_value=applied_migrations,
            ):
                pending_migrations = migration_manager.get_pending_migrations()

                assert len(pending_migrations) == 1
                assert pending_migrations[0]["version"] == "004"

    def test_get_migration_status(self, migration_manager, mock_storage):
        """Test getting comprehensive migration status."""
        applied_migrations = [{"version": "001"}, {"version": "002"}]
        pending_migrations = [{"version": "003"}, {"version": "004"}]

        with patch.object(
            migration_manager, "get_current_schema_version", return_value="002"
        ):
            with patch.object(
                migration_manager,
                "get_applied_migrations",
                return_value=applied_migrations,
            ):
                with patch.object(
                    migration_manager,
                    "get_pending_migrations",
                    return_value=pending_migrations,
                ):
                    status = migration_manager.get_migration_status()

                    expected_status = {
                        "current_version": "002",
                        "total_applied": 2,
                        "total_pending": 2,
                        "applied_migrations": applied_migrations,
                        "pending_migrations": pending_migrations,
                    }

                    assert status == expected_status

    def test_checksum_validation(self, migration_manager, mock_storage):
        """Test migration checksum validation."""
        # Create migration file
        migration_content = "def up(session): pass\ndef down(session): pass"
        migration_dir = Path(tempfile.mkdtemp()) / "migrations"
        migration_dir.mkdir(parents=True, exist_ok=True)
        migration_file = migration_dir / "v001_test.py"
        migration_file.write_text(migration_content)

        migration_manager.migrations_dir = migration_dir

        # Calculate expected checksum
        expected_checksum = hashlib.md5(migration_content.encode()).hexdigest()

        # Mock applied migration with different checksum
        applied_migrations = [
            {"version": "001", "filename": "v001_test.py", "checksum": "wrong_checksum"}
        ]

        with patch.object(
            migration_manager, "get_applied_migrations", return_value=applied_migrations
        ):
            validation_results = migration_manager.validate_migration_checksums()

            assert len(validation_results["mismatched"]) == 1
            assert validation_results["mismatched"][0]["version"] == "001"
            assert validation_results["valid"] == 0
            assert validation_results["total"] == 1


class TestMigrationIntegrityChecks:
    """Test migration integrity and validation checks."""

    @pytest.fixture
    def mock_session(self):
        return Mock(spec=Session)

    @pytest.fixture
    def mock_storage(self, mock_session):
        storage = Mock(spec=PostgreSQLStorage)
        storage.db = mock_session
        return storage

    @pytest.fixture
    def migration_manager(self, mock_storage):
        manager = MigrationManager(mock_storage)
        manager.migrations_dir = Path(tempfile.mkdtemp()) / "migrations"
        manager.migrations_dir.mkdir(parents=True, exist_ok=True)
        return manager

    def test_validate_migration_sequence(self, migration_manager):
        """Test validation of migration version sequence."""
        # Create migration files with gap in sequence
        migrations = [
            {"version": "001", "filename": "v001_initial.py"},
            {"version": "002", "filename": "v002_users.py"},
            # Missing 003
            {"version": "004", "filename": "v004_jobs.py"},
            {"version": "005", "filename": "v005_collections.py"},
        ]

        with patch.object(
            migration_manager, "get_migration_files", return_value=migrations
        ):
            validation_result = migration_manager.validate_migration_sequence()

            assert validation_result["valid"] is False
            assert "missing" in validation_result
            assert "003" in validation_result["missing"]

    def test_validate_migration_dependencies(self, migration_manager):
        """Test validation of migration dependencies."""
        # Create migrations with dependency metadata
        migration1_content = '''"""Initial migration"""
dependencies = []

def up(session): pass
def down(session): pass
'''

        migration2_content = '''"""User migration"""
dependencies = ["001"]

def up(session): pass
def down(session): pass
'''

        migration3_content = '''"""Job migration"""
dependencies = ["001", "002"]

def up(session): pass
def down(session): pass
'''

        migration1_file = migration_manager.migrations_dir / "v001_initial.py"
        migration1_file.write_text(migration1_content)

        migration2_file = migration_manager.migrations_dir / "v002_users.py"
        migration2_file.write_text(migration2_content)

        migration3_file = migration_manager.migrations_dir / "v003_jobs.py"
        migration3_file.write_text(migration3_content)

        validation_result = migration_manager.validate_migration_dependencies()

        assert validation_result["valid"] is True
        assert len(validation_result["dependency_graph"]) == 3

    def test_detect_migration_conflicts(self, migration_manager):
        """Test detection of conflicting migrations."""
        # Create migrations that modify same table
        migration1_content = '''"""Add user table"""
from sqlalchemy import text

def up(session):
    session.execute(text("CREATE TABLE users (id SERIAL PRIMARY KEY);"))

def down(session):
    session.execute(text("DROP TABLE users;"))
'''

        migration2_content = '''"""Also add user table"""
from sqlalchemy import text

def up(session):
    session.execute(text("CREATE TABLE users (name VARCHAR(255));"))

def down(session):
    session.execute(text("DROP TABLE users;"))
'''

        migration1_file = migration_manager.migrations_dir / "v001_users.py"
        migration1_file.write_text(migration1_content)

        migration2_file = migration_manager.migrations_dir / "v002_users_conflict.py"
        migration2_file.write_text(migration2_content)

        conflicts = migration_manager.detect_potential_conflicts()

        # Should detect both migrations create 'users' table
        assert len(conflicts) > 0
        assert any("users" in conflict["description"] for conflict in conflicts)

    def test_validate_rollback_safety(self, migration_manager):
        """Test validation of rollback safety."""
        # Create migration with destructive operation
        migration_content = '''"""Destructive migration"""
from sqlalchemy import text

def up(session):
    session.execute(text("DROP TABLE old_data;"))

def down(session):
    # Cannot restore dropped data
    session.execute(text("CREATE TABLE old_data (id SERIAL);"))
'''

        migration_file = migration_manager.migrations_dir / "v001_destructive.py"
        migration_file.write_text(migration_content)

        safety_report = migration_manager.validate_rollback_safety()

        # Should identify potential data loss
        assert len(safety_report["warnings"]) > 0
        assert any("DROP" in warning for warning in safety_report["warnings"])

    def test_migration_performance_analysis(self, migration_manager):
        """Test migration performance analysis."""
        # Create migration with potentially slow operation
        migration_content = '''"""Slow migration"""
from sqlalchemy import text

def up(session):
    # Adding index on large table
    session.execute(text("CREATE INDEX idx_large_table ON big_table (column);"))
    # Full table scan operation
    session.execute(text("UPDATE big_table SET status = 'processed';"))

def down(session):
    session.execute(text("DROP INDEX idx_large_table;"))
'''

        migration_file = migration_manager.migrations_dir / "v001_slow.py"
        migration_file.write_text(migration_content)

        performance_analysis = migration_manager.analyze_migration_performance()

        # Should identify potentially slow operations
        assert len(performance_analysis["slow_operations"]) > 0
        slow_ops = [op["type"] for op in performance_analysis["slow_operations"]]
        assert any(op_type in ["CREATE INDEX", "UPDATE"] for op_type in slow_ops)


class TestMigrationUtilities:
    """Test migration utility functions and helpers."""

    @pytest.fixture
    def mock_session(self):
        return Mock(spec=Session)

    @pytest.fixture
    def mock_storage(self, mock_session):
        storage = Mock(spec=PostgreSQLStorage)
        storage.db = mock_session
        return storage

    @pytest.fixture
    def migration_manager(self, mock_storage):
        manager = MigrationManager(mock_storage)
        manager.migrations_dir = Path(tempfile.mkdtemp()) / "migrations"
        manager.migrations_dir.mkdir(parents=True, exist_ok=True)
        return manager

    def test_create_migration_file(self, migration_manager):
        """Test creating new migration file from template."""
        migration_name = "add_user_authentication"

        # Mock getting next version
        with patch.object(
            migration_manager, "get_next_migration_version", return_value=5
        ):
            migration_path = migration_manager.create_migration_file(migration_name)

            assert migration_path.exists()
            assert migration_path.name == "v005_add_user_authentication.py"

            # Verify file contains template structure
            content = migration_path.read_text()
            assert "def up(session):" in content
            assert "def down(session):" in content
            assert "from sqlalchemy import text" in content
            assert migration_name.replace("_", " ") in content

    def test_get_next_migration_version(self, migration_manager):
        """Test calculating next migration version number."""
        # Mock existing migrations
        existing_migrations = [
            {"version": "001"},
            {"version": "003"},
            {"version": "005"},
        ]

        with patch.object(
            migration_manager, "get_migration_files", return_value=existing_migrations
        ):
            next_version = migration_manager.get_next_migration_version()

            # Should be next sequential number (6)
            assert next_version == 6

    def test_get_next_migration_version_empty(self, migration_manager):
        """Test getting next version when no migrations exist."""
        with patch.object(migration_manager, "get_migration_files", return_value=[]):
            next_version = migration_manager.get_next_migration_version()

            assert next_version == 1

    def test_backup_schema_before_migration(self, migration_manager, mock_storage):
        """Test creating schema backup before migration."""
        with patch("subprocess.run") as mock_subprocess:
            mock_subprocess.return_value.returncode = 0

            backup_path = migration_manager.backup_schema()

            assert backup_path is not None
            assert str(backup_path).endswith(".sql")
            mock_subprocess.assert_called_once()

    def test_restore_schema_from_backup(self, migration_manager, mock_storage):
        """Test restoring schema from backup."""
        backup_file = Path(tempfile.mktemp(suffix=".sql"))
        backup_file.write_text("-- Schema backup\nCREATE TABLE test (id SERIAL);")

        with patch("subprocess.run") as mock_subprocess:
            mock_subprocess.return_value.returncode = 0

            result = migration_manager.restore_schema_from_backup(backup_file)

            assert result is True
            mock_subprocess.assert_called_once()

    def test_migration_logging(self, migration_manager, mock_storage):
        """Test migration execution logging."""
        with patch("logging.getLogger") as mock_logger:
            mock_log = Mock()
            mock_logger.return_value = mock_log

            # Test successful migration logging
            migration = {"version": "001", "filename": "test.py"}

            with patch.object(migration_manager, "apply_migration", return_value=True):
                migration_manager.apply_migrations_with_logging([migration])

                # Verify logging calls
                assert mock_log.info.called
                assert any("001" in str(call) for call in mock_log.info.call_args_list)

    def test_migration_backup_cleanup(self, migration_manager):
        """Test cleanup of old migration backups."""
        # Create test backup files
        backup_dir = Path(tempfile.mkdtemp()) / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)

        old_backup = backup_dir / "schema_backup_old.sql"
        old_backup.write_text("old backup")

        recent_backup = backup_dir / "schema_backup_recent.sql"
        recent_backup.write_text("recent backup")

        # Set old backup timestamp
        old_time = datetime.now().timestamp() - (8 * 24 * 60 * 60)  # 8 days ago
        os.utime(old_backup, (old_time, old_time))

        cleaned_count = migration_manager.cleanup_old_backups(
            backup_dir, max_age_days=7
        )

        assert cleaned_count == 1
        assert not old_backup.exists()
        assert recent_backup.exists()
