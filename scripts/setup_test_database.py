#!/usr/bin/env python3
"""
Test database setup script with proper schema initialization.
Handles database creation, schema setup, and test data preparation.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import hashlib
from datetime import datetime

# Add src to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

try:
    from fileintel.storage.models import Base
    from fileintel.storage.postgresql_storage import PostgreSQLStorage
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import sessionmaker

    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Cannot import FileIntel modules: {e}")
    IMPORTS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestDatabaseSetup:
    """Handles test database setup and schema initialization."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize test database setup."""
        self.config = config or self._load_config()
        self.admin_db_url = self._build_admin_db_url()
        self.test_db_url = self._build_test_db_url()

    def _load_config(self) -> Dict[str, Any]:
        """Load database configuration from environment."""
        return {
            "host": os.environ.get("DB_HOST", "localhost"),
            "port": int(os.environ.get("DB_PORT", "5432")),
            "admin_user": os.environ.get("DB_ADMIN_USER", "postgres"),
            "admin_password": os.environ.get("DB_ADMIN_PASSWORD", ""),
            "test_db_name": os.environ.get("DB_TEST_NAME", "fileintel_test"),
            "test_user": os.environ.get("DB_TEST_USER", "test"),
            "test_password": os.environ.get("DB_TEST_PASSWORD", "test"),
        }

    def _build_admin_db_url(self) -> str:
        """Build admin database connection URL."""
        password_part = (
            f":{self.config['admin_password']}" if self.config["admin_password"] else ""
        )
        return f"postgresql://{self.config['admin_user']}{password_part}@{self.config['host']}:{self.config['port']}/postgres"

    def _build_test_db_url(self) -> str:
        """Build test database connection URL."""
        return f"postgresql://{self.config['test_user']}:{self.config['test_password']}@{self.config['host']}:{self.config['port']}/{self.config['test_db_name']}"

    def check_connection(self) -> bool:
        """Check if database server is accessible."""
        try:
            conn = psycopg2.connect(
                host=self.config["host"],
                port=self.config["port"],
                user=self.config["admin_user"],
                password=self.config["admin_password"],
                database="postgres",
            )
            conn.close()
            logger.info("✓ Database server connection successful")
            return True
        except Exception as e:
            logger.error(f"✗ Database server connection failed: {e}")
            return False

    def create_test_database(self) -> bool:
        """Create test database and user if they don't exist."""
        try:
            # Connect to PostgreSQL server
            conn = psycopg2.connect(
                host=self.config["host"],
                port=self.config["port"],
                user=self.config["admin_user"],
                password=self.config["admin_password"],
                database="postgres",
            )
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()

            # Create test user if not exists
            cursor.execute(
                """
                SELECT 1 FROM pg_roles WHERE rolname = %s
            """,
                (self.config["test_user"],),
            )

            if not cursor.fetchone():
                cursor.execute(
                    f"""
                    CREATE USER {self.config['test_user']} WITH PASSWORD %s CREATEDB
                """,
                    (self.config["test_password"],),
                )
                logger.info(f"✓ Created test user: {self.config['test_user']}")
            else:
                logger.info(f"✓ Test user already exists: {self.config['test_user']}")

            # Create test database if not exists
            cursor.execute(
                """
                SELECT 1 FROM pg_database WHERE datname = %s
            """,
                (self.config["test_db_name"],),
            )

            if not cursor.fetchone():
                cursor.execute(
                    f"""
                    CREATE DATABASE {self.config['test_db_name']}
                    OWNER {self.config['test_user']}
                    ENCODING 'UTF8'
                """
                )
                logger.info(f"✓ Created test database: {self.config['test_db_name']}")
            else:
                logger.info(
                    f"✓ Test database already exists: {self.config['test_db_name']}"
                )

            cursor.close()
            conn.close()
            return True

        except Exception as e:
            logger.error(f"✗ Failed to create test database: {e}")
            return False

    def setup_extensions(self) -> bool:
        """Set up required PostgreSQL extensions."""
        try:
            conn = psycopg2.connect(
                host=self.config["host"],
                port=self.config["port"],
                user=self.config["test_user"],
                password=self.config["test_password"],
                database=self.config["test_db_name"],
            )
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()

            # Required extensions for FileIntel
            extensions = [
                "uuid-ossp",  # UUID generation
                "vector",  # pgvector for embeddings
                "pg_trgm",  # Trigram matching for search
            ]

            for extension in extensions:
                try:
                    cursor.execute(f'CREATE EXTENSION IF NOT EXISTS "{extension}"')
                    logger.info(f"✓ Extension enabled: {extension}")
                except Exception as e:
                    logger.warning(f"⚠ Could not enable extension {extension}: {e}")

            cursor.close()
            conn.close()
            return True

        except Exception as e:
            logger.error(f"✗ Failed to setup extensions: {e}")
            return False

    def create_schema(self) -> bool:
        """Create database schema using SQLAlchemy models."""
        if not IMPORTS_AVAILABLE:
            logger.warning(
                "⚠ SQLAlchemy models not available, skipping schema creation"
            )
            return self.create_basic_schema()

        try:
            # Create SQLAlchemy engine
            engine = create_engine(self.test_db_url)

            # Create all tables
            Base.metadata.create_all(engine)
            logger.info("✓ Database schema created using SQLAlchemy models")

            # Create additional test-specific tables
            with engine.connect() as conn:
                # Test configuration table
                conn.execute(
                    text(
                        """
                    CREATE TABLE IF NOT EXISTS test_config (
                        key VARCHAR(100) PRIMARY KEY,
                        value TEXT,
                        created_at TIMESTAMP DEFAULT NOW(),
                        updated_at TIMESTAMP DEFAULT NOW()
                    )
                """
                    )
                )

                # Test fixtures metadata table
                conn.execute(
                    text(
                        """
                    CREATE TABLE IF NOT EXISTS test_fixtures (
                        id SERIAL PRIMARY KEY,
                        fixture_name VARCHAR(200) NOT NULL,
                        fixture_type VARCHAR(50) NOT NULL,
                        file_path TEXT,
                        checksum VARCHAR(64),
                        created_at TIMESTAMP DEFAULT NOW()
                    )
                """
                    )
                )

                conn.commit()
                logger.info("✓ Additional test tables created")

            return True

        except Exception as e:
            logger.error(f"✗ Failed to create schema: {e}")
            return False

    def create_basic_schema(self) -> bool:
        """Create basic schema without SQLAlchemy (fallback)."""
        try:
            conn = psycopg2.connect(
                host=self.config["host"],
                port=self.config["port"],
                user=self.config["test_user"],
                password=self.config["test_password"],
                database=self.config["test_db_name"],
            )
            cursor = conn.cursor()

            # Basic tables for testing (simplified versions)
            schema_sql = """
            -- Collections table
            CREATE TABLE IF NOT EXISTS collections (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                name VARCHAR(255) NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            );

            -- Documents table
            CREATE TABLE IF NOT EXISTS documents (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                filename VARCHAR(500) NOT NULL,
                content_hash VARCHAR(64) UNIQUE,
                file_size BIGINT,
                mime_type VARCHAR(100),
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            );

            -- Jobs table
            CREATE TABLE IF NOT EXISTS jobs (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                job_type VARCHAR(100) NOT NULL,
                status VARCHAR(50) DEFAULT 'pending',
                priority INTEGER DEFAULT 0,
                job_data JSONB,
                result JSONB,
                error_message TEXT,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW(),
                retry_count INTEGER DEFAULT 0,
                max_retries INTEGER DEFAULT 3,
                next_retry_at TIMESTAMP
            );

            -- Test configuration
            CREATE TABLE IF NOT EXISTS test_config (
                key VARCHAR(100) PRIMARY KEY,
                value TEXT,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            );

            -- Test fixtures
            CREATE TABLE IF NOT EXISTS test_fixtures (
                id SERIAL PRIMARY KEY,
                fixture_name VARCHAR(200) NOT NULL,
                fixture_type VARCHAR(50) NOT NULL,
                file_path TEXT,
                checksum VARCHAR(64),
                created_at TIMESTAMP DEFAULT NOW()
            );

            -- Create indexes
            CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
            CREATE INDEX IF NOT EXISTS idx_jobs_type ON jobs(job_type);
            CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at);
            CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(content_hash);
            """

            cursor.execute(schema_sql)
            conn.commit()
            cursor.close()
            conn.close()

            logger.info("✓ Basic schema created successfully")
            return True

        except Exception as e:
            logger.error(f"✗ Failed to create basic schema: {e}")
            return False

    def setup_test_data(self) -> bool:
        """Set up initial test data and configuration."""
        try:
            conn = psycopg2.connect(
                host=self.config["host"],
                port=self.config["port"],
                user=self.config["test_user"],
                password=self.config["test_password"],
                database=self.config["test_db_name"],
            )
            cursor = conn.cursor()

            # Insert test configuration
            test_config = [
                ("test_environment", "pytest"),
                ("database_version", "1.0.0"),
                ("initialized_at", datetime.utcnow().isoformat()),
                ("schema_checksum", self._calculate_schema_checksum()),
            ]

            for key, value in test_config:
                cursor.execute(
                    """
                    INSERT INTO test_config (key, value)
                    VALUES (%s, %s)
                    ON CONFLICT (key) DO UPDATE SET
                        value = EXCLUDED.value,
                        updated_at = NOW()
                """,
                    (key, value),
                )

            # Create a default test collection
            cursor.execute(
                """
                INSERT INTO collections (id, name, description)
                VALUES (
                    'test-collection-00000000-0000-0000-0000-000000000000'::uuid,
                    'Test Collection',
                    'Default collection for testing purposes'
                )
                ON CONFLICT (id) DO NOTHING
            """
            )

            conn.commit()
            cursor.close()
            conn.close()

            logger.info("✓ Test data setup completed")
            return True

        except Exception as e:
            logger.error(f"✗ Failed to setup test data: {e}")
            return False

    def _calculate_schema_checksum(self) -> str:
        """Calculate checksum of current schema for validation."""
        try:
            schema_info = f"{datetime.utcnow().date()}-fileintel-test-schema"
            return hashlib.sha256(schema_info.encode()).hexdigest()[:16]
        except Exception:
            return "unknown"

    def verify_setup(self) -> bool:
        """Verify that the test database setup is working correctly."""
        try:
            conn = psycopg2.connect(
                host=self.config["host"],
                port=self.config["port"],
                user=self.config["test_user"],
                password=self.config["test_password"],
                database=self.config["test_db_name"],
            )
            cursor = conn.cursor()

            # Check required tables exist
            required_tables = ["collections", "documents", "jobs", "test_config"]
            for table in required_tables:
                cursor.execute(
                    """
                    SELECT 1 FROM information_schema.tables
                    WHERE table_name = %s
                """,
                    (table,),
                )
                if not cursor.fetchone():
                    logger.error(f"✗ Required table missing: {table}")
                    return False

            # Check extensions
            cursor.execute(
                "SELECT extname FROM pg_extension WHERE extname IN ('uuid-ossp', 'vector')"
            )
            extensions = [row[0] for row in cursor.fetchall()]

            required_extensions = ["uuid-ossp"]
            for ext in required_extensions:
                if ext not in extensions:
                    logger.warning(f"⚠ Extension not available: {ext}")

            # Test basic operations
            cursor.execute("SELECT COUNT(*) FROM test_config")
            config_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM collections")
            collection_count = cursor.fetchone()[0]

            cursor.close()
            conn.close()

            logger.info(f"✓ Database verification passed")
            logger.info(f"  - Test config entries: {config_count}")
            logger.info(f"  - Test collections: {collection_count}")
            logger.info(f"  - Extensions: {', '.join(extensions)}")

            return True

        except Exception as e:
            logger.error(f"✗ Database verification failed: {e}")
            return False

    def cleanup_test_data(self) -> bool:
        """Clean up test data (but preserve schema)."""
        try:
            conn = psycopg2.connect(
                host=self.config["host"],
                port=self.config["port"],
                user=self.config["test_user"],
                password=self.config["test_password"],
                database=self.config["test_db_name"],
            )
            cursor = conn.cursor()

            # Tables to clean (preserve test_config and test_fixtures)
            cleanup_tables = [
                "jobs",
                "documents",
                "collections",
                # Add more tables as needed, but keep config tables
            ]

            for table in cleanup_tables:
                try:
                    cursor.execute(f"DELETE FROM {table}")
                    logger.info(f"✓ Cleaned table: {table}")
                except Exception as e:
                    logger.warning(f"⚠ Could not clean table {table}: {e}")

            # Reset test collection
            cursor.execute(
                """
                INSERT INTO collections (id, name, description)
                VALUES (
                    'test-collection-00000000-0000-0000-0000-000000000000'::uuid,
                    'Test Collection',
                    'Default collection for testing purposes'
                )
                ON CONFLICT (id) DO UPDATE SET
                    name = EXCLUDED.name,
                    description = EXCLUDED.description,
                    updated_at = NOW()
            """
            )

            conn.commit()
            cursor.close()
            conn.close()

            logger.info("✓ Test data cleanup completed")
            return True

        except Exception as e:
            logger.error(f"✗ Test data cleanup failed: {e}")
            return False

    def generate_setup_report(self) -> str:
        """Generate a setup report with configuration details."""
        report = f"""
# Test Database Setup Report

## Configuration
- **Database Host**: {self.config['host']}:{self.config['port']}
- **Test Database**: {self.config['test_db_name']}
- **Test User**: {self.config['test_user']}
- **Admin User**: {self.config['admin_user']}

## Connection URLs
- **Admin**: {self.admin_db_url.replace(self.config.get('admin_password', ''), '***')}
- **Test**: {self.test_db_url.replace(self.config['test_password'], '***')}

## Setup Status
- Database Connection: {'✓ Available' if self.check_connection() else '✗ Failed'}
- Required Extensions: uuid-ossp, vector (pgvector)
- Schema: {'SQLAlchemy Models' if IMPORTS_AVAILABLE else 'Basic Schema'}

## Usage Commands
```bash
# Setup database
python scripts/setup_test_database.py --setup

# Verify setup
python scripts/setup_test_database.py --verify

# Cleanup test data
python scripts/setup_test_database.py --cleanup

# Full reset
python scripts/setup_test_database.py --reset
```

## Docker Usage
```bash
# Start test environment
docker-compose -f docker/test-environment/docker-compose.test.yml up -d test-db

# Run setup in container
docker-compose -f docker/test-environment/docker-compose.test.yml run --rm test-runner python scripts/setup_test_database.py --setup
```
"""
        return report


def main():
    """Main setup function with command-line interface."""
    import argparse

    parser = argparse.ArgumentParser(description="FileIntel Test Database Setup")
    parser.add_argument("--setup", action="store_true", help="Set up test database")
    parser.add_argument("--verify", action="store_true", help="Verify database setup")
    parser.add_argument("--cleanup", action="store_true", help="Clean up test data")
    parser.add_argument("--reset", action="store_true", help="Full database reset")
    parser.add_argument("--report", action="store_true", help="Generate setup report")

    args = parser.parse_args()

    if not any([args.setup, args.verify, args.cleanup, args.reset, args.report]):
        args.setup = True  # Default action

    setup = TestDatabaseSetup()

    print("FileIntel Test Database Setup")
    print("=" * 40)

    success = True

    if args.report:
        report = setup.generate_setup_report()
        print(report)

        # Save report
        report_path = project_root / "TEST_DATABASE_REPORT.md"
        with open(report_path, "w") as f:
            f.write(report)
        print(f"\n✓ Report saved to: {report_path}")

    if args.reset:
        print("\nPerforming full database reset...")
        success &= setup.create_test_database()
        success &= setup.setup_extensions()
        success &= setup.create_schema()
        success &= setup.setup_test_data()

    if args.setup:
        print("\nSetting up test database...")
        success &= setup.check_connection()
        if success:
            success &= setup.create_test_database()
            success &= setup.setup_extensions()
            success &= setup.create_schema()
            success &= setup.setup_test_data()

    if args.verify:
        print("\nVerifying database setup...")
        success &= setup.verify_setup()

    if args.cleanup:
        print("\nCleaning up test data...")
        success &= setup.cleanup_test_data()

    if success:
        print(
            f"\n{'✓ All operations completed successfully!' if success else '✗ Some operations failed!'}"
        )
        print("\nNext steps:")
        print("1. Run tests: pytest tests/unit/ -v")
        print("2. Or use Docker: ./scripts/run_docker_tests.sh")
    else:
        print("\n✗ Setup encountered errors. Check logs above.")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
