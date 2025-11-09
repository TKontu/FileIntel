# FileIntel Testing Guide

This document explains how to run tests in the FileIntel project.

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures and test configuration
├── unit/                    # Unit tests (fast, comprehensive business logic testing)
│   ├── test_query_processor.py
│   ├── test_analysis_processor.py
│   ├── test_job_processor_registry.py
│   └── test_cli_client.py
└── integration/             # Integration tests (require real PostgreSQL setup)
    ├── test_api_endpoints.py
    └── test_worker_integration.py
```

## Running Tests

### Unit Tests (Recommended for Development)

```bash
# Run all unit tests (default)
pytest

# Run specific unit test files
pytest tests/unit/test_query_processor.py -v
pytest tests/unit/test_analysis_processor.py -v
```

**Features:**
- ✅ Fast execution (no database connections)
- ✅ Use mocked dependencies
- ✅ Perfect for TDD and development
- ✅ Run in CI/CD pipelines

### Integration Tests with In-Memory Database

```bash
# Run integration tests with SQLite in-memory database
pytest -m integration -v

# Run specific integration test
pytest tests/integration/test_storage_integration.py -v
```

**Features:**
- ✅ Real database operations (SQLite in-memory)
- ✅ No external PostgreSQL required
- ✅ Test database schema and relationships
- ✅ Fast setup/teardown per test

### Integration Tests with Real PostgreSQL

```bash
# First, set up your .env file with real database credentials
# Then run integration tests that need PostgreSQL
pytest tests/integration/test_api_endpoints.py -v --no-cov

# Or run all integration tests with real DB
pytest -m integration -v --tb=short
```

**Features:**
- ✅ Tests against real PostgreSQL
- ✅ Uses your .env configuration
- ❌ Requires database setup
- ❌ Slower execution

## Database Test Options

### Option 1: In-Memory SQLite (Recommended)

**Best for:** Most database testing needs

```python
def test_create_collection(test_storage):
    """Uses SQLite in-memory database via test_storage fixture"""
    collection = test_storage.create_collection("test_name")
    assert collection.name == "test_name"
```

**Pros:**
- Fast and reliable
- No external dependencies
- Automatic cleanup
- Perfect for CI/CD

**Cons:**
- Not 100% identical to PostgreSQL
- No vector operations (pgvector)

### Option 2: TestContainers (Advanced)

Add this to your test setup for full PostgreSQL testing:

```bash
pip install testcontainers[postgresql]
```

```python
import pytest
from testcontainers.postgres import PostgresContainer

@pytest.fixture(scope="session")
def postgres_container():
    with PostgresContainer("postgres:15") as postgres:
        yield postgres

def test_with_real_postgres(postgres_container):
    # Get connection string from container
    db_url = postgres_container.get_connection_url()
    # Use real PostgreSQL for testing
```

### Option 3: Dedicated Test Database

**Setup:**
1. Create a dedicated PostgreSQL test database
2. Set credentials in your `.env` file:
   ```
   DB_USER=test_user
   DB_PASSWORD=test_password
   DB_HOST=localhost
   DB_PORT=5432
   DB_NAME=fileintel_test
   ```
3. Run integration tests that will use this database

## Environment Configuration

Tests automatically load from your `.env` file:

```env
# .env file at project root
DB_USER=your_db_user
DB_PASSWORD=your_db_password
DB_HOST=localhost
DB_PORT=5432
DB_NAME=your_test_db
OPENAI_API_KEY=your_openai_key
FILEINTEL_API_KEY=your_api_key
```

If `.env` doesn't exist, tests use safe defaults.

## Test Categories

### Unit Tests
- Mock all external dependencies
- Test individual components in isolation
- Fast execution (< 1 second per test)
- No database, no API calls, no file I/O

### Integration Tests
- Test component interactions
- May use real or in-memory databases
- Test API endpoints end-to-end
- Verify data flow between layers

### End-to-End Tests
- Full system testing
- Real database + real API + real files
- Slower but comprehensive
- Best for critical user workflows

## Best Practices

### For Development (Daily Work)
```bash
# Run unit tests frequently
pytest

# Run integration tests before commits
pytest -m integration
```

### For CI/CD Pipelines
```bash
# Fast unit tests
pytest tests/unit/ --cov=src --cov-report=xml

# Integration tests with in-memory DB
pytest tests/integration/ -m integration
```

### For Production Validation
```bash
# Full test suite with real database
pytest tests/ -v --tb=short
```

## Troubleshooting

### "No module named" errors
```bash
# Make sure you're in the virtual environment
source .venv/bin/activate
pip install -e .
```

### Database connection errors
```bash
# For unit tests, should not happen (uses mocks)
pytest tests/unit/

# For integration tests, check your .env file or use in-memory SQLite
pytest tests/integration/test_storage_integration.py
```

### Import errors
```bash
# Clear Python cache and try again
find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true
pytest tests/unit/test_query_processor.py -v
```

## Coverage Reports

```bash
# Generate HTML coverage report
pytest --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html  # or your browser
```

The comprehensive test suite ensures your refactored FileIntel system works correctly at all levels! 🎉
