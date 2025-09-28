#!/bin/bash
# Test environment entrypoint script

set -e

echo "FileIntel Test Environment Starting..."

# Wait for database
echo "Waiting for test database..."
while ! pg_isready -h ${DB_HOST} -p ${DB_PORT} -U ${DB_USER} -d ${DB_NAME}; do
    echo "Database not ready, waiting..."
    sleep 2
done
echo "✓ Database ready"

# Wait for Redis
echo "Waiting for Redis..."
while ! redis-cli -h ${REDIS_HOST:-test-redis} -p ${REDIS_PORT:-6379} ping > /dev/null 2>&1; do
    echo "Redis not ready, waiting..."
    sleep 2
done
echo "✓ Redis ready"

# Run database migrations
echo "Running database migrations..."
if [ -f "/app/scripts/run_migrations.py" ]; then
    python /app/scripts/run_migrations.py || echo "Migrations failed or not needed"
else
    echo "No migration script found, skipping..."
fi

# Create test data directory structure
mkdir -p /app/test-data/documents /app/test-data/fixtures /app/test-logs

# Set up test fixtures if they exist
if [ -d "/app/tests/fixtures" ]; then
    echo "Setting up test fixtures..."
    cp -r /app/tests/fixtures/* /app/test-data/fixtures/ 2>/dev/null || echo "No fixtures to copy"
fi

# Check test environment
echo "Checking test environment..."
python -c "
import sys
sys.path.insert(0, '/app/src')

# Test basic imports
try:
    from fileintel.core.config import get_config
    print('✓ Configuration import successful')
except ImportError as e:
    print(f'✗ Configuration import failed: {e}')
    sys.exit(1)

# Test database connection
try:
    import psycopg2
    conn = psycopg2.connect(
        host='${DB_HOST}',
        port='${DB_PORT}',
        database='${DB_NAME}',
        user='${DB_USER}',
        password='${DB_PASSWORD}'
    )
    conn.close()
    print('✓ Database connection successful')
except Exception as e:
    print(f'✗ Database connection failed: {e}')
    sys.exit(1)

print('✓ Test environment ready')
"

# Determine what to run based on arguments
if [ $# -eq 0 ]; then
    echo "No arguments provided. Running default test suite..."
    exec python -m pytest tests/unit/ -v --tb=short
elif [ "$1" = "unit" ]; then
    echo "Running unit tests..."
    exec python -m pytest tests/unit/ -v --tb=short
elif [ "$1" = "integration" ]; then
    echo "Running integration tests..."
    exec python -m pytest tests/integration/ -v --tb=short
elif [ "$1" = "all" ]; then
    echo "Running all tests..."
    exec python -m pytest tests/ -v --tb=short
elif [ "$1" = "coverage" ]; then
    echo "Running tests with coverage..."
    exec python -m pytest tests/ -v --tb=short --cov=src/fileintel --cov-report=html --cov-report=term
elif [ "$1" = "shell" ]; then
    echo "Starting interactive shell..."
    exec /bin/bash
else
    echo "Running custom test pattern: $*"
    exec python -m pytest "$@" -v --tb=short
fi
