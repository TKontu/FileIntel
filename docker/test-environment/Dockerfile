# Docker-based test environment for FileIntel
# Provides consistent testing environment across platforms
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry==1.6.1

# Configure Poetry
RUN poetry config virtualenvs.create false \
    && poetry config installer.parallel true

# Create app directory
WORKDIR /app

# Copy poetry files
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry install --no-interaction --no-ansi

# Copy source code
COPY src/ src/
COPY tests/ tests/
COPY scripts/ scripts/
COPY config/ config/
COPY migrations/ migrations/

# Copy test configuration files
COPY docker/test-environment/pytest.ini pytest.ini
COPY docker/test-environment/test-entrypoint.sh test-entrypoint.sh
RUN chmod +x test-entrypoint.sh

# Create test data directories
RUN mkdir -p /app/test-data/documents \
    && mkdir -p /app/test-data/fixtures \
    && mkdir -p /app/test-logs

# Set test environment variables
ENV OPENAI_API_KEY="test-key-placeholder"
ENV FILEINTEL_API_KEY="test-api-key"
ENV DB_HOST="localhost"
ENV DB_PORT="5432"
ENV DB_NAME="fileintel_test"
ENV DB_USER="test"
ENV DB_PASSWORD="test"
ENV LOG_LEVEL="DEBUG"
ENV TESTING="true"

# Expose port for test API
EXPOSE 8000

# Default command runs test suite
CMD ["./test-entrypoint.sh"]
