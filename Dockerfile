# Stage 1: Build the virtual environment
FROM python:3.9-slim as builder

# Install system dependencies required by Python packages
# - libmagic1 is for python-magic
# - tesseract-ocr is for pytesseract
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN pip install poetry

# Set working directory
WORKDIR /app

# Copy dependency definition files
COPY pyproject.toml poetry.lock* ./

# Install dependencies into a virtual environment
# --no-root: Don't install the project itself yet
# --no-dev: Don't install development dependencies
# Poetry will create a poetry.lock file if one doesn't exist
RUN poetry config virtualenvs.in-project true && \
    poetry install --no-root

# Stage 2: Final application image
FROM python:3.9-slim

# Install system dependencies for runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /app/.venv/ .venv/

# Activate the virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Create a non-root user for security
RUN useradd --create-home appuser
USER appuser
WORKDIR /home/appuser/app

# Copy the application source code
COPY ./src ./src
COPY ./config ./config
COPY ./scripts ./scripts
COPY ./docker-entrypoint.sh .
COPY ./tests ./tests

# Create logs directory and set permissions
RUN mkdir logs && chown appuser:appuser logs

# Make the entrypoint script executable
RUN sed -i 's/$//' docker-entrypoint.sh && chmod +x docker-entrypoint.sh

# Expose the port the app runs on
EXPOSE 8000

# Set the entrypoint
ENTRYPOINT ["./docker-entrypoint.sh"]

# Command to run the application
CMD ["uvicorn", "src.document_analyzer.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
