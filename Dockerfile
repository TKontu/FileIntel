FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user and the data directory
RUN useradd --create-home appuser
RUN mkdir /data && chown appuser:appuser /data

# Switch to the non-root user
USER appuser
WORKDIR /home/appuser/app

# Add the user's local bin directory to the PATH
ENV PATH="/home/appuser/.local/bin:$PATH"

# Copy dependency files and install dependencies
COPY --chown=appuser:appuser pyproject.toml ./
RUN pip install "poetry==1.5.1"
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY --chown=appuser:appuser . .

# Make the entrypoint script executable
RUN chmod +x docker-entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["uvicorn", "src.fileintel.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
