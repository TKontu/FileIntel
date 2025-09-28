#!/bin/sh -e

# Only run migrations if this service is designated to handle them
if [ "${RUN_MIGRATIONS:-false}" = "true" ]; then
    echo "Running database migrations..."
    python scripts/run_migrations.py

    if [ $? -ne 0 ]; then
        echo "ERROR: Database migrations failed"
        exit 1
    fi

    echo "Database migrations completed successfully"
else
    echo "Skipping database migrations - not designated migration runner"
fi

# Execute the command passed to this script.
echo "Executing command: $@"
exec "$@"
