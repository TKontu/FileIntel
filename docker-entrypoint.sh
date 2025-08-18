#!/bin/sh -e

# Migrations are run from one place to avoid race conditions.
echo "Running database migrations..."
alembic upgrade head

# Execute the command passed to this script.
echo "Executing command: $@"
exec "$@"
