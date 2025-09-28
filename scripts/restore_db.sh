#!/bin/sh

set -e

# Find the most recent backup file
LATEST_BACKUP=$(ls -t /backups/*.sql | head -n 1)

if [ -z "$LATEST_BACKUP" ]; then
    echo "No backup file found."
    exit 1
fi

echo "Restoring database from $LATEST_BACKUP..."
psql -h "$POSTGRES_HOST" -U "$POSTGRES_USER" -d "$POSTGRES_DB" < "$LATEST_BACKUP"
echo "Restore complete."
