#!/bin/sh

set -e

# Create the backup directory if it doesn't exist
mkdir -p /backups

# Perform the backup
echo "Backing up database..."
export PGPASSWORD=$POSTGRES_PASSWORD
pg_dump -h "$POSTGRES_HOST" -U "$POSTGRES_USER" -d "$POSTGRES_DB" > "/backups/backup_$(date +%Y-%m-%dT%H:%M:%S).sql"
unset PGPASSWORD
echo "Backup complete."
