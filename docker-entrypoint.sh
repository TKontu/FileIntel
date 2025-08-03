#!/bin/sh

# Run the database initialization script
python /home/appuser/app/scripts/init_db.py

# Execute the command passed to this script
exec "$@"
