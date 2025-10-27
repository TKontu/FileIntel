#!/bin/bash
# Monitor memory usage of Celery workers in real-time
# Usage: ./scripts/monitor_worker_memory.sh [interval_seconds]

INTERVAL=${1:-5}  # Default 5 seconds

echo "Monitoring Celery worker memory (interval: ${INTERVAL}s)"
echo "Press Ctrl+C to stop"
echo "---------------------------------------------------"

while true; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')]"

    # Docker stats for celery-worker container
    docker stats --no-stream --format "table {{.Container}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.CPUPerc}}" \
        | grep -E "celery-worker|CONTAINER"

    echo ""

    # Celery worker process details inside container
    docker compose exec -T celery-worker ps aux | head -n 1
    docker compose exec -T celery-worker ps aux | grep -E "celery|python" | grep -v grep

    echo "---------------------------------------------------"
    sleep "$INTERVAL"
done
