#!/usr/bin/env python3
"""
Generate a comprehensive memory report for the FileIntel system.

This script collects memory information from:
- Docker containers
- PostgreSQL database size
- Redis memory usage
- System-level metrics

Usage:
    python scripts/memory_report.py [--format json|text]
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from typing import Dict, Any


def run_command(cmd: list, capture_output: bool = True) -> str:
    """Run a shell command and return output."""
    try:
        if capture_output:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True
            )
            return result.stdout.strip()
        else:
            subprocess.run(cmd, check=True)
            return ""
    except subprocess.CalledProcessError as e:
        return f"Error: {e}"
    except FileNotFoundError:
        return "Command not found"


def get_docker_memory() -> Dict[str, Any]:
    """Get memory usage for Docker containers."""
    cmd = ["docker", "stats", "--no-stream", "--format", "{{.Name}}\t{{.MemUsage}}\t{{.MemPerc}}"]
    output = run_command(cmd)

    containers = {}
    for line in output.split("\n"):
        if line:
            parts = line.split("\t")
            if len(parts) >= 3:
                name = parts[0]
                containers[name] = {
                    "usage": parts[1],
                    "percent": parts[2],
                }

    return containers


def get_postgres_size() -> Dict[str, Any]:
    """Get PostgreSQL database size."""
    cmd = [
        "docker", "compose", "exec", "-T", "postgres",
        "psql", "-U", "user", "-d", "fileintel", "-t", "-c",
        "SELECT pg_size_pretty(pg_database_size('fileintel'));"
    ]
    size = run_command(cmd)

    # Get table sizes
    table_cmd = [
        "docker", "compose", "exec", "-T", "postgres",
        "psql", "-U", "user", "-d", "fileintel", "-t", "-c",
        "SELECT schemaname, tablename, pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size "
        "FROM pg_tables WHERE schemaname = 'public' ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC LIMIT 10;"
    ]
    tables = run_command(table_cmd)

    return {
        "total_size": size.strip(),
        "top_tables": tables,
    }


def get_redis_memory() -> Dict[str, Any]:
    """Get Redis memory usage."""
    cmd = ["docker", "compose", "exec", "-T", "redis", "redis-cli", "INFO", "memory"]
    output = run_command(cmd)

    memory_info = {}
    for line in output.split("\n"):
        if ":" in line and not line.startswith("#"):
            key, value = line.split(":", 1)
            if key.startswith("used_memory"):
                memory_info[key] = value.strip()

    return memory_info


def get_system_memory() -> Dict[str, Any]:
    """Get system memory information."""
    # Try to get host system memory
    free_cmd = ["free", "-h"]
    free_output = run_command(free_cmd)

    return {"free_output": free_output}


def generate_report(format_type: str = "text") -> None:
    """Generate and display memory report."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "docker_containers": get_docker_memory(),
        "postgres": get_postgres_size(),
        "redis": get_redis_memory(),
        "system": get_system_memory(),
    }

    if format_type == "json":
        print(json.dumps(report, indent=2))
    else:
        # Text format
        print("=" * 70)
        print(f"FileIntel Memory Report - {report['timestamp']}")
        print("=" * 70)

        print("\n📦 Docker Container Memory:")
        print("-" * 70)
        for name, info in report["docker_containers"].items():
            print(f"  {name:30s} {info['usage']:20s} {info['percent']}")

        print("\n🗄️  PostgreSQL Database:")
        print("-" * 70)
        print(f"  Total Size: {report['postgres']['total_size']}")
        print("\n  Top 10 Tables:")
        print(report['postgres']['top_tables'])

        print("\n🔴 Redis Memory:")
        print("-" * 70)
        for key, value in report["redis"].items():
            print(f"  {key:30s} {value}")

        print("\n💻 System Memory:")
        print("-" * 70)
        print(report["system"]["free_output"])

        print("\n" + "=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate FileIntel memory report")
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    args = parser.parse_args()

    generate_report(args.format)


if __name__ == "__main__":
    main()
