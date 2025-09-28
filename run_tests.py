#!/usr/bin/env python
"""
Test runner script for FileIntel with GraphRAG.

This script helps run tests with proper environment setup.
"""

import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def setup_test_environment():
    """Setup test environment variables."""
    # Set test environment variables
    test_env = {
        **os.environ,  # Preserve existing environment
        "PYTHONPATH": str(Path(__file__).parent / "src"),
        "DB_HOST": "localhost",
        "OPENAI_API_KEY": "test_key",
        "FILEINTEL_API_KEY": "test_api_key",
        "GRAPHRAG_LLM_MODEL": "gpt-4",
        "GRAPHRAG_EMBEDDING_MODEL": "text-embedding-3-small",
        "GRAPHRAG_INDEX_BASE_PATH": "./test_graphrag_indices",
    }
    return test_env


def run_tests(test_path=None, coverage=False, verbose=False):
    """Run pytest with appropriate settings."""
    cmd = [sys.executable, "-m", "pytest"]

    if coverage:
        cmd.extend(["--cov=src", "--cov-report=html", "--cov-report=term"])

    if verbose:
        cmd.append("-v")

    if test_path:
        cmd.append(test_path)
    else:
        cmd.append("tests/")

    # Add markers for async tests
    cmd.extend(["-m", "not slow"])  # Skip slow tests by default

    env = setup_test_environment()

    print(f"Running: {' '.join(cmd)}")
    print(f"Test environment set up")

    try:
        result = subprocess.run(cmd, env=env, cwd=Path(__file__).parent)
        return result.returncode
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
        return 1
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1


def run_graphrag_tests():
    """Run only GraphRAG-related tests."""
    return run_tests("-m graphrag", verbose=True)


def run_celery_tests():
    """Run only Celery task tests."""
    return run_tests("-m celery", verbose=True)


def run_api_v2_tests():
    """Run only v2 API tests."""
    return run_tests("-m v2_api", verbose=True)


def run_workflow_tests():
    """Run only workflow orchestration tests."""
    return run_tests("-m workflow", verbose=True)


def run_unit_tests():
    """Run unit tests only."""
    return run_tests("tests/unit/", verbose=True)


def run_integration_tests():
    """Run integration tests only."""
    return run_tests("tests/integration/", verbose=True)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "graphrag":
            exit_code = run_graphrag_tests()
        elif command == "celery":
            exit_code = run_celery_tests()
        elif command == "api_v2":
            exit_code = run_api_v2_tests()
        elif command == "workflow":
            exit_code = run_workflow_tests()
        elif command == "unit":
            exit_code = run_unit_tests()
        elif command == "integration":
            exit_code = run_integration_tests()
        elif command == "coverage":
            exit_code = run_tests(coverage=True, verbose=True)
        else:
            print(f"Unknown command: {command}")
            print(
                "Available commands: graphrag, celery, api_v2, workflow, unit, integration, coverage"
            )
            exit_code = 1
    else:
        # Run all tests
        exit_code = run_tests(verbose=True)

    sys.exit(exit_code)
