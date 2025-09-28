#!/usr/bin/env python3
"""
Test environment setup script for Poetry configuration across WSL/Windows.
Handles cross-platform compatibility and dependency resolution.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
import json
import shutil


class TestEnvironmentSetup:
    """Handles test environment setup with cross-platform Poetry compatibility."""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.is_wsl = self._detect_wsl()
        self.poetry_cmd = self._find_poetry_command()
        self.python_cmd = self._find_python_command()

    def _detect_wsl(self) -> bool:
        """Detect if running in WSL environment."""
        try:
            with open("/proc/version", "r") as f:
                return "microsoft" in f.read().lower() or "wsl" in f.read().lower()
        except (FileNotFoundError, OSError):
            return False

    def _find_poetry_command(self) -> str:
        """Find the correct Poetry command for the current environment."""
        if self.is_wsl:
            # WSL specific Poetry path
            wsl_poetry = "/home/linux/.local/bin/poetry"
            if Path(wsl_poetry).exists():
                return wsl_poetry

        # Try standard Poetry commands
        for cmd in ["poetry", "python -m poetry", "python3 -m poetry"]:
            try:
                subprocess.run(
                    [cmd.split()[0], "--version"], capture_output=True, check=True
                )
                return cmd
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue

        raise RuntimeError("Poetry not found. Please install Poetry.")

    def _find_python_command(self) -> str:
        """Find the correct Python command for the current environment."""
        if self.is_wsl:
            return "/usr/bin/python3"

        for cmd in ["python", "python3"]:
            try:
                result = subprocess.run(
                    [cmd, "--version"], capture_output=True, check=True, text=True
                )
                if "Python 3." in result.stdout:
                    return cmd
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue

        raise RuntimeError("Python 3 not found.")

    def check_environment(self) -> dict:
        """Check current test environment status."""
        status = {
            "platform": platform.system(),
            "is_wsl": self.is_wsl,
            "poetry_cmd": self.poetry_cmd,
            "python_cmd": self.python_cmd,
            "project_root": str(self.project_root),
            "poetry_available": False,
            "virtual_env": None,
            "dependencies_installed": False,
        }

        # Check Poetry availability
        try:
            result = subprocess.run(
                [self.poetry_cmd.split()[0], "--version"],
                capture_output=True,
                check=True,
                text=True,
            )
            status["poetry_available"] = True
            status["poetry_version"] = result.stdout.strip()
        except Exception as e:
            status["poetry_error"] = str(e)

        # Check virtual environment
        try:
            if self.is_wsl:
                env_result = subprocess.run(
                    [self.poetry_cmd, "env", "info", "--path"],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                )
                if env_result.returncode == 0:
                    status["virtual_env"] = env_result.stdout.strip()
            else:
                env_result = subprocess.run(
                    [self.poetry_cmd, "env", "info"],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                )
                status["virtual_env"] = (
                    "detected" if env_result.returncode == 0 else None
                )
        except Exception as e:
            status["env_error"] = str(e)

        return status

    def setup_cross_platform_config(self) -> bool:
        """Configure Poetry for cross-platform compatibility."""
        try:
            os.chdir(self.project_root)

            # Set Poetry configuration for cross-platform compatibility
            config_commands = [
                [self.poetry_cmd, "config", "virtualenvs.in-project", "false"],
                [self.poetry_cmd, "config", "virtualenvs.create", "true"],
                [self.poetry_cmd, "config", "installer.parallel", "true"],
            ]

            if self.is_wsl:
                # WSL-specific configurations
                config_commands.extend(
                    [
                        [
                            self.poetry_cmd,
                            "config",
                            "virtualenvs.use-poetry-python",
                            "false",
                        ],
                        [
                            self.poetry_cmd,
                            "config",
                            "virtualenvs.options.always-copy",
                            "false",
                        ],
                    ]
                )

            for cmd in config_commands:
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"Warning: {' '.join(cmd)} failed: {result.stderr}")

            return True

        except Exception as e:
            print(f"Failed to configure Poetry: {e}")
            return False

    def create_test_runner_script(self):
        """Create cross-platform test runner script."""
        test_runner_content = f'''#!/usr/bin/env python3
"""Cross-platform test runner for FileIntel."""

import os
import sys
import subprocess
from pathlib import Path

# Auto-detected configuration
PROJECT_ROOT = Path(__file__).parent
IS_WSL = {self.is_wsl}
POETRY_CMD = "{self.poetry_cmd}"
PYTHON_CMD = "{self.python_cmd}"

def run_tests(test_type="unit", pattern=None):
    """Run tests with proper environment setup."""
    os.chdir(PROJECT_ROOT)

    if IS_WSL:
        # WSL-specific command pattern
        if pattern:
            cmd = [POETRY_CMD, "run", PYTHON_CMD, "-m", "pytest", f"tests/{{test_type}}/{{pattern}}", "-v"]
        else:
            cmd = [POETRY_CMD, "run", PYTHON_CMD, "-m", "pytest", f"tests/{{test_type}}/", "-v"]
    else:
        # Windows/standard command pattern
        if pattern:
            cmd = [POETRY_CMD, "run", "pytest", f"tests/{{test_type}}/{{pattern}}", "-v"]
        else:
            cmd = [POETRY_CMD, "run", "pytest", f"tests/{{test_type}}/", "-v"]

    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode == 0
    except Exception as e:
        print(f"Test execution failed: {{e}}")
        return False

def check_environment():
    """Check if test environment is ready."""
    try:
        # Check Poetry
        subprocess.run([POETRY_CMD.split()[0], "--version"], check=True, capture_output=True)

        # Check dependencies (attempt import test)
        import_cmd = [POETRY_CMD, "run", PYTHON_CMD, "-c", "import pytest; print('Dependencies OK')"]
        result = subprocess.run(import_cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print("✓ Test environment ready")
            return True
        else:
            print("✗ Dependencies missing - run 'poetry install'")
            return False

    except Exception as e:
        print(f"✗ Environment check failed: {{e}}")
        return False

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FileIntel Test Runner")
    parser.add_argument("test_type", choices=["unit", "integration", "all"],
                       default="unit", nargs="?", help="Test type to run")
    parser.add_argument("--pattern", help="Test file pattern")
    parser.add_argument("--check", action="store_true", help="Check environment only")

    args = parser.parse_args()

    if args.check:
        sys.exit(0 if check_environment() else 1)

    if args.test_type == "all":
        success = run_tests("unit") and run_tests("integration")
    else:
        success = run_tests(args.test_type, args.pattern)

    sys.exit(0 if success else 1)
'''

        test_runner_path = self.project_root / "run_tests_cross_platform.py"
        with open(test_runner_path, "w") as f:
            f.write(test_runner_content)

        # Make executable on Unix systems
        if not platform.system() == "Windows":
            os.chmod(test_runner_path, 0o755)

        return test_runner_path

    def generate_environment_report(self) -> str:
        """Generate detailed environment report."""
        status = self.check_environment()

        report = f"""
# FileIntel Test Environment Report

## Platform Information
- **Platform**: {status['platform']}
- **WSL Environment**: {status['is_wsl']}
- **Project Root**: {status['project_root']}

## Poetry Configuration
- **Poetry Command**: `{status['poetry_cmd']}`
- **Poetry Available**: {status['poetry_available']}
- **Python Command**: `{status['python_cmd']}`
"""

        if status.get("poetry_version"):
            report += f"- **Poetry Version**: {status['poetry_version']}\n"

        if status.get("virtual_env"):
            report += f"- **Virtual Environment**: {status['virtual_env']}\n"

        if status.get("poetry_error"):
            report += f"- **Poetry Error**: {status['poetry_error']}\n"

        report += f"""
## Recommended Commands

### WSL Environment
```bash
# Check environment
{status['poetry_cmd']} env info

# Install dependencies
{status['poetry_cmd']} install

# Run tests
{status['poetry_cmd']} run {status['python_cmd']} -m pytest tests/unit/ -v
```

### Windows Environment
```cmd
REM Install dependencies
{status['poetry_cmd']} install

REM Run tests
{status['poetry_cmd']} run pytest tests/unit/ -v
```

## Cross-Platform Test Runner
Use the generated `run_tests_cross_platform.py` script:
```bash
# Check environment
{status['python_cmd']} run_tests_cross_platform.py --check

# Run unit tests
{status['python_cmd']} run_tests_cross_platform.py unit

# Run specific test pattern
{status['python_cmd']} run_tests_cross_platform.py unit --pattern "*alerting*"
```
"""
        return report


def main():
    """Main setup function."""
    setup = TestEnvironmentSetup()

    print("FileIntel Test Environment Setup")
    print("=" * 40)

    # Check current status
    status = setup.check_environment()
    print(f"Platform: {status['platform']}")
    print(f"WSL: {status['is_wsl']}")
    print(f"Poetry: {status['poetry_available']}")

    # Configure Poetry
    print("\nConfiguring Poetry for cross-platform compatibility...")
    if setup.setup_cross_platform_config():
        print("✓ Poetry configuration updated")
    else:
        print("✗ Poetry configuration failed")

    # Create test runner
    print("\nCreating cross-platform test runner...")
    test_runner = setup.create_test_runner_script()
    print(f"✓ Created: {test_runner}")

    # Generate report
    report = setup.generate_environment_report()
    report_path = setup.project_root / "TEST_ENVIRONMENT_REPORT.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"✓ Generated: {report_path}")

    print("\n" + "=" * 40)
    print("Setup complete! Next steps:")
    print("1. Run: poetry install")
    print("2. Test: python run_tests_cross_platform.py --check")
    print("3. Run tests: python run_tests_cross_platform.py unit")


if __name__ == "__main__":
    main()
