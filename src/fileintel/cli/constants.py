"""
CLI Constants for FileIntel.

Centralizes magic numbers and configuration values used across CLI modules.
"""

# Default limits and pagination
DEFAULT_TASK_LIMIT = 20
DEFAULT_TASK_OFFSET = 0
DEFAULT_TASK_TIMEOUT = 300  # seconds (5 minutes)
DEFAULT_POLL_INTERVAL = 2.0  # seconds

# Progress and display
JSON_INDENT = 2
TASK_ID_DISPLAY_LENGTH = 12  # characters to show of task ID
PROGRESS_BAR_TOTAL = 100  # percentage

# Exit codes
CLI_SUCCESS = 0
CLI_ERROR = 1

# API Configuration
DEFAULT_API_PORT = 8000
