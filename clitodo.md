# FileIntel CLI Plan

This document outlines the plan for creating a command-line interface (CLI) for the FileIntel application.

## Core Features

The CLI will provide a user-friendly way to interact with the FileIntel API, covering the following areas:

- **Collections Management:** Create, list, get, and delete collections.
- **Document Management:** Upload, list, and delete documents within collections.
- **Querying:** Submit queries to collections and retrieve results.
- **Job Management:** Check the status and results of jobs.

## Command Structure

The CLI will be structured with intuitive subcommands:

```
fileintel
├── collections
│   ├── create <name>              # Create a new collection
│   ├── list                       # List all collections
│   ├── get <identifier>           # Get details of a collection
│   └── delete <identifier>        # Delete a collection
├── documents
│   ├── upload <collection> <path> # Upload a document to a collection
│   ├── list <collection>          # List documents in a collection
│   └── delete <collection> <doc>  # Delete a document
├── query <collection> <question>    # Submit a query to a collection
└── jobs
    ├── status <job_id>            # Get the status of a job
    └── result <job_id> [--md]     # Get the result of a job (default: JSON, --md for Markdown)
```

## Implementation Plan

1.  **Dependencies:**
    - Add `typer` for the CLI framework.
    - Add `rich` for enhanced terminal output.
    - Add `requests` for making API calls.

2.  **Project Structure:**
    - Create a new directory `src/cli` for all CLI-related code.
    - Create a `src/cli/client.py` to handle communication with the FileIntel API.
    - Implement each command group in its own file (e.g., `src/cli/collections.py`).

3.  **Entry Point:**
    - Create a main `src/cli/main.py` to orchestrate the CLI commands.
    - Update `pyproject.toml` to add a `[tool.poetry.scripts]` section, creating a console script entry point for `fileintel`.

4.  **Development Workflow:**
    - Implement the `collections` commands first.
    - Follow with the `documents` commands.
    - Implement the `query` and `jobs` commands.
    - Add robust error handling and user feedback.
    - Write unit tests for the CLI commands.
