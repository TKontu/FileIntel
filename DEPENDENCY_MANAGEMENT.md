# How to Manage Python Dependencies

This project uses `poetry` for dependency management and `pre-commit` to automate and enforce consistency.

## One-Time Setup

Before you make your first commit, you need to install the git hooks. Run this command once from the root of the project:

```bash
pre-commit install
```

This will activate the automated checks that run before each commit.

## The New Workflow

The process for adding, updating, or removing dependencies is now simpler and safer. The pre-commit hooks will handle the tedious parts for you.

### How to Add a New Dependency

To add a new library (e.g., `requests`), run:

```bash
poetry add requests
```

This will automatically update your `pyproject.toml` and `poetry.lock` files.

### How to Update a Dependency

To update a specific library to its latest allowed version, run:

```bash
poetry update requests
```

To update all libraries, run:
```bash
poetry update
```

### How to Remove a Dependency

To remove a library, run:

```bash
poetry remove requests
```

## Committing Your Changes

When you are ready to commit your changes, simply `git add` the `pyproject.toml` file and commit.

```bash
git add pyproject.toml
git commit -m "Add requests dependency"
```

**What Happens Automatically:**

1.  The pre-commit hook will trigger.
2.  It will run `poetry-check` to ensure your `pyproject.toml` is valid.
3.  It will then run `poetry-lock`. This command checks if `poetry.lock` is in sync with `pyproject.toml`. If it's not, **the hook will automatically update `poetry.lock` for you.**
4.  The hook will also run `black` to format your code and `flake8` to check for style issues.

If the hook makes any changes (like updating the lock file), the commit will be aborted with a message. This is expected. Just review the changes and run `git add .` and `git commit` again. The second time, the checks will pass, and your commit will be successful.

This process guarantees that your `pyproject.toml` and `poetry.lock` files are always in sync in your commits.
