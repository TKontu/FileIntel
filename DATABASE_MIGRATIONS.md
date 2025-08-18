# How to Manage Database Migrations

This guide outlines the process for creating, applying, and reverting database schema changes using Alembic.

## Initial Setup

For the migration system to work, the Alembic environment must be able to find your database models. This is configured in `migrations/env.py`.

The `target_metadata` variable in this file must be set to your SQLAlchemy `Base.metadata` object. This has already been configured:

```python
# migrations/env.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.document_analyzer.storage.models import Base
target_metadata = Base.metadata
```

## The Automated Application Process

The project is configured to automatically apply database migrations when the `api` or `worker` services start up.

When you run `docker-compose up`, the entrypoint script in the container will execute `alembic upgrade head`. This command applies any new migration scripts to the database, ensuring your schema is always in sync with the code.

## How to Create a New Migration (The "Upgrade" Path)

When you change the SQLAlchemy models in `src/document_analyzer/storage/models.py` (e.g., add a new table or column), you must generate a migration script.

**Step 1: Ensure the database is running**

```bash
docker-compose up -d postgres
```

**Step 2: Generate the migration script**
Run the following command to have Alembic compare the current database state with your updated models and generate a new script.

```bash
docker-compose run --rm api alembic revision --autogenerate -m "A descriptive message about the changes"
```

- Replace `"A descriptive message..."` with a short, clear description (e.g., `"Add description to collections table"`).

**Step 3: Review the generated script**
A new file will be created in the `migrations/versions/` directory. It's good practice to review this file to ensure the generated `upgrade()` and `downgrade()` functions match your intended changes.

**Step 4: Apply the migration**
The migration will be applied automatically the next time you start the application services. To apply it immediately, you can run:

```bash
docker-compose up -d --force-recreate api worker
```

## How to Revert a Migration (The "Downgrade" Path)

To undo a migration, you first modify the code, then generate a new migration script that reverses the changes.

**Step 1: Revert the model change**
Undo the change you made in `src/document_analyzer/storage/models.py`. For example, remove the column you just added.

**Step 2: Generate the downgrade migration**
Run the same `autogenerate` command as before. Alembic will detect that a column has been removed from the model and will generate a script with the appropriate `op.drop_column()` command in its `upgrade()` function.

```bash
docker-compose run --rm api alembic revision --autogenerate -m "Revert: A descriptive message"
```

**Step 3: Apply the downgrade**
Restart the services to apply the new migration, which will effectively undo the previous one.

```bash
docker-compose up -d --force-recreate api worker
```

## Troubleshooting

### FAILED: Can't proceed with --autogenerate option

This error means Alembic can't find your database models. Ensure that `migrations/env.py` is correctly configured to point to your `Base.metadata` object as described in the "Initial Setup" section.

### FAILED: Can't locate revision identified by '...'

This error occurs when the migration version recorded in your database (in the `alembic_version` table) points to a migration file that no longer exists on disk. This can happen if you manually delete a migration file.

**To Fix:** You need to "stamp" the database to the last known _correct_ version.

1.  **Find the last correct migration hash:** Look in the `migrations/versions/` directory for the most recent migration file that should have been applied. The filename will contain the hash (e.g., `3950395dfef3`).

2.  **Stamp the database:** Run the following command, replacing `<hash>` with the correct migration hash.
    `bash
    docker-compose exec postgres psql -U user -d fileintel -c "UPDATE alembic_version SET version_num='<hash>'"
    `
    This manually resets the database's internal version counter, resolving the inconsistency. After this, you can restart your application.
