# How Database Schema is Managed

This document outlines the current process for managing the database schema.

## Schema Management Strategy

The project uses a direct, model-driven approach to schema management. Instead of using a versioned migration system like Alembic, the application relies on SQLAlchemy's `Base.metadata.create_all()` function.

This function is called every time the API service starts up. It performs the following actions:

1.  It connects to the database.
2.  It inspects the current schema.
3.  It creates any tables that are defined in the SQLAlchemy models (in `src/document_analyzer/storage/models.py`) but do not yet exist in the database.

**Note:** This approach does **not** handle schema alterations (e.g., adding or removing a column from an existing table).

## How to Make Schema Changes

Because the system does not handle automatic alterations, making changes to the database schema requires a manual reset.

**The process is as follows:**

1.  **Modify the Models:** Make the desired changes to your table definitions in `src/document_analyzer/storage/models.py`.

2.  **Reset the Database:** To apply these changes, you must completely reset the database. This will wipe all existing data. Run the following commands:

    ```bash
    # Stop all services and remove the associated volumes (including the database)
    docker-compose down -v

    # Rebuild the images and start the services with a fresh, empty database
    docker-compose up --build -d
    ```

3.  **Verification:** Upon startup, the API service will automatically create the new, complete schema that matches your updated models.
