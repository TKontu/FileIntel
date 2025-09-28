import os
from alembic.config import Config
from alembic import command
from dotenv import load_dotenv
from logging.config import fileConfig

print("Starting migration script...")
load_dotenv()
print("Dotenv loaded.")


def create_migration(message: str):
    print("Creating Alembic config...")
    alembic_cfg = Config("alembic.ini")
    fileConfig(alembic_cfg.config_file_name)  # ← setup logging
    print("Alembic config created.")

    print("Setting script location...")
    alembic_cfg.set_main_option("script_location", "migrations")
    print("Script location set.")

    db_url = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    print(f"Setting database URL to: {db_url}")
    alembic_cfg.set_main_option("sqlalchemy.url", db_url)
    print("Database URL set.")

    print("Running revision command...")
    command.revision(alembic_cfg, message=message, autogenerate=True)
    print("Revision command finished.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python create_migration.py <message>")
        sys.exit(1)
    create_migration(sys.argv[1])
