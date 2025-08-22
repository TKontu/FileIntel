import os
from alembic.config import Config
from alembic import command
from dotenv import load_dotenv
from logging.config import fileConfig

print("Starting migration script...")
load_dotenv()
print("Dotenv loaded.")


def run_migrations():
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

    print("Running upgrade command...")
    command.upgrade(alembic_cfg, "head")
    print("Upgrade command finished.")


if __name__ == "__main__":
    run_migrations()
