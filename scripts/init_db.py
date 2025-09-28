import os
import sys
import time

from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from fileintel.storage.models import Base

# Construct the database URL from environment variables
db_user = os.getenv("DB_USER", "user")
db_password = os.getenv("DB_PASSWORD", "password")
db_host = os.getenv("DB_HOST", "postgres")
db_port = os.getenv("DB_PORT", "5432")
db_name = os.getenv("DB_NAME", "fileintel")

DATABASE_URL = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"


def init_db():
    # Wait for the database to be ready
    for _ in range(10):
        try:
            engine = create_engine(DATABASE_URL)
            connection = engine.connect()
            connection.close()
            print("Database is ready!")
            break
        except OperationalError:
            print("Database not ready yet, waiting...")
            time.sleep(1)
    else:
        print("Could not connect to the database.")
        sys.exit(1)

    with engine.connect() as connection:
        # Drop all tables first
        print("Dropping all tables...")
        Base.metadata.drop_all(bind=engine)
        print("Tables dropped successfully.")

        # Create pgvector extension
        print("Creating pgvector extension...")
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        print("pgvector extension created successfully.")

        # Create tables
        print("Creating tables...")
        Base.metadata.create_all(bind=engine, checkfirst=False)
        print("Tables created successfully.")


if __name__ == "__main__":
    init_db()
