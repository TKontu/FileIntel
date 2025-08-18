from ..storage.postgresql_storage import PostgreSQLStorage
from ..storage.models import SessionLocal


def get_storage():
    db = SessionLocal()
    try:
        yield PostgreSQLStorage(db)
    finally:
        db.close()
