from ..storage.postgresql_storage import PostgreSQLStorage
from ..storage.models import SessionLocal, Collection
from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader
from ..core.config import get_config
from typing import Optional, Generator
import uuid

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_api_key(api_key: Optional[str] = Security(api_key_header)) -> Optional[str]:
    config = get_config()
    if config.api.authentication.enabled:
        if not api_key:
            raise HTTPException(status_code=403, detail="API key is required")
        if api_key == config.api.authentication.api_key:
            return api_key
        else:
            raise HTTPException(status_code=401, detail="Invalid API Key")
    return None


def get_storage() -> Generator[PostgreSQLStorage, None, None]:
    db = SessionLocal()
    try:
        yield PostgreSQLStorage(db)
    finally:
        db.close()


def get_collection_by_id_or_name(
    collection_identifier: str, storage: PostgreSQLStorage = Depends(get_storage)
) -> Collection:
    try:
        # Try to interpret as UUID
        uuid.UUID(collection_identifier)
        collection = storage.get_collection(collection_identifier)
    except ValueError:
        # If not a UUID, treat as a name
        collection = storage.get_collection_by_name(collection_identifier)

    if not collection:
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{collection_identifier}' not found",
        )
    return collection
