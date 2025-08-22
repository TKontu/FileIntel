from ..storage.postgresql_storage import PostgreSQLStorage
from ..storage.models import SessionLocal, Collection, Document
from ..storage.base import StorageInterface
from fastapi import Depends, HTTPException
import uuid


def get_storage():
    db = SessionLocal()
    try:
        yield PostgreSQLStorage(db)
    finally:
        db.close()


def get_collection_by_id_or_name(
    collection_identifier: str, storage: StorageInterface = Depends(get_storage)
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


def get_document_by_id_or_filename(
    document_identifier: str,
    collection: Collection = Depends(get_collection_by_id_or_name),
    storage: StorageInterface = Depends(get_storage),
) -> Document:
    try:
        # Try to interpret as UUID
        uuid.UUID(document_identifier)
        document = storage.get_document(document_identifier)
    except ValueError:
        # If not a UUID, treat as a filename
        document = storage.get_document_by_filename_and_collection(
            document_identifier, collection.id
        )

    if not document or document.collection_id != collection.id:
        raise HTTPException(
            status_code=404,
            detail=f"Document '{document_identifier}' not found in collection '{collection.name}'",
        )
    return document
