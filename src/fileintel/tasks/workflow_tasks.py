"""
Workflow orchestration Celery tasks.

Demonstrates advanced Celery patterns: groups, chords, chains, and callbacks
for complex document processing and analysis workflows.
"""

import logging
from typing import List, Dict, Any, Optional
from celery import group, chain, chord, signature

from fileintel.celery_config import app
from .base import BaseFileIntelTask
from .document_tasks import process_document, extract_document_metadata
from .llm_tasks import generate_and_store_chunk_embedding
from .graphrag_tasks import build_graph_index

logger = logging.getLogger(__name__)


@app.task(base=BaseFileIntelTask, bind=True, queue="document_processing")
def complete_collection_analysis(
    self,
    collection_id: str,
    file_paths: List[str],
    build_graph: bool = True,
    extract_metadata: bool = True,
    generate_embeddings: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """
    Complete end-to-end collection analysis using advanced Celery patterns.

    This demonstrates:
    - Group: Parallel document processing
    - Chord: Callback after all documents processed
    - Chain: Sequential dependent operations

    Args:
        collection_id: Collection identifier
        file_paths: List of document file paths
        build_graph: Whether to build GraphRAG index
        extract_metadata: Whether to extract LLM-based metadata
        generate_embeddings: Whether to generate embeddings
        **kwargs: Additional parameters

    Returns:
        Dict containing complete analysis results
    """
    self.validate_input(
        ["collection_id", "file_paths"],
        collection_id=collection_id,
        file_paths=file_paths,
    )

    try:
        self.update_progress(0, 6, "Orchestrating complete collection analysis")

        # Update collection status to processing
        from fileintel.celery_config import get_shared_storage
        from fileintel.core.config import get_config

        config = get_config()
        storage = get_shared_storage()
        try:
            storage.update_collection_status(collection_id, "processing")

            # Phase 1: Parallel document processing (GROUP)
            self.update_progress(1, 6, "Starting parallel document processing")

            # Create signatures for document processing tasks
            document_signatures = [
                process_document.s(
                    file_path=file_path,
                    document_id=f"{collection_id}_doc_{i}",
                    collection_id=collection_id,
                    **kwargs,
                )
                for i, file_path in enumerate(file_paths)
            ]

            # Create completion callback for collection status update
            # Note: workflow_results will be passed automatically as first arg by chord
            completion_callback = mark_collection_completed.s(collection_id)

            # Choose workflow based on requested operations
            if extract_metadata and generate_embeddings:
                # Full workflow: documents → metadata → embeddings → completion
                workflow_result = chord(document_signatures)(
                    generate_collection_metadata_and_embeddings.s(
                        collection_id=collection_id,
                    )
                ).apply_async()

                logger.info(
                    f"Started full workflow with {len(document_signatures)} document tasks for collection {collection_id}"
                )

                return {
                    "collection_id": collection_id,
                    "workflow_task_id": workflow_result.id,
                    "status": "processing_with_metadata_and_embeddings",
                    "message": f"Started processing {len(file_paths)} documents with metadata extraction and embedding generation",
                }
            elif extract_metadata:
                # Metadata only workflow: documents → metadata → completion
                workflow_result = chord(document_signatures)(
                    generate_collection_metadata.s(
                        collection_id=collection_id,
                    )
                ).apply_async()

                logger.info(
                    f"Started metadata workflow with {len(document_signatures)} document tasks for collection {collection_id}"
                )

                return {
                    "collection_id": collection_id,
                    "workflow_task_id": workflow_result.id,
                    "status": "processing_with_metadata",
                    "message": f"Started processing {len(file_paths)} documents with metadata extraction",
                }
            elif generate_embeddings:
                # Embeddings only workflow: documents → embeddings → completion
                workflow_result = chord(document_signatures)(
                    generate_collection_embeddings_simple.s(
                        collection_id=collection_id,
                    )
                ).apply_async()

                logger.info(
                    f"Started embeddings workflow with {len(document_signatures)} document tasks for collection {collection_id}"
                )

                return {
                    "collection_id": collection_id,
                    "workflow_task_id": workflow_result.id,
                    "status": "processing_with_embeddings",
                    "message": f"Started processing {len(file_paths)} documents with embedding generation",
                }
            else:
                # Simple document processing only - use chord with completion callback
                workflow_result = chord(document_signatures)(
                    completion_callback
                ).apply_async()

                logger.info(
                    f"Started simple workflow with {len(document_signatures)} document tasks and completion callback for collection {collection_id}"
                )

                return {
                    "collection_id": collection_id,
                    "workflow_task_id": workflow_result.id,
                    "status": "processing_documents",
                    "message": f"Started processing {len(file_paths)} documents",
                }
        finally:
            storage.close()

    except Exception as e:
        logger.error(f"Error in complete collection analysis: {e}")

        # Update collection status to failed
        try:
            storage = get_shared_storage()
            storage.update_collection_status(collection_id, "failed")
        except:
            pass  # Don't fail the task if status update fails

        return {"collection_id": collection_id, "error": str(e), "status": "failed"}


@app.task(base=BaseFileIntelTask, bind=True, queue="document_processing")
def mark_collection_completed(
    self, workflow_results, collection_id: str
) -> Dict[str, Any]:
    """
    Mark collection as completed after all processing is done.
    Called as callback after document processing and embedding generation.
    """
    try:
        from fileintel.celery_config import get_shared_storage

        storage = get_shared_storage()
        # Check if any document processing failed
        has_failures = False
        if isinstance(workflow_results, list):
            has_failures = any(
                result.get("status") == "failed"
                for result in workflow_results
                if isinstance(result, dict)
            )

        final_status = "failed" if has_failures else "completed"
        storage.update_collection_status(collection_id, final_status)

        logger.info(f"Collection {collection_id} marked as {final_status}")

        return {
            "collection_id": collection_id,
            "status": final_status,
            "workflow_results": workflow_results,
        }

    except Exception as e:
        logger.error(f"Error marking collection {collection_id} as completed: {e}")

        # Attempt to update collection status to failed even if callback processing fails
        try:
            from fileintel.celery_config import get_shared_storage

            storage = get_shared_storage()
            storage.update_collection_status(collection_id, "failed")
            logger.info(
                f"Updated collection {collection_id} status to failed due to callback error"
            )
        except Exception as status_error:
            logger.error(
                f"Failed to update collection status after callback error: {status_error}"
            )

        return {
            "collection_id": collection_id,
            "error": str(e),
            "status": "completion_callback_failed",
        }


@app.task(base=BaseFileIntelTask, bind=True, queue="embedding_processing")
def generate_collection_embeddings_simple(
    self, document_results, collection_id: str
) -> Dict[str, Any]:
    """
    Simplified embedding generation for collection - replaces the wrapper chain.

    Args:
        document_results: Results from document processing group (chord input)
        collection_id: Collection to generate embeddings for

    Returns:
        Dict containing embedding results and completion status
    """
    try:
        from fileintel.celery_config import get_shared_storage
        from fileintel.core.config import get_config
        from .llm_tasks import generate_and_store_chunk_embedding
        from celery import chord, group

        config = get_config()
        storage = get_shared_storage()
        try:
            self.update_progress(0, 3, "Starting simplified embedding generation")

            # Get chunks that need embeddings (chunks are created during document processing)
            # For two-tier chunking, only vector chunks need embeddings
            if getattr(config.rag, 'enable_two_tier_chunking', False):
                chunks = storage.get_chunks_by_type_for_collection(collection_id, 'vector')
                logger.info(f"Two-tier chunking enabled: processing only vector chunks for embeddings")
            else:
                chunks = storage.get_all_chunks_for_collection(collection_id)
            if not chunks:
                # No embeddings to generate, call completion directly (don't wait for result)
                self.update_progress(
                    2, 3, "No chunks found, calling completion callback"
                )
                completion_task = mark_collection_completed.apply_async(
                    args=[document_results, collection_id]
                )

                self.update_progress(
                    3, 3, "Collection processing completed (no embeddings)"
                )
                return {
                    "collection_id": collection_id,
                    "embeddings_generated": 0,
                    "status": "completed",
                    "message": "No chunks found requiring embeddings",
                    "completion_task_id": completion_task.id,
                }

            logger.info(
                f"Found {len(chunks)} chunks to process for collection {collection_id}"
            )
            self.update_progress(
                1, 3, f"Generating embeddings for {len(chunks)} chunks"
            )

            # Create embedding jobs for all chunks - direct approach
            embedding_jobs = group(
                generate_and_store_chunk_embedding.s(chunk.id, chunk.chunk_text)
                for chunk in chunks
            )

            # Execute embeddings asynchronously (don't block with .get())
            embedding_result = embedding_jobs.apply_async()

            self.update_progress(
                2, 3, "Embeddings started, calling completion callback"
            )
            completion_task = mark_collection_completed.apply_async(
                args=[document_results, collection_id]
            )

            self.update_progress(3, 3, "Collection processing tasks initiated")

            # Return task IDs instead of blocking results
            successful_embeddings = len(chunks)  # Assume all will succeed for now

            return {
                "collection_id": collection_id,
                "total_chunks": len(chunks),
                "embeddings_task_id": embedding_result.id,
                "completion_task_id": completion_task.id,
                "status": "processing",
                "message": f"Started embedding generation for {len(chunks)} chunks",
            }
        finally:
            storage.close()

    except Exception as e:
        logger.error(
            f"Error in simplified embedding generation for collection {collection_id}: {e}"
        )

        # Attempt to update collection status to failed
        try:
            from fileintel.celery_config import get_shared_storage

            storage = get_shared_storage()
            storage.update_collection_status(collection_id, "failed")
            logger.info(
                f"Updated collection {collection_id} status to failed due to embedding error"
            )
        except Exception as status_error:
            logger.error(
                f"Failed to update collection status after embedding error: {status_error}"
            )

        return {"collection_id": collection_id, "error": str(e), "status": "failed"}


@app.task(base=BaseFileIntelTask, bind=True, queue="document_processing")
def incremental_collection_update(
    self,
    collection_id: str,
    new_file_paths: List[str],
    existing_embeddings: List[List[float]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Incremental collection update using CHORD pattern.

    Demonstrates:
    - Group: Process new documents in parallel
    - Callback: Update existing collection after all processing complete

    Args:
        collection_id: Collection identifier
        new_file_paths: New documents to add
        existing_embeddings: Existing embeddings to merge with
        **kwargs: Additional parameters

    Returns:
        Dict containing update results
    """
    self.validate_input(
        ["collection_id", "new_file_paths"],
        collection_id=collection_id,
        new_file_paths=new_file_paths,
    )

    try:
        self.update_progress(0, 3, "Starting incremental collection update")

        # Update collection status to processing
        from fileintel.celery_config import get_shared_storage

        storage = get_shared_storage()
        try:
            storage.update_collection_status(collection_id, "processing")

            # Process new documents in parallel (GROUP)
            new_doc_jobs = group(
                process_document.s(
                    file_path=file_path,
                    document_id=f"{collection_id}_new_{i}",
                    collection_id=collection_id,
                    **kwargs,
                )
                for i, file_path in enumerate(new_file_paths)
            )

            # Execute with callback using CHORD
            callback = update_collection_index.s(
                collection_id=collection_id, existing_embeddings=existing_embeddings
            )

            # Chord: callback runs after all group tasks complete (don't block with .get())
            chord_result = chord(new_doc_jobs)(callback).apply_async()

            self.update_progress(3, 3, "Incremental update workflow started")

            return {
                "collection_id": collection_id,
                "incremental_workflow_id": chord_result.id,
                "new_documents": len(new_file_paths),
                "status": "processing",
                "message": f"Started incremental update for {len(new_file_paths)} new documents",
            }
        finally:
            storage.close()

    except Exception as e:
        logger.error(f"Error in incremental collection update: {e}")

        # Update collection status to failed
        try:
            storage = get_shared_storage()
            storage.update_collection_status(collection_id, "failed")
        except:
            pass  # Don't fail the task if status update fails

        return {"collection_id": collection_id, "error": str(e), "status": "failed"}


@app.task(base=BaseFileIntelTask, bind=True, queue="graphrag_indexing")
def update_collection_index(
    self,
    document_results: List[Dict[str, Any]],
    collection_id: str,
    existing_embeddings: List[List[float]] = None,
) -> Dict[str, Any]:
    """
    Callback task to update collection index after document processing.

    This is typically called as a chord callback after parallel processing.

    Args:
        document_results: Results from parallel document processing
        collection_id: Collection identifier
        existing_embeddings: Existing embeddings to merge

    Returns:
        Dict containing index update results
    """
    try:
        self.update_progress(0, 3, "Updating collection index")

        # Filter successful documents
        successful_docs = [
            doc for doc in document_results if doc.get("status") == "completed"
        ]

        if not successful_docs:
            return {
                "collection_id": collection_id,
                "error": "No new documents to index",
                "status": "failed",
            }

        # Generate embeddings for new content
        self.update_progress(1, 3, "Generating embeddings for new content")

        all_new_chunks = []
        for doc in successful_docs:
            all_new_chunks.extend(doc.get("chunks", []))

        if all_new_chunks:
            # Generate embeddings for individual chunks using the existing pattern
            from celery import group

            embedding_jobs = group(
                generate_and_store_chunk_embedding.s(f"temp_chunk_{i}", chunk_text)
                for i, chunk_text in enumerate(all_new_chunks)
            )
            embedding_task = embedding_jobs.apply_async()

            # Don't block - assume all embeddings will succeed for collection status
            # The actual embeddings are stored in DB by individual tasks
            new_embeddings = [
                f"embedding_{i}" for i in range(len(all_new_chunks))
            ]  # Placeholder list for counting

            logger.info(f"Started embedding generation for {len(all_new_chunks)} chunks (task: {embedding_task.id})")
        else:
            new_embeddings = []

        # Merge with existing embeddings if provided
        if existing_embeddings:
            total_embeddings = existing_embeddings + new_embeddings
        else:
            total_embeddings = new_embeddings

        # Update GraphRAG index if requested
        self.update_progress(2, 3, "Updating GraphRAG index")

        graph_docs = [
            {
                "document_id": doc.get("document_id"),
                "content": "\n".join(doc.get("chunks", [])),
            }
            for doc in successful_docs
        ]

        # For incremental updates, we might want to rebuild the entire index
        # In production, this could be optimized for true incremental updates
        graph_task = build_graph_index.apply_async(
            args=[graph_docs, collection_id]
        )
        logger.info(f"Started GraphRAG index update (task: {graph_task.id})")

        # Update collection status to completed
        from fileintel.celery_config import get_shared_storage

        storage = get_shared_storage()
        try:
            storage.update_collection_status(collection_id, "completed")

            result = {
                "collection_id": collection_id,
                "new_documents_added": len(successful_docs),
                "total_embeddings": len(total_embeddings),
                "new_embeddings_generated": len(new_embeddings),
                "graphrag_task_id": graph_task.id,
                "status": "completed",
            }

            self.update_progress(3, 3, "Collection index update completed")
            logger.info(
                f"Collection {collection_id} incremental update completed and marked as completed"
            )
            return result
        finally:
            storage.close()

    except Exception as e:
        logger.error(f"Error updating collection index: {e}")

        # Update collection status to failed
        try:
            from fileintel.celery_config import get_shared_storage

            storage = get_shared_storage()
            try:
                storage.update_collection_status(collection_id, "failed")
            finally:
                storage.close()
        except:
            pass  # Don't fail the task if status update fails

        return {"collection_id": collection_id, "error": str(e), "status": "failed"}


@app.task(base=BaseFileIntelTask, bind=True, queue="llm_processing")
def generate_collection_metadata(
    self, document_results, collection_id: str
) -> Dict[str, Any]:
    """
    Extract metadata for all documents in a collection after document processing.

    Args:
        document_results: Results from document processing group (chord input)
        collection_id: Collection to extract metadata for

    Returns:
        Dict containing metadata extraction results and completion status
    """
    try:
        from fileintel.celery_config import get_shared_storage
        from .llm_tasks import extract_document_metadata
        from celery import group

        storage = get_shared_storage()
        try:
            self.update_progress(0, 3, "Starting collection metadata extraction")

            # Get documents that were successfully processed
            successful_docs = [
                doc for doc in document_results
                if isinstance(doc, dict) and doc.get("status") == "completed"
            ]

            if not successful_docs:
                self.update_progress(2, 3, "No successful documents found, calling completion")
                completion_task = mark_collection_completed.apply_async(
                    args=[document_results, collection_id]
                )

                self.update_progress(3, 3, "Collection processing completed (no metadata)")
                return {
                    "collection_id": collection_id,
                    "metadata_extracted": 0,
                    "status": "completed",
                    "message": "No documents found for metadata extraction",
                    "completion_task_id": completion_task.id,
                }

            logger.info(
                f"Found {len(successful_docs)} documents to extract metadata for in collection {collection_id}"
            )
            self.update_progress(
                1, 3, f"Extracting metadata for {len(successful_docs)} documents"
            )

            # Create metadata extraction jobs for all documents
            metadata_jobs = []
            for doc_result in successful_docs:
                document_id = doc_result.get("document_id")
                if document_id:
                    # Get document and chunks for metadata extraction
                    document = storage.get_document(document_id)
                    if document:
                        chunks = storage.get_all_chunks_for_document(document_id)
                        if chunks:
                            text_chunks = [chunk.chunk_text for chunk in chunks[:3]]
                            file_metadata = document.document_metadata if document.document_metadata else None

                            metadata_jobs.append(
                                extract_document_metadata.s(
                                    document_id=document_id,
                                    text_chunks=text_chunks,
                                    file_metadata=file_metadata
                                )
                            )

            if metadata_jobs:
                # Execute metadata extraction asynchronously
                metadata_result = group(metadata_jobs).apply_async()

                self.update_progress(2, 3, "Metadata extraction started, calling completion")
                completion_task = mark_collection_completed.apply_async(
                    args=[document_results, collection_id]
                )

                self.update_progress(3, 3, "Collection metadata extraction tasks initiated")

                return {
                    "collection_id": collection_id,
                    "total_documents": len(successful_docs),
                    "metadata_jobs_started": len(metadata_jobs),
                    "metadata_task_id": metadata_result.id,
                    "completion_task_id": completion_task.id,
                    "status": "processing",
                    "message": f"Started metadata extraction for {len(metadata_jobs)} documents",
                }
            else:
                self.update_progress(2, 3, "No documents found for metadata extraction")
                completion_task = mark_collection_completed.apply_async(
                    args=[document_results, collection_id]
                )

                return {
                    "collection_id": collection_id,
                    "metadata_extracted": 0,
                    "status": "completed",
                    "message": "No valid documents found for metadata extraction",
                    "completion_task_id": completion_task.id,
                }
        finally:
            storage.close()

    except Exception as e:
        logger.error(
            f"Error in collection metadata extraction for collection {collection_id}: {e}"
        )

        # Attempt to update collection status to failed
        try:
            from fileintel.celery_config import get_shared_storage

            storage = get_shared_storage()
            try:
                storage.update_collection_status(collection_id, "failed")
                logger.info(
                    f"Updated collection {collection_id} status to failed due to metadata error"
                )
            finally:
                storage.close()
        except Exception as status_error:
            logger.error(
                f"Failed to update collection status after metadata error: {status_error}"
            )

        return {"collection_id": collection_id, "error": str(e), "status": "failed"}


@app.task(base=BaseFileIntelTask, bind=True, queue="llm_processing")
def generate_collection_metadata_and_embeddings(
    self, document_results, collection_id: str
) -> Dict[str, Any]:
    """
    Extract metadata and generate embeddings for all documents in a collection.

    Args:
        document_results: Results from document processing group (chord input)
        collection_id: Collection to process

    Returns:
        Dict containing processing results and completion status
    """
    try:
        from fileintel.celery_config import get_shared_storage
        from .llm_tasks import extract_document_metadata
        from celery import group

        storage = get_shared_storage()
        try:
            self.update_progress(0, 4, "Starting collection metadata and embeddings generation")

            # Get documents that were successfully processed
            successful_docs = [
                doc for doc in document_results
                if isinstance(doc, dict) and doc.get("status") == "completed"
            ]

            if not successful_docs:
                self.update_progress(3, 4, "No successful documents found, calling completion")
                completion_task = mark_collection_completed.apply_async(
                    args=[document_results, collection_id]
                )

                return {
                    "collection_id": collection_id,
                    "metadata_extracted": 0,
                    "embeddings_generated": 0,
                    "status": "completed",
                    "message": "No documents found for processing",
                    "completion_task_id": completion_task.id,
                }

            # Step 1: Start metadata extraction
            self.update_progress(1, 4, f"Starting metadata extraction for {len(successful_docs)} documents")

            metadata_jobs = []
            for doc_result in successful_docs:
                document_id = doc_result.get("document_id")
                if document_id:
                    document = storage.get_document(document_id)
                    if document:
                        chunks = storage.get_all_chunks_for_document(document_id)
                        if chunks:
                            text_chunks = [chunk.chunk_text for chunk in chunks[:3]]
                            file_metadata = document.document_metadata if document.document_metadata else None

                            metadata_jobs.append(
                                extract_document_metadata.s(
                                    document_id=document_id,
                                    text_chunks=text_chunks,
                                    file_metadata=file_metadata
                                )
                            )

            if metadata_jobs:
                metadata_result = group(metadata_jobs).apply_async()
                logger.info(f"Started metadata extraction for {len(metadata_jobs)} documents")

            # Step 2: Start embeddings generation
            self.update_progress(2, 4, "Starting embeddings generation")

            chunks = storage.get_all_chunks_for_collection(collection_id)
            if chunks:
                embedding_jobs = group(
                    generate_and_store_chunk_embedding.s(chunk.id, chunk.chunk_text)
                    for chunk in chunks
                )
                embedding_result = embedding_jobs.apply_async()
                logger.info(f"Started embedding generation for {len(chunks)} chunks")
            else:
                embedding_result = None

            # Step 3: Call completion
            self.update_progress(3, 4, "Starting completion callback")
            completion_task = mark_collection_completed.apply_async(
                args=[document_results, collection_id]
            )

            self.update_progress(4, 4, "Collection metadata and embeddings processing initiated")

            result = {
                "collection_id": collection_id,
                "total_documents": len(successful_docs),
                "completion_task_id": completion_task.id,
                "status": "processing",
            }

            if metadata_jobs:
                result.update({
                    "metadata_jobs_started": len(metadata_jobs),
                    "metadata_task_id": metadata_result.id,
                })

            if embedding_result:
                result.update({
                    "total_chunks": len(chunks),
                    "embeddings_task_id": embedding_result.id,
                })

            result["message"] = f"Started metadata extraction and embedding generation for collection"

            return result
        finally:
            storage.close()

    except Exception as e:
        logger.error(
            f"Error in collection metadata and embeddings for collection {collection_id}: {e}"
        )

        # Attempt to update collection status to failed
        try:
            from fileintel.celery_config import get_shared_storage

            storage = get_shared_storage()
            try:
                storage.update_collection_status(collection_id, "failed")
                logger.info(
                    f"Updated collection {collection_id} status to failed due to processing error"
                )
            finally:
                storage.close()
        except Exception as status_error:
            logger.error(
                f"Failed to update collection status after processing error: {status_error}"
            )

        return {"collection_id": collection_id, "error": str(e), "status": "failed"}


# Removed pipeline_document_analysis - was unused and contained .get() anti-pattern
