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
from .progress_tracker import WorkflowProgressTracker

logger = logging.getLogger(__name__)


@app.task(queue="default")
def track_task_completion(workflow_id: str, task_name: str) -> None:
    """
    Track completion of a single task in a workflow for progress logging.

    Called automatically as a link callback after each task completes.
    Logs progress at intervals (every 10 tasks or 10% progress).

    NOTE: This task must NEVER raise exceptions, as it's a non-critical
    logging operation that shouldn't break workflows.

    Args:
        workflow_id: Unique workflow identifier (parent task ID)
        task_name: Human-readable task type (e.g., "document", "embedding")
    """
    try:
        import time
        tracker = WorkflowProgressTracker()
        current, total, should_log = tracker.increment(workflow_id)

        if should_log and total > 0:
            percentage = (current / total) * 100
            if current == total:
                logger.info(f"Completed: {current}/{total} {task_name}s (100%)")
                # Cleanup Redis keys when workflow completes
                tracker.cleanup(workflow_id)
            else:
                logger.info(f"Progress: {current}/{total} {task_name}s ({percentage:.0f}%)")

        # Mark workflow as active (for orphan detection) - 2 hour TTL
        try:
            tracker.redis.setex(f"workflow:{workflow_id}:last_active", 7200, int(time.time()))
        except Exception as timestamp_error:
            # Non-critical, just log and continue
            logger.debug(f"Failed to update workflow timestamp: {timestamp_error}")
    except Exception as e:
        # CRITICAL: Never let progress tracking errors break workflows
        logger.error(f"Progress tracking failed for workflow {workflow_id}: {e}", exc_info=True)


@app.task(queue="default")
def cleanup_orphaned_progress_keys() -> Dict[str, Any]:
    """
    Periodic task to clean up abandoned workflow progress keys.

    Finds workflows that haven't been active for >2 hours and removes their Redis keys.
    Should be scheduled to run periodically (e.g., every hour via Celery beat).

    Returns:
        Dict with cleanup statistics
    """
    try:
        import time
        tracker = WorkflowProgressTracker()
        redis = tracker.redis

        # Find all workflow keys
        workflow_pattern = "workflow:*:total"
        workflow_keys = redis.keys(workflow_pattern)

        cleaned = 0
        active = 0
        current_time = int(time.time())
        inactive_threshold = 7200  # 2 hours

        for total_key in workflow_keys:
            try:
                # Extract workflow_id from key (format: "workflow:ID:total")
                workflow_id = total_key.split(":")[1]

                # Check last active timestamp
                last_active_key = f"workflow:{workflow_id}:last_active"
                last_active = redis.get(last_active_key)

                if last_active is None:
                    # No timestamp means very old or completed workflow - clean it up
                    tracker.cleanup(workflow_id)
                    cleaned += 1
                    logger.debug(f"Cleaned up workflow {workflow_id} (no timestamp)")
                else:
                    # Check if inactive for too long
                    last_active_time = int(last_active)
                    inactive_duration = current_time - last_active_time

                    if inactive_duration > inactive_threshold:
                        tracker.cleanup(workflow_id)
                        cleaned += 1
                        logger.info(
                            f"Cleaned up orphaned workflow {workflow_id} "
                            f"(inactive for {inactive_duration // 60} minutes)"
                        )
                    else:
                        active += 1
            except Exception as key_error:
                logger.warning(f"Error processing workflow key {total_key}: {key_error}")
                continue

        result = {
            "cleaned": cleaned,
            "active": active,
            "status": "completed"
        }

        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} orphaned workflows, {active} still active")

        return result

    except Exception as e:
        logger.error(f"Orphaned key cleanup failed: {e}", exc_info=True)
        return {"error": str(e), "status": "failed"}


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
        from fileintel.celery_config import get_storage_context
        from fileintel.core.config import get_config

        config = get_config()
        with get_storage_context() as storage:
            storage.update_collection_status(collection_id, "processing")

            # Phase 1: Parallel document processing (GROUP)
            self.update_progress(1, 6, "Starting parallel document processing")

            # Initialize progress tracking
            workflow_id = self.request.id
            tracker = WorkflowProgressTracker()
            tracker.initialize(workflow_id, len(file_paths))
            logger.info(f"Starting processing of {len(file_paths)} documents")

            # Create signatures for document processing tasks with progress tracking
            document_signatures = [
                process_document.s(
                    file_path=file_path,
                    document_id=f"{collection_id}_doc_{i}",
                    collection_id=collection_id,
                    **kwargs,
                ).set(link=track_task_completion.si(workflow_id, "document"))
                for i, file_path in enumerate(file_paths)
            ]

            # Validate we have documents to process
            if not document_signatures:
                storage.update_collection_status(collection_id, "failed")
                return {
                    "collection_id": collection_id,
                    "error": "No valid documents to process",
                    "status": "failed"
                }

            # Create completion callback for collection status update
            # Note: workflow_results will be passed automatically as first arg by chord
            completion_callback = mark_collection_completed.s(collection_id)

            # Choose workflow based on requested operations
            if extract_metadata and generate_embeddings:
                # Full workflow: documents → metadata → embeddings → completion
                # Note: chord()(callback) in Celery 5.x automatically calls apply_async() internally
                workflow_result = chord(document_signatures)(
                    generate_collection_metadata_and_embeddings.s(
                        collection_id=collection_id,
                    )
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
                )

                return {
                    "collection_id": collection_id,
                    "workflow_task_id": workflow_result.id,
                    "status": "processing_documents",
                    "message": f"Started processing {len(file_paths)} documents",
                }

    except Exception as e:
        logger.error(f"Error in complete collection analysis: {e}")

        # Update collection status to failed using context manager
        try:
            from fileintel.celery_config import get_storage_context

            with get_storage_context() as storage:
                storage.update_collection_status(collection_id, "failed")
        except Exception as status_error:
            logger.error(f"Failed to update collection status after error: {status_error}")

        return {"collection_id": collection_id, "error": str(e), "status": "failed"}


@app.task(base=BaseFileIntelTask, bind=True, queue="document_processing")
def mark_collection_completed(
    self,
    workflow_results: List[Dict[str, Any]],
    collection_id: str
) -> Dict[str, Any]:
    """Mark collection as completed after all processing tasks finish.

    This function is called automatically by Celery as a chord callback when
    all parallel document processing tasks complete. It analyzes the results,
    determines the final collection status, and updates the database.

    Args:
        self: Celery task instance (bound via bind=True)
        workflow_results: List of results from document processing tasks.
                         Automatically passed by Celery chord primitive.
                         Each result is a dict with keys:
                         - status: "success" or "failed"
                         - document_id: ID of processed document
                         - message: Processing message
                         - (other task-specific fields)
        collection_id: Collection identifier

    Returns:
        Dict containing:
        - collection_id: The collection ID
        - status: "completed" or "failed"
        - workflow_results: Original results from all tasks

    Note:
        This function is called automatically by Celery when the chord
        (parallel task group) completes. Do not call manually.

    Example workflow_results:
        [
            {"status": "success", "document_id": "doc1", "message": "Processed"},
            {"status": "failed", "document_id": "doc2", "message": "Parse error"},
        ]
    """
    try:
        from fileintel.celery_config import get_storage_context

        with get_storage_context() as storage:
            # Enhanced failure detection: count successes and failures explicitly
            success_count = 0
            failure_count = 0
            unknown_count = 0

            if isinstance(workflow_results, list):
                for result in workflow_results:
                    if isinstance(result, dict):
                        status = result.get("status", "unknown")
                        # Accept both "success" and "completed" as success indicators
                        if status in ("success", "completed"):
                            success_count += 1
                        elif status == "failed":
                            failure_count += 1
                        else:
                            # Unknown status treated as failure (fail-safe)
                            unknown_count += 1
                            failure_count += 1
                            logger.warning(
                                f"Document task returned unknown status '{status}' for collection {collection_id}"
                            )

            # Determine final status and message
            if failure_count > 0:
                final_status = "failed"
                message = f"{failure_count} documents failed, {success_count} succeeded"
                if unknown_count > 0:
                    message += f" ({unknown_count} with unknown status)"
            else:
                final_status = "completed"
                message = f"All {success_count} documents processed successfully"

            storage.update_collection_status(collection_id, final_status)

            logger.info(f"Collection {collection_id} marked as {final_status}: {message}")

        return {
            "collection_id": collection_id,
            "status": final_status,
            "message": message,
            "success_count": success_count,
            "failure_count": failure_count,
            "workflow_results": workflow_results,
        }

    except Exception as e:
        logger.error(f"Error marking collection {collection_id} as completed: {e}")

        # Attempt to update collection status to failed even if callback processing fails
        try:
            from fileintel.celery_config import get_storage_context

            with get_storage_context() as storage:
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

            logger.info(f"Starting generation of {len(chunks)} embeddings")
            self.update_progress(
                1, 3, f"Generating embeddings for {len(chunks)} chunks"
            )

            # Initialize progress tracking
            workflow_id = self.request.id
            tracker = WorkflowProgressTracker()
            tracker.initialize(workflow_id, len(chunks))

            # Create embedding jobs for all chunks with progress tracking
            embedding_jobs = group(
                generate_and_store_chunk_embedding.s(chunk.id, chunk.chunk_text).set(
                    link=track_task_completion.si(workflow_id, "embedding")
                )
                for chunk in chunks
            )

            # Use chord to ensure completion callback runs AFTER all embeddings finish
            # Fixed: Previously scheduled completion immediately, causing it to never run
            self.update_progress(
                2, 3, "Starting embeddings with completion callback"
            )

            completion_callback = mark_collection_completed.s(collection_id)
            # embedding_jobs is already a group, use it directly
            # Note: chord()(callback) already calls apply_async() in Celery 5.x
            workflow_result = chord(embedding_jobs)(completion_callback)

            self.update_progress(3, 3, "Collection processing workflow initiated")

            return {
                "collection_id": collection_id,
                "total_chunks": len(chunks),
                "workflow_task_id": workflow_result.id,
                "status": "processing",
                "message": f"Started embedding generation for {len(chunks)} chunks with completion callback",
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

            # Validate we have new documents to process
            if not new_file_paths:
                storage.update_collection_status(collection_id, "completed")
                return {
                    "collection_id": collection_id,
                    "error": "No new documents to process",
                    "status": "completed",
                    "message": "No new file paths provided for incremental update"
                }

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
            # Note: chord()(callback) already calls apply_async() in Celery 5.x
            chord_result = chord(new_doc_jobs)(callback)

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

        # Build list of all async jobs that need to complete
        from celery import group, chord
        all_jobs = []

        if all_new_chunks:
            # Generate embeddings for individual chunks
            embedding_jobs = [
                generate_and_store_chunk_embedding.s(f"temp_chunk_{i}", chunk_text)
                for i, chunk_text in enumerate(all_new_chunks)
            ]
            all_jobs.extend(embedding_jobs)
            logger.info(f"Added {len(all_new_chunks)} embedding jobs to workflow")

        # Update GraphRAG index if requested
        self.update_progress(2, 3, "Building update workflow")

        graph_docs = [
            {
                "document_id": doc.get("document_id"),
                "content": "\n".join(doc.get("chunks", [])),
            }
            for doc in successful_docs
        ]

        # Add GraphRAG indexing job to the workflow
        graph_job = build_graph_index.s(graph_docs, collection_id)
        all_jobs.append(graph_job)

        # Use chord to ensure all jobs complete before marking as done
        completion_callback = finalize_incremental_update.s(
            collection_id, len(successful_docs), len(all_new_chunks)
        )

        task_group = group(all_jobs)  # Pass list directly
        # Note: chord()(callback) already calls apply_async() in Celery 5.x
        workflow_result = chord(task_group)(completion_callback)

        logger.info(f"Started incremental update workflow with {len(all_jobs)} jobs for collection {collection_id}")

        self.update_progress(3, 3, "Incremental update workflow initiated")

        return {
            "collection_id": collection_id,
            "new_documents_added": len(successful_docs),
            "workflow_task_id": workflow_result.id,
            "embedding_jobs": len(all_new_chunks),
            "status": "processing",
            "message": f"Started incremental update with {len(all_jobs)} background jobs"
        }

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


@app.task(base=BaseFileIntelTask, bind=True, queue="document_processing")
def finalize_incremental_update(
    self, job_results, collection_id: str, documents_added: int, embeddings_count: int
) -> Dict[str, Any]:
    """
    Finalize incremental collection update after all background jobs complete.

    This is called as a chord callback after embeddings and GraphRAG indexing finish.

    Args:
        job_results: Results from all background jobs (embeddings + graphrag)
        collection_id: Collection being updated
        documents_added: Number of documents added
        embeddings_count: Number of embeddings generated

    Returns:
        Dict with finalization status
    """
    try:
        from fileintel.celery_config import get_storage_context

        with get_storage_context() as storage:
            storage.update_collection_status(collection_id, "completed")
            logger.info(
                f"Collection {collection_id} incremental update finalized: {documents_added} docs, {embeddings_count} embeddings"
            )

        return {
            "collection_id": collection_id,
            "documents_added": documents_added,
            "embeddings_count": embeddings_count,
            "status": "completed",
            "message": "Incremental update completed successfully"
        }

    except Exception as e:
        logger.error(f"Error finalizing incremental update for collection {collection_id}: {e}")

        # Try to mark as failed
        try:
            from fileintel.celery_config import get_storage_context
            with get_storage_context() as storage:
                storage.update_collection_status(collection_id, "failed")
        except Exception:
            pass

        return {
            "collection_id": collection_id,
            "error": str(e),
            "status": "failed"
        }


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
        from celery import group, chord

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

            logger.info(f"Starting metadata extraction for {len(successful_docs)} documents")
            self.update_progress(
                1, 3, f"Extracting metadata for {len(successful_docs)} documents"
            )

            # Initialize progress tracking
            workflow_id = self.request.id
            tracker = WorkflowProgressTracker()

            # First pass: collect job parameters and count
            job_params = []
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
                            job_params.append({
                                "document_id": document_id,
                                "text_chunks": text_chunks,
                                "file_metadata": file_metadata
                            })

            # Initialize tracker BEFORE creating jobs with callbacks
            if job_params:
                tracker.initialize(workflow_id, len(job_params))

            # Second pass: create jobs with progress tracking callbacks
            metadata_jobs = []
            for params in job_params:
                metadata_jobs.append(
                    extract_document_metadata.s(
                        document_id=params["document_id"],
                        text_chunks=params["text_chunks"],
                        file_metadata=params["file_metadata"]
                    ).set(link=track_task_completion.si(workflow_id, "metadata"))
                )

            if metadata_jobs:
                # Use chord to ensure completion callback runs AFTER all metadata jobs finish
                # Fixed: Previously scheduled completion immediately, causing race condition
                self.update_progress(2, 3, "Starting metadata extraction with completion callback")

                completion_callback = mark_collection_completed.s(collection_id)
                task_group = group(metadata_jobs)  # Pass list directly
                # Note: chord()(callback) already calls apply_async() in Celery 5.x
                workflow_result = chord(task_group)(completion_callback)

                self.update_progress(3, 3, "Collection metadata extraction workflow initiated")

                return {
                    "collection_id": collection_id,
                    "total_documents": len(successful_docs),
                    "metadata_jobs_started": len(metadata_jobs),
                    "workflow_task_id": workflow_result.id,
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
        from celery import group, chord

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

            # Step 2: Combine metadata and embedding jobs into one chord
            self.update_progress(2, 4, "Building workflow with metadata and embeddings")

            # Collect all jobs that need to complete
            all_jobs = []

            # Add metadata extraction jobs
            if metadata_jobs:
                all_jobs.extend(metadata_jobs)
                logger.info(f"Added {len(metadata_jobs)} metadata jobs to workflow")

            # Add embedding generation jobs
            chunks = storage.get_all_chunks_for_collection(collection_id)
            if chunks:
                embedding_jobs = [
                    generate_and_store_chunk_embedding.s(chunk.id, chunk.chunk_text)
                    for chunk in chunks
                ]
                all_jobs.extend(embedding_jobs)
                logger.info(f"Added {len(chunks)} embedding jobs to workflow")

            if all_jobs:
                # Use chord to ensure completion callback runs AFTER ALL jobs finish
                # Fixed: Now includes both metadata AND embeddings in the chord
                self.update_progress(3, 4, "Starting workflow with completion callback")

                completion_callback = mark_collection_completed.s(collection_id)
                # Create group from list of signatures
                task_group = group(all_jobs)  # Pass list directly, not unpacked
                # Note: chord()(callback) already calls apply_async() in Celery 5.x
                workflow_result = chord(task_group)(completion_callback)

                logger.info(
                    f"Started chord workflow: {len(all_jobs)} jobs (metadata+embeddings) → completion callback for collection {collection_id}"
                )

                self.update_progress(4, 4, "Collection metadata and embeddings workflow initiated")

                result = {
                    "collection_id": collection_id,
                    "total_documents": len(successful_docs),
                    "workflow_task_id": workflow_result.id,
                    "status": "processing",
                    "metadata_jobs": len(metadata_jobs) if metadata_jobs else 0,
                    "embedding_jobs": len(chunks) if chunks else 0,
                    "total_jobs": len(all_jobs),
                    "message": f"Started {len(all_jobs)} background jobs for metadata and embeddings"
                }

                return result
            else:
                # No embeddings to generate, call completion directly
                self.update_progress(3, 4, "No chunks found, calling completion callback")
                completion_task = mark_collection_completed.apply_async(
                    args=[document_results, collection_id]
                )

                self.update_progress(4, 4, "Collection processing completed (no embeddings)")

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

                result["message"] = f"Started metadata extraction for collection (no embeddings)"

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
