"""Test fixtures package for FileIntel."""

from .test_documents import (
    TestDocumentFixtures,
    test_documents,
    sample_text_doc,
    sample_markdown_doc,
    test_document_files,
    test_collection_data,
    test_job_data,
)

from .cleanup_manager import (
    TestCleanupManager,
    get_global_cleanup_manager,
    global_cleanup,
    test_cleanup,
    isolated_test_env,
    cleanup_after_test,
    temporary_test_collection,
    temporary_test_documents,
)

__all__ = [
    # Document fixtures
    "TestDocumentFixtures",
    "test_documents",
    "sample_text_doc",
    "sample_markdown_doc",
    "test_document_files",
    "test_collection_data",
    "test_job_data",
    # Cleanup management
    "TestCleanupManager",
    "get_global_cleanup_manager",
    "global_cleanup",
    "test_cleanup",
    "isolated_test_env",
    "cleanup_after_test",
    "temporary_test_collection",
    "temporary_test_documents",
]
