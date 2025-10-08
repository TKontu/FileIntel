#!/usr/bin/env python3
"""Test script to identify Celery workflow issues."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

def test_task_signatures():
    """Test that all tasks can create signatures properly."""
    from fileintel.tasks.workflow_tasks import (
        complete_collection_analysis,
        generate_collection_metadata_and_embeddings,
        generate_collection_metadata,
        generate_collection_embeddings_simple,
        mark_collection_completed
    )

    print("Testing task signature creation...")

    # Test each task can create a signature
    tasks_to_test = [
        ("complete_collection_analysis", complete_collection_analysis),
        ("generate_collection_metadata_and_embeddings", generate_collection_metadata_and_embeddings),
        ("generate_collection_metadata", generate_collection_metadata),
        ("generate_collection_embeddings_simple", generate_collection_embeddings_simple),
        ("mark_collection_completed", mark_collection_completed),
    ]

    for task_name, task in tasks_to_test:
        try:
            sig = task.s()
            print(f"✓ {task_name}: {type(sig).__name__}")
            print(f"  Has apply_async: {hasattr(sig, 'apply_async')}")
        except Exception as e:
            print(f"✗ {task_name}: {e}")

    # Test chord construction
    print("\nTesting chord construction...")
    try:
        from celery import chord
        from fileintel.tasks.document_tasks import process_document

        # Create dummy signatures
        doc_sigs = [process_document.s(file_path="/tmp/test.pdf", document_id="test", collection_id="test")]
        callback_sig = generate_collection_metadata_and_embeddings.s(collection_id="test")

        print(f"Document signatures type: {type(doc_sigs)}")
        print(f"Callback signature type: {type(callback_sig)}")

        # Try to create chord
        chord_obj = chord(doc_sigs)(callback_sig)
        print(f"Chord object type: {type(chord_obj)}")
        print(f"Chord has apply_async: {hasattr(chord_obj, 'apply_async')}")

    except Exception as e:
        print(f"✗ Chord construction failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_task_signatures()
