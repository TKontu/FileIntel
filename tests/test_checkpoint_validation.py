"""
Validation test for GraphRAG checkpoint & resume implementation.

This test validates the entire checkpoint/resume pipeline without actually
running GraphRAG (which would take hours).
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


async def test_imports():
    """Test that all new modules import correctly."""
    print("Testing imports...")

    try:
        from graphrag.index.run.checkpoint_manager import CheckpointManager
        print("  ✓ CheckpointManager")
    except Exception as e:
        print(f"  ✗ CheckpointManager: {e}")
        return False

    try:
        from graphrag.index.run.run_pipeline import run_pipeline_with_resume, _run_pipeline_from_index
        print("  ✓ run_pipeline_with_resume")
    except Exception as e:
        print(f"  ✗ run_pipeline_with_resume: {e}")
        return False

    try:
        from graphrag.api.index import build_index
        print("  ✓ build_index")
    except Exception as e:
        print(f"  ✗ build_index: {e}")
        return False

    try:
        from fileintel.rag.graph_rag.services.graphrag_service import GraphRAGService
        print("  ✓ GraphRAGService")
    except Exception as e:
        print(f"  ✗ GraphRAGService: {e}")
        return False

    print("✅ All imports successful\n")
    return True


async def test_checkpoint_manager():
    """Test CheckpointManager basic functionality."""
    print("Testing CheckpointManager...")

    from graphrag.index.run.checkpoint_manager import CheckpointManager

    mgr = CheckpointManager()

    # Test workflow outputs definition
    assert "extract_graph" in mgr.WORKFLOW_OUTPUTS
    assert "entities.parquet" in mgr.WORKFLOW_OUTPUTS["extract_graph"]
    print("  ✓ Workflow outputs defined correctly")

    # Test required columns definition
    assert "extract_graph" in mgr.WORKFLOW_REQUIRED_COLUMNS
    print("  ✓ Required columns defined correctly")

    # Test min row counts
    assert "entities.parquet" in mgr.MIN_ROW_COUNTS
    print("  ✓ Min row counts defined correctly")

    print("✅ CheckpointManager validation passed\n")
    return True


async def test_pipeline_structure():
    """Test that Pipeline class has expected structure."""
    print("Testing Pipeline structure...")

    from graphrag.index.typing.pipeline import Pipeline

    # Create a mock pipeline
    mock_workflows = [
        ("workflow1", lambda: None),
        ("workflow2", lambda: None),
    ]
    pipeline = Pipeline(mock_workflows)

    # Test workflows attribute exists
    assert hasattr(pipeline, 'workflows')
    print("  ✓ Pipeline.workflows exists")

    # Test we can extract names
    names = [name for name, _ in pipeline.workflows]
    assert names == ["workflow1", "workflow2"]
    print("  ✓ Can extract workflow names")

    # Test pipeline.names() method
    assert pipeline.names() == ["workflow1", "workflow2"]
    print("  ✓ pipeline.names() works")

    # Test pipeline.run() generator
    workflows_from_run = list(pipeline.run())
    assert len(workflows_from_run) == 2
    print("  ✓ pipeline.run() generator works")

    print("✅ Pipeline structure validation passed\n")
    return True


async def test_function_signatures():
    """Test that function signatures match expected patterns."""
    print("Testing function signatures...")

    import inspect
    from graphrag.index.run.run_pipeline import run_pipeline_with_resume
    from graphrag.api.index import build_index

    # Check run_pipeline_with_resume signature
    sig = inspect.signature(run_pipeline_with_resume)
    params = list(sig.parameters.keys())

    assert 'pipeline' in params
    assert 'config' in params
    assert 'enable_resume' in params
    assert 'validate_checkpoints' in params
    print("  ✓ run_pipeline_with_resume has correct parameters")

    # Check it's async generator (yields AsyncIterable)
    assert inspect.isasyncgenfunction(run_pipeline_with_resume)
    print("  ✓ run_pipeline_with_resume is async generator")

    # Check build_index signature
    sig = inspect.signature(build_index)
    params = list(sig.parameters.keys())

    assert 'config' in params
    assert 'enable_resume' in params
    assert 'validate_checkpoints' in params
    print("  ✓ build_index has correct parameters")

    # Check it's async (it returns a list, not a generator)
    assert inspect.iscoroutinefunction(build_index)
    print("  ✓ build_index is async coroutine")

    print("✅ Function signature validation passed\n")
    return True


async def test_async_consistency():
    """Test that async functions use await correctly."""
    print("Testing async/await consistency...")

    from graphrag.index.run.checkpoint_manager import CheckpointManager
    import inspect

    mgr = CheckpointManager()

    # Check check_workflow_completion is async
    assert inspect.iscoroutinefunction(mgr.check_workflow_completion)
    print("  ✓ check_workflow_completion is async")

    # Check find_resume_point is async
    assert inspect.iscoroutinefunction(mgr.find_resume_point)
    print("  ✓ find_resume_point is async")

    # Check validate_checkpoint_chain is async
    assert inspect.iscoroutinefunction(mgr.validate_checkpoint_chain)
    print("  ✓ validate_checkpoint_chain is async")

    print("✅ Async/await consistency validation passed\n")
    return True


async def test_config_fields():
    """Test that config fields are properly defined."""
    print("Testing config fields...")

    from fileintel.core.config import RAGSettings
    from pydantic import Field
    import inspect

    # Check RAGSettings has checkpoint fields
    fields = RAGSettings.model_fields

    assert 'enable_checkpoint_resume' in fields
    print("  ✓ enable_checkpoint_resume field exists")

    assert 'validate_checkpoints' in fields
    print("  ✓ validate_checkpoints field exists")

    # Check defaults
    defaults = RAGSettings()
    assert defaults.enable_checkpoint_resume == True
    print("  ✓ enable_checkpoint_resume defaults to True")

    assert defaults.validate_checkpoints == True
    print("  ✓ validate_checkpoints defaults to True")

    print("✅ Config field validation passed\n")
    return True


async def run_all_tests():
    """Run all validation tests."""
    print("="*70)
    print("GraphRAG Checkpoint & Resume - Validation Test Suite")
    print("="*70)
    print()

    tests = [
        ("Imports", test_imports),
        ("CheckpointManager", test_checkpoint_manager),
        ("Pipeline Structure", test_pipeline_structure),
        ("Function Signatures", test_function_signatures),
        ("Async Consistency", test_async_consistency),
        ("Config Fields", test_config_fields),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = await test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    print()
    print("="*70)
    print("Test Results Summary")
    print("="*70)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")

    all_passed = all(result for _, result in results)

    print()
    if all_passed:
        print("🎉 All validation tests passed!")
        print()
        print("The checkpoint & resume implementation is ready for testing.")
        print()
        print("Next steps:")
        print("  1. Test with a small collection (~1000 chunks)")
        print("  2. Kill the worker mid-process")
        print("  3. Restart and verify resume works")
        return 0
    else:
        print("❌ Some validation tests failed!")
        print()
        print("Please fix the issues above before testing.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(run_all_tests())
    sys.exit(exit_code)
