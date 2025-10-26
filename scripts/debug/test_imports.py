#!/usr/bin/env python3
"""Test script to verify GraphRAG imports work correctly."""

import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def test_graphrag_imports():
    """Test GraphRAG imports with fallback system."""
    try:
        from fileintel.rag.graph_rag._graphrag_imports import (
            LanguageModelConfig,
            ModelType,
            GraphRagConfig,
            StorageConfig,
            InputConfig,
            OutputConfig,
        )

        print("✓ GraphRAG base imports successful")

        # Test that classes can be instantiated (might use fallbacks)
        try:
            # This might fail in production but should work with fallbacks
            config = LanguageModelConfig(model="test", type=ModelType.OpenAIChat)
            print("✓ LanguageModelConfig instantiation successful")
        except RuntimeError as e:
            if "GraphRAG is not available" in str(e):
                print("✓ GraphRAG fallback working correctly")
            else:
                print(f"✗ Unexpected error: {e}")

    except ImportError as e:
        print(f"✗ GraphRAG imports failed: {e}")
        return False

    return True


def test_config_adapter():
    """Test config adapter imports."""
    try:
        from fileintel.rag.graph_rag.adapters.config_adapter import (
            GraphRAGConfigAdapter,
        )

        print("✓ GraphRAG config adapter import successful")

        adapter = GraphRAGConfigAdapter()
        print("✓ GraphRAGConfigAdapter instantiation successful")

    except ImportError as e:
        print(f"✗ Config adapter import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Config adapter error: {e}")
        return False

    return True


def main():
    """Run all import tests."""
    print("Testing GraphRAG imports...")

    success = True
    success &= test_graphrag_imports()
    success &= test_config_adapter()

    if success:
        print("\n✓ All imports working correctly!")
        return 0
    else:
        print("\n✗ Some imports failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
