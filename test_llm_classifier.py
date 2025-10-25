#!/usr/bin/env python3
"""
Test script for LLM-based query classification.

This script demonstrates the new LLM classification functionality and
compares it with keyword-based classification.

Usage:
    python test_llm_classifier.py

Set environment variables to test different configurations:
    RAG_CLASSIFICATION_METHOD=llm        # LLM only
    RAG_CLASSIFICATION_METHOD=keyword    # Keyword only
    RAG_CLASSIFICATION_METHOD=hybrid     # LLM with keyword fallback (default)
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from fileintel.core.config import load_config, get_config
from fileintel.rag.query_classifier import QueryClassifier


# Test queries covering different classification scenarios
TEST_QUERIES = [
    # Clear VECTOR queries (factual lookups)
    "What is quantum computing?",
    "Define photosynthesis",
    "Tell me about the history of Rome",

    # Clear GRAPH queries (relationships)
    "How are quantum computing and AI related?",
    "Show me connections between entities in the network",
    "Compare classical and quantum computing",

    # HYBRID queries (both needed)
    "Compare X and Y and provide detailed information about each",
    "What are the relationships between stakeholders and what do the documents say?",

    # Ambiguous queries (interesting test cases)
    "Tell me everything about X",
    "Summarize the key points",
    "Who is involved in this?",
]


def test_classifier(config_path='config/default.yaml'):
    """Test the query classifier with various queries."""

    # Load config
    print("=" * 80)
    print("LLM Query Classification Test")
    print("=" * 80)
    print()

    config = load_config(config_path)

    # Display config
    print(f"Configuration:")
    print(f"  Classification Method: {config.rag.classification_method}")
    print(f"  Classification Model: {config.rag.classification_model}")
    print(f"  Cache Enabled: {config.rag.classification_cache_enabled}")
    print(f"  Cache TTL: {config.rag.classification_cache_ttl}s")
    print()

    # Create classifier
    classifier = QueryClassifier(config)

    # Test each query
    print("Testing Queries:")
    print("-" * 80)

    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\n[{i}/{len(TEST_QUERIES)}] Query: \"{query}\"")

        try:
            result = classifier.classify(query)

            # Display result
            print(f"  → Type: {result['type']}")
            print(f"  → Confidence: {result['confidence']:.2f}")
            print(f"  → Method: {result.get('method', 'unknown')}")
            print(f"  → Latency: {result.get('latency_ms', 0)}ms")
            print(f"  → Cached: {result.get('cached', False)}")
            if 'fallback_used' in result:
                print(f"  → Fallback Used: {result['fallback_used']}")
            print(f"  → Reasoning: {result.get('reasoning', 'N/A')}")

        except Exception as e:
            print(f"  ✗ ERROR: {e}")

    print()
    print("-" * 80)

    # Display cache stats
    cache_stats = classifier.get_cache_stats()
    print("\nCache Statistics:")
    print(f"  Hits: {cache_stats['hits']}")
    print(f"  Misses: {cache_stats['misses']}")
    print(f"  Hit Rate: {cache_stats['hit_rate_percent']:.1f}%")
    print(f"  Cache Size: {cache_stats['cache_size']} entries")
    print()

    # Test cache by running same queries again
    if cache_stats['enabled']:
        print("Testing cache with repeated queries...")
        print("-" * 80)

        # Run first 3 queries again
        for query in TEST_QUERIES[:3]:
            result = classifier.classify(query)
            print(f"  Query: \"{query[:40]}...\"")
            print(f"    Cached: {result.get('cached', False)}, "
                  f"Type: {result['type']}, "
                  f"Latency: {result.get('latency_ms', 0)}ms")

        # Final cache stats
        cache_stats = classifier.get_cache_stats()
        print(f"\n  Final cache hit rate: {cache_stats['hit_rate_percent']:.1f}%")
        print()

    print("=" * 80)
    print("Test Complete!")
    print("=" * 80)


if __name__ == "__main__":
    try:
        test_classifier()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
