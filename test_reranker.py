#!/usr/bin/env python3
"""
Test script for Reranker Service.

This script demonstrates the reranking functionality with sample documents and queries.
Shows how reranking can improve result relevance by re-scoring retrieval results.

Usage:
    python test_reranker.py

Requirements:
    - FlagEmbedding installed (pip install FlagEmbedding)
    - Reranking enabled in config (RAG_RERANKING_ENABLED=true)
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from fileintel.core.config import get_config
from fileintel.rag.reranker_service import RerankerService


def main():
    """Test reranker with sample passages."""
    print("=" * 80)
    print("FileIntel Reranker Service Test")
    print("=" * 80)
    print()

    # Load config
    config = get_config()

    # Check if reranking is enabled
    if not config.rag.reranking.enabled:
        print("⚠️  Reranking is DISABLED in config")
        print("   Enable it by setting RAG_RERANKING_ENABLED=true in .env")
        print()
        print("   To test anyway, we'll enable it temporarily...")
        config.rag.reranking.enabled = True
        print("   ✓ Enabled for this test session")
        print()

    print(f"Model: {config.rag.reranking.model_name}")
    print(f"Model Type: {config.rag.reranking.model_type}")
    print(f"Initial K: {config.rag.reranking.initial_retrieval_k}")
    print(f"Final K: {config.rag.reranking.final_top_k}")
    print()

    # Initialize reranker
    print("Initializing RerankerService...")
    try:
        reranker = RerankerService(config)
        print("✓ RerankerService initialized successfully")
        print()
    except ImportError as e:
        print(f"✗ Failed to initialize: {e}")
        print()
        print("Install FlagEmbedding with:")
        print("  pip install FlagEmbedding")
        return 1
    except Exception as e:
        print(f"✗ Failed to initialize: {e}")
        return 1

    # Test queries and sample passages
    test_cases = [
        {
            "query": "What are the benefits of machine learning?",
            "passages": [
                {
                    "content": "Machine learning algorithms can process large amounts of data quickly.",
                    "similarity_score": 0.75,
                    "source": "doc1.pdf",
                },
                {
                    "content": "The weather tomorrow will be sunny with a high of 75 degrees.",
                    "similarity_score": 0.72,  # High score but irrelevant
                    "source": "doc2.pdf",
                },
                {
                    "content": "Machine learning provides automated insights, reduces human error, and improves decision-making through pattern recognition.",
                    "similarity_score": 0.68,  # Lower score but very relevant
                    "source": "doc3.pdf",
                },
                {
                    "content": "Coffee beans are grown in tropical regions near the equator.",
                    "similarity_score": 0.70,  # High score but irrelevant
                    "source": "doc4.pdf",
                },
                {
                    "content": "Key advantages of ML include scalability, adaptability, and continuous learning from new data.",
                    "similarity_score": 0.65,  # Lower score but relevant
                    "source": "doc5.pdf",
                },
            ],
        },
        {
            "query": "How does climate change affect biodiversity?",
            "passages": [
                {
                    "content": "Climate change leads to habitat loss and species extinction through temperature shifts.",
                    "similarity_score": 0.80,
                    "source": "eco1.pdf",
                },
                {
                    "content": "The stock market has been volatile in recent months.",
                    "similarity_score": 0.76,  # High score but irrelevant
                    "source": "eco2.pdf",
                },
                {
                    "content": "Rising temperatures disrupt ecosystems, alter migration patterns, and threaten species survival.",
                    "similarity_score": 0.73,
                    "source": "eco3.pdf",
                },
                {
                    "content": "Biodiversity loss accelerates as climate zones shift faster than species can adapt.",
                    "similarity_score": 0.69,
                    "source": "eco4.pdf",
                },
            ],
        },
    ]

    # Run reranking tests
    for i, test_case in enumerate(test_cases, 1):
        print(f"{'=' * 80}")
        print(f"Test Case {i}: {test_case['query']}")
        print(f"{'=' * 80}")
        print()

        print(f"Original Ranking (by similarity score):")
        print("-" * 80)
        for j, passage in enumerate(test_case["passages"], 1):
            content_preview = passage["content"][:60] + "..." if len(passage["content"]) > 60 else passage["content"]
            print(f"  {j}. [Score: {passage['similarity_score']:.3f}] {content_preview}")
        print()

        # Rerank passages
        print("Reranking passages...")
        reranked = reranker.rerank(
            query=test_case["query"],
            passages=test_case["passages"],
            top_k=config.rag.reranking.final_top_k,
        )
        print()

        print(f"Reranked Results (top {len(reranked)}):")
        print("-" * 80)
        for j, passage in enumerate(reranked, 1):
            content_preview = passage["content"][:60] + "..." if len(passage["content"]) > 60 else passage["content"]
            orig_score = passage.get("original_score", 0.0)
            rerank_score = passage.get("reranked_score", 0.0)
            orig_rank = passage.get("original_rank", 0)

            # Indicate if ranking changed significantly
            rank_change = ""
            if "original_rank" in passage:
                change = orig_rank - j
                if change > 0:
                    rank_change = f" ↑{change}"
                elif change < 0:
                    rank_change = f" ↓{abs(change)}"

            print(f"  {j}. [Orig: {orig_score:.3f} → Rerank: {rerank_score:.3f}]{rank_change}")
            print(f"     {content_preview}")
        print()

    # Print statistics
    print("=" * 80)
    print("Reranker Statistics")
    print("=" * 80)
    stats = reranker.get_stats()
    print(f"Model: {stats['model_name']}")
    print(f"Total Reranks: {stats['total_reranks']}")
    print(f"Average Latency: {stats['average_latency_ms']:.1f}ms")
    print()

    print("✓ Test completed successfully!")
    print()
    print("Next Steps:")
    print("  1. Enable reranking in production: RAG_RERANKING_ENABLED=true")
    print("  2. Adjust initial_k and final_k in config for your use case")
    print("  3. Monitor latency impact and adjust batch_size if needed")
    print("  4. Consider using min_score_threshold to filter low-relevance results")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
