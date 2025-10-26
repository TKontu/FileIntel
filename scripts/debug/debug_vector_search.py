#!/usr/bin/env python3
"""Debug script to test vector search directly."""

import sys
sys.path.append('src')

from fileintel.core.config import get_config
from fileintel.storage.postgresql_storage import PostgreSQLStorage
from fileintel.storage.vector_search_storage import VectorSearchStorage

def test_vector_search():
    """Test vector search directly without going through the API."""

    # Get config and create storage
    config = get_config()
    storage = PostgreSQLStorage(config)
    vector_storage = VectorSearchStorage(storage.db)

    collection_id = "986ed8e2-9e5d-44f9-87d9-73e6b27771de"
    query = "what is this about"

    print(f"Testing vector search for collection: {collection_id}")
    print(f"Query: {query}")

    # First, let's try to generate a query embedding manually
    try:
        from fileintel.llm.providers.unified_llm_provider import UnifiedLLMProvider

        llm_provider = UnifiedLLMProvider(config, storage)
        query_embedding = llm_provider.generate_embedding(query)

        print(f"Generated query embedding with {len(query_embedding)} dimensions")

        # Test with very low similarity threshold
        for threshold in [0.0, 0.3, 0.5, 0.7]:
            print(f"\nTesting with similarity threshold: {threshold}")
            chunks = vector_storage.find_relevant_chunks_in_collection(
                collection_id=collection_id,
                query_embedding=query_embedding,
                limit=5,
                similarity_threshold=threshold
            )
            print(f"Found {len(chunks)} chunks")
            if chunks:
                print(f"Top similarity: {chunks[0].get('similarity', 'N/A')}")
                print(f"Preview: {chunks[0].get('text', '')[:100]}...")

    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()

    finally:
        storage.close()

if __name__ == "__main__":
    test_vector_search()