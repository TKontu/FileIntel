"""
Debug script to show the exact RAG prompt sent to the LLM
"""
import asyncio
from fileintel.core.config import Settings
from fileintel.storage.postgresql_storage import PostgreSQLStorage
from fileintel.rag.vector_rag.services.vector_rag_service import VectorRAGService

async def main():
    settings = Settings()
    storage = PostgreSQLStorage(settings)

    # Get a collection
    collections = storage.list_collections()
    if not collections:
        print("No collections found")
        return

    collection_id = collections[0]['id']
    collection_name = collections[0]['name']

    print(f"Testing with collection: {collection_name} ({collection_id})\n")

    # Create vector RAG service
    vector_rag = VectorRAGService(settings, storage)

    # Test query
    query = "concurrent engineering is"

    print("="*80)
    print("QUERY:", query)
    print("="*80)

    # Get chunks (this is what the service does internally)
    from fileintel.llm_integration.embedding_provider import OpenAIEmbeddingProvider
    embedding_provider = OpenAIEmbeddingProvider(settings=settings)

    query_embedding = embedding_provider.get_embeddings([query])[0]
    similar_chunks = storage.find_relevant_chunks_in_collection(
        collection_id=collection_id,
        query_embedding=query_embedding,
        limit=5,
        similarity_threshold=0.0
    )

    print(f"\nFound {len(similar_chunks)} chunks\n")

    # Build the context exactly as the LLM provider does
    from fileintel.citation import format_in_text_citation

    context_parts = []
    for i, chunk in enumerate(similar_chunks[:8], 1):
        chunk_text = chunk.get("text", "")

        # Format citation
        source_info = format_in_text_citation(chunk)

        # Show what metadata is available
        print(f"--- Chunk {i} ---")
        print(f"Citation: {source_info}")
        chunk_metadata = chunk.get("metadata", chunk.get("chunk_metadata", {}))
        print(f"Metadata keys: {list(chunk_metadata.keys())}")
        print(f"  page_number: {chunk_metadata.get('page_number')}")
        print(f"  page_range: {chunk_metadata.get('page_range')}")
        print(f"  pages: {chunk_metadata.get('pages')}")
        print(f"Text preview: {chunk_text[:100]}...")
        print()

        context_parts.append(f"[{source_info}]: {chunk_text}")

    context = "\n\n".join(context_parts)

    # Build the full prompt
    from fileintel.llm_integration.unified_provider import UnifiedLLMProvider
    llm = UnifiedLLMProvider(settings, storage)

    query_type = "general"
    prompt = llm._build_rag_prompt(query, context, query_type)

    print("\n" + "="*80)
    print("FULL PROMPT SENT TO LLM:")
    print("="*80)
    print(prompt)
    print("="*80)

if __name__ == "__main__":
    asyncio.run(main())
