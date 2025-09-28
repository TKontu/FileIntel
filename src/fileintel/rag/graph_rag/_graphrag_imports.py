"""
GraphRAG import utilities.

Simple optional dependency handling for GraphRAG components.
"""

# Try to import GraphRAG - if not available, set everything to None
try:
    from graphrag.api.query import global_search, local_search
    from graphrag.api.index import build_index
    from graphrag.config.models.graph_rag_config import GraphRagConfig
    from graphrag.config.models.text_embedding_config import TextEmbeddingConfig
    from graphrag.config.models.storage_config import StorageConfig
    from graphrag.config.models.input_config import InputConfig
    from graphrag.query.indexer_adapters import (
        read_indexer_entities,
        read_indexer_communities,
    )

    # Handle optional components
    try:
        from graphrag.config.models.output_config import OutputConfig
    except ImportError:
        OutputConfig = StorageConfig

    try:
        from graphrag.config.models.language_model_config import LanguageModelConfig
        from graphrag.config.enums import ModelType
    except ImportError:

        class LanguageModelConfig:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        class ModelType:
            OpenAIChat = "openai_chat"
            OpenAIEmbedding = "openai_embedding"

    GRAPHRAG_AVAILABLE = True

except ImportError:
    # GraphRAG not available - set everything to None for None checks
    global_search = None
    local_search = None
    build_index = None
    GraphRagConfig = None
    StorageConfig = None
    InputConfig = None
    OutputConfig = None
    LanguageModelConfig = None
    ModelType = None
    read_indexer_entities = None
    read_indexer_communities = None
    TextEmbeddingConfig = None

    GRAPHRAG_AVAILABLE = False


__all__ = [
    "global_search",
    "local_search",
    "build_index",
    "GraphRagConfig",
    "StorageConfig",
    "InputConfig",
    "OutputConfig",
    "LanguageModelConfig",
    "ModelType",
    "read_indexer_entities",
    "read_indexer_communities",
    "TextEmbeddingConfig",
    "GRAPHRAG_AVAILABLE",
]
