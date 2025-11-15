# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing embed_text, load_strategy and create_row_from_embedding_data methods definition."""

import logging
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from graphrag.cache.pipeline_cache import PipelineCache
from graphrag.callbacks.workflow_callbacks import WorkflowCallbacks
from graphrag.config.embeddings import create_collection_name
from graphrag.index.operations.embed_text.strategies.typing import TextEmbeddingStrategy
from graphrag.vector_stores.base import BaseVectorStore, VectorStoreDocument
from graphrag.vector_stores.factory import VectorStoreFactory

logger = logging.getLogger(__name__)

# Per Azure OpenAI Limits
# https://learn.microsoft.com/en-us/azure/ai-services/openai/reference
DEFAULT_EMBEDDING_BATCH_SIZE = 500


class TextEmbedStrategyType(str, Enum):
    """TextEmbedStrategyType class definition."""

    openai = "openai"
    mock = "mock"

    def __repr__(self):
        """Get a string representation."""
        return f'"{self.value}"'


async def embed_text(
    input: pd.DataFrame,
    callbacks: WorkflowCallbacks,
    cache: PipelineCache,
    embed_column: str,
    strategy: dict,
    embedding_name: str,
    id_column: str = "id",
    title_column: str | None = None,
):
    """Embed a piece of text into a vector space. The operation outputs a new column containing a mapping between doc_id and vector."""
    logger.debug(f"embed_text called for embedding_name={embedding_name}, input shape={input.shape}")
    vector_store_config = strategy.get("vector_store")
    logger.debug(f"vector_store_config present: {vector_store_config is not None}")

    if vector_store_config:
        logger.debug(f"Using vector store path for embeddings")
        collection_name = _get_collection_name(vector_store_config, embedding_name)
        logger.debug(f"Collection name obtained: {collection_name}")

        logger.debug(f"About to create vector store...")
        # CRITICAL FIX: Run blocking I/O in executor to prevent event loop blocking
        import asyncio
        loop = asyncio.get_event_loop()
        vector_store: BaseVectorStore = await loop.run_in_executor(
            None, _create_vector_store, vector_store_config, collection_name
        )
        logger.debug(f"Vector store created successfully")

        vector_store_workflow_config = vector_store_config.get(
            embedding_name, vector_store_config
        )
        logger.debug(f"About to start embedding with vector store...")
        result = await _text_embed_with_vector_store(
            input=input,
            callbacks=callbacks,
            cache=cache,
            embed_column=embed_column,
            strategy=strategy,
            vector_store=vector_store,
            vector_store_config=vector_store_workflow_config,
            id_column=id_column,
            title_column=title_column,
        )
        logger.debug(f"Embedding with vector store completed")
        return result

    logger.debug(f"Using in-memory path for embeddings")
    return await _text_embed_in_memory(
        input=input,
        callbacks=callbacks,
        cache=cache,
        embed_column=embed_column,
        strategy=strategy,
    )


async def _text_embed_in_memory(
    input: pd.DataFrame,
    callbacks: WorkflowCallbacks,
    cache: PipelineCache,
    embed_column: str,
    strategy: dict,
):
    strategy_type = strategy["type"]
    strategy_exec = load_strategy(strategy_type)
    strategy_config = {**strategy}

    texts: list[str] = input[embed_column].to_numpy().tolist()
    result = await strategy_exec(texts, callbacks, cache, strategy_config)

    return result.embeddings


async def _text_embed_with_vector_store(
    input: pd.DataFrame,
    callbacks: WorkflowCallbacks,
    cache: PipelineCache,
    embed_column: str,
    strategy: dict[str, Any],
    vector_store: BaseVectorStore,
    vector_store_config: dict,
    id_column: str = "id",
    title_column: str | None = None,
):
    logger.debug(f"_text_embed_with_vector_store started, input shape: {input.shape}")
    strategy_type = strategy["type"]
    logger.debug(f"Loading strategy: {strategy_type}")
    strategy_exec = load_strategy(strategy_type)
    strategy_config = {**strategy}

    # Get vector-storage configuration
    insert_batch_size: int = (
        vector_store_config.get("batch_size") or DEFAULT_EMBEDDING_BATCH_SIZE
    )
    logger.debug(f"Batch size: {insert_batch_size}")

    overwrite: bool = vector_store_config.get("overwrite", True)
    logger.debug(f"Overwrite mode: {overwrite}")

    if embed_column not in input.columns:
        msg = f"Column {embed_column} not found in input dataframe with columns {input.columns}"
        raise ValueError(msg)
    title = title_column or embed_column
    if title not in input.columns:
        msg = (
            f"Column {title} not found in input dataframe with columns {input.columns}"
        )
        raise ValueError(msg)
    if id_column not in input.columns:
        msg = f"Column {id_column} not found in input dataframe with columns {input.columns}"
        raise ValueError(msg)

    total_rows = 0
    for row in input[embed_column]:
        if isinstance(row, list):
            total_rows += len(row)
        else:
            total_rows += 1

    i = 0
    starting_index = 0

    all_results = []

    num_total_batches = (input.shape[0] + insert_batch_size - 1) // insert_batch_size
    while insert_batch_size * i < input.shape[0]:
        logger.info(
            "uploading text embeddings batch %d/%d of size %d to vector store",
            i + 1,
            num_total_batches,
            insert_batch_size,
        )
        batch = input.iloc[insert_batch_size * i : insert_batch_size * (i + 1)]
        texts: list[str] = batch[embed_column].to_numpy().tolist()
        titles: list[str] = batch[title].to_numpy().tolist()
        ids: list[str] = batch[id_column].to_numpy().tolist()
        result = await strategy_exec(texts, callbacks, cache, strategy_config)
        # CRITICAL: Preserve None embeddings to maintain length alignment with input DataFrame
        # The vector store loading will skip None values, but the returned list must match input length
        if result.embeddings:
            all_results.extend(result.embeddings)

        vectors = result.embeddings or []
        documents: list[VectorStoreDocument] = []
        for doc_id, doc_text, doc_title, doc_vector in zip(
            ids, texts, titles, vectors, strict=True
        ):
            # Skip documents with None embeddings (failed embedding attempts)
            if doc_vector is None:
                # Provide diagnostic information to help debug why embedding failed
                text_preview = doc_text[:200] if len(doc_text) > 200 else doc_text
                logger.warning(
                    f"Skipping document {doc_id} - embedding failed. "
                    f"Text length: {len(doc_text)}, "
                    f"Preview: {text_preview!r}. "
                    f"Check earlier ERROR logs for details about why this embedding failed."
                )
                continue

            if type(doc_vector) is np.ndarray:
                doc_vector = doc_vector.tolist()
            document = VectorStoreDocument(
                id=doc_id,
                text=doc_text,
                vector=doc_vector,
                attributes={"title": doc_title},
            )
            documents.append(document)

        # CRITICAL FIX: Run blocking I/O in executor to prevent event loop blocking
        import asyncio
        loop = asyncio.get_event_loop()

        # Skip loading if all embeddings failed for this batch
        if documents:
            logger.info(f"About to load {len(documents)} documents to LanceDB (batch {i+1}/{num_total_batches}, overwrite={overwrite and i == 0})")
            await loop.run_in_executor(
                None,
                lambda: vector_store.load_documents(documents, overwrite and i == 0)
            )
            logger.info(f"Successfully loaded {len(documents)} documents to LanceDB (batch {i+1}/{num_total_batches})")
        else:
            logger.warning(f"Batch {i+1}/{num_total_batches} had no valid embeddings - skipping LanceDB load")
        starting_index += len(documents)
        i += 1

    return all_results


def _create_vector_store(
    vector_store_config: dict, collection_name: str
) -> BaseVectorStore:
    logger.debug(f"_create_vector_store called with collection_name={collection_name}")
    vector_store_type: str = str(vector_store_config.get("type"))
    logger.debug(f"vector_store_type={vector_store_type}")

    if collection_name:
        vector_store_config.update({"collection_name": collection_name})

    logger.debug(f"Creating vector store instance...")
    vector_store = VectorStoreFactory().create_vector_store(
        vector_store_type, kwargs=vector_store_config
    )
    logger.debug(f"Vector store instance created: {type(vector_store)}")

    logger.debug(f"Connecting to vector store...")
    vector_store.connect(**vector_store_config)
    logger.debug(f"Vector store connected successfully")
    return vector_store


def _get_collection_name(vector_store_config: dict, embedding_name: str) -> str:
    container_name = vector_store_config.get("container_name", "default")
    collection_name = create_collection_name(container_name, embedding_name)

    msg = f"using vector store {vector_store_config.get('type')} with container_name {container_name} for embedding {embedding_name}: {collection_name}"
    logger.info(msg)
    return collection_name


def load_strategy(strategy: TextEmbedStrategyType) -> TextEmbeddingStrategy:
    """Load strategy method definition."""
    match strategy:
        case TextEmbedStrategyType.openai:
            from graphrag.index.operations.embed_text.strategies.openai import (
                run as run_openai,
            )

            return run_openai
        case TextEmbedStrategyType.mock:
            from graphrag.index.operations.embed_text.strategies.mock import (
                run as run_mock,
            )

            return run_mock
        case _:
            msg = f"Unknown strategy: {strategy}"
            raise ValueError(msg)
