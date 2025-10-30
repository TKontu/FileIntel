# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing fnllm model provider definitions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
from fnllm.base.config import RetryStrategy
from fnllm.openai import (
    create_openai_chat_llm,
    create_openai_client,
    create_openai_embeddings_llm,
)
from openai import AsyncOpenAI

from graphrag.language_model.providers.fnllm.events import FNLLMEvents
from graphrag.language_model.providers.fnllm.utils import (
    _create_cache,
    _create_error_handler,
    _create_openai_config,
    run_coroutine_sync,
)
from graphrag.language_model.response.base import (
    BaseModelOutput,
    BaseModelResponse,
    ModelResponse,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Generator

    from fnllm.openai.types.client import OpenAIChatLLM as FNLLMChatLLM
    from fnllm.openai.types.client import OpenAIEmbeddingsLLM as FNLLMEmbeddingLLM

    from graphrag.cache.pipeline_cache import PipelineCache
    from graphrag.callbacks.workflow_callbacks import WorkflowCallbacks
    from graphrag.config.models.language_model_config import (
        LanguageModelConfig,
    )


class OpenAIChatFNLLM:
    """An OpenAI Chat Model provider using the fnllm library."""

    model: FNLLMChatLLM

    def __init__(
        self,
        *,
        name: str,
        config: LanguageModelConfig,
        callbacks: WorkflowCallbacks | None = None,
        cache: PipelineCache | None = None,
    ) -> None:
        model_config = _create_openai_config(config, azure=False)
        error_handler = _create_error_handler(callbacks) if callbacks else None
        model_cache = _create_cache(cache, name)
        client = create_openai_client(model_config)
        self.model = create_openai_chat_llm(
            model_config,
            client=client,
            cache=model_cache,
            events=FNLLMEvents(error_handler) if error_handler else None,
        )
        self.config = config

    async def achat(
        self, prompt: str, history: list | None = None, **kwargs
    ) -> ModelResponse:
        """
        Chat with the Model using the given prompt.

        Args:
            prompt: The prompt to chat with.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            The response from the Model.
        """
        import time
        import logging
        logger = logging.getLogger(__name__)

        fnllm_start = time.time()
        logger.debug(f"FNLLM DEBUG: Starting achat call, prompt length: {len(prompt)}")

        if history is None:
            model_start = time.time()
            logger.debug(f"FNLLM DEBUG: Calling self.model() without history")
            response = await self.model(prompt, **kwargs)
            model_end = time.time()
            logger.debug(f"FNLLM DEBUG: self.model() call completed in {model_end - model_start:.2f} seconds")
        else:
            model_start = time.time()
            logger.debug(f"FNLLM DEBUG: Calling self.model() with history (length: {len(history)})")
            response = await self.model(prompt, history=history, **kwargs)
            model_end = time.time()
            logger.debug(f"FNLLM DEBUG: self.model() call completed in {model_end - model_start:.2f} seconds")

        response_creation_start = time.time()
        logger.debug(f"FNLLM DEBUG: Creating BaseModelResponse")
        result = BaseModelResponse(
            output=BaseModelOutput(
                content=response.output.content,
                full_response=response.output.raw_model.to_dict(),
            ),
            parsed_response=response.parsed_json,
            history=response.history,
            cache_hit=response.cache_hit,
            tool_calls=response.tool_calls,
            metrics=response.metrics,
        )
        response_creation_end = time.time()
        logger.debug(f"FNLLM DEBUG: BaseModelResponse creation completed in {response_creation_end - response_creation_start:.2f} seconds")

        fnllm_end = time.time()
        logger.debug(f"FNLLM DEBUG: Total achat completed in {fnllm_end - fnllm_start:.2f} seconds")
        return result

    async def achat_stream(
        self, prompt: str, history: list | None = None, **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Stream Chat with the Model using the given prompt.

        Args:
            prompt: The prompt to chat with.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            A generator that yields strings representing the response.
        """
        if history is None:
            response = await self.model(prompt, stream=True, **kwargs)
        else:
            response = await self.model(prompt, history=history, stream=True, **kwargs)
        async for chunk in response.output.content:
            if chunk is not None:
                yield chunk

    def chat(self, prompt: str, history: list | None = None, **kwargs) -> ModelResponse:
        """
        Chat with the Model using the given prompt.

        Args:
            prompt: The prompt to chat with.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            The response from the Model.
        """
        return run_coroutine_sync(self.achat(prompt, history=history, **kwargs))

    def chat_stream(
        self, prompt: str, history: list | None = None, **kwargs
    ) -> Generator[str, None]:
        """
        Stream Chat with the Model using the given prompt.

        Args:
            prompt: The prompt to chat with.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            A generator that yields strings representing the response.
        """
        msg = "chat_stream is not supported for synchronous execution"
        raise NotImplementedError(msg)


class OpenAIEmbeddingFNLLM:
    """An OpenAI Embedding Model provider using the fnllm library."""

    model: FNLLMEmbeddingLLM

    def __init__(
        self,
        *,
        name: str,
        config: LanguageModelConfig,
        callbacks: WorkflowCallbacks | None = None,
        cache: PipelineCache | None = None,
    ) -> None:
        model_config = _create_openai_config(config, azure=False)
        error_handler = _create_error_handler(callbacks) if callbacks else None
        model_cache = _create_cache(cache, name)
        #client = create_openai_client(model_config) ## Default client removed to ensure graceful batch processing.

        # Start of customization

        # Create custom httpx client with proper connection pool settings
        # Prevents race condition where connections close while requests are in flight
        # See: docs/graphrag_embedding_todo.md for detailed root cause analysis
        http_client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=100,
                max_keepalive_connections=100,  # Keep all connections alive (vs default 20)
                keepalive_expiry=600,  # 10 minutes (vs default 5 seconds)
            ),
            timeout=httpx.Timeout(timeout=model_config.timeout or 60.0),
        )

        # Create OpenAI client with custom http_client
        client = AsyncOpenAI(
            api_key=model_config.api_key,
            base_url=model_config.base_url,
            organization=model_config.organization,
            timeout=model_config.timeout,
            max_retries=model_config.max_retries if model_config.retry_strategy == RetryStrategy.NATIVE else 0,
            http_client=http_client,
        )

        # End of customization

        self.model = create_openai_embeddings_llm(
            model_config,
            client=client,
            cache=model_cache,
            events=FNLLMEvents(error_handler) if error_handler else None,
        )
        self.config = config

    async def aembed_batch(self, text_list: list[str], **kwargs) -> list[list[float]]:
        """
        Embed the given text using the Model.

        Args:
            text: The text to embed.
            kwargs: Additional arguments to pass to the LLM.

        Returns
        -------
            The embeddings of the text.
        """
        response = await self.model(text_list, **kwargs)
        if response.output.embeddings is None:
            msg = "No embeddings found in response"
            raise ValueError(msg)
        embeddings: list[list[float]] = response.output.embeddings
        return embeddings

    async def aembed(self, text: str, **kwargs) -> list[float]:
        """
        Embed the given text using the Model.

        Args:
            text: The text to embed.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            The embeddings of the text.
        """
        response = await self.model([text], **kwargs)
        if response.output.embeddings is None:
            msg = "No embeddings found in response"
            raise ValueError(msg)
        embeddings: list[float] = response.output.embeddings[0]
        return embeddings

    def embed_batch(self, text_list: list[str], **kwargs) -> list[list[float]]:
        """
        Embed the given text using the Model.

        Args:
            text: The text to embed.
            kwargs: Additional arguments to pass to the LLM.

        Returns
        -------
            The embeddings of the text.
        """
        return run_coroutine_sync(self.aembed_batch(text_list, **kwargs))

    def embed(self, text: str, **kwargs) -> list[float]:
        """
        Embed the given text using the Model.

        Args:
            text: The text to embed.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            The embeddings of the text.
        """
        return run_coroutine_sync(self.aembed(text, **kwargs))


class AzureOpenAIChatFNLLM:
    """An Azure OpenAI Chat LLM provider using the fnllm library."""

    model: FNLLMChatLLM

    def __init__(
        self,
        *,
        name: str,
        config: LanguageModelConfig,
        callbacks: WorkflowCallbacks | None = None,
        cache: PipelineCache | None = None,
    ) -> None:
        model_config = _create_openai_config(config, azure=True)
        error_handler = _create_error_handler(callbacks) if callbacks else None
        model_cache = _create_cache(cache, name)
        client = create_openai_client(model_config)
        self.model = create_openai_chat_llm(
            model_config,
            client=client,
            cache=model_cache,
            events=FNLLMEvents(error_handler) if error_handler else None,
        )
        self.config = config

    async def achat(
        self, prompt: str, history: list | None = None, **kwargs
    ) -> ModelResponse:
        """
        Chat with the Model using the given prompt.

        Args:
            prompt: The prompt to chat with.
            history: The conversation history.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            The response from the Model.
        """
        import time
        import logging
        logger = logging.getLogger(__name__)

        fnllm_azure_start = time.time()
        logger.debug(f"FNLLM AZURE DEBUG: Starting achat call, prompt length: {len(prompt)}")

        if history is None:
            model_start = time.time()
            logger.debug(f"FNLLM AZURE DEBUG: Calling self.model() without history")
            response = await self.model(prompt, **kwargs)
            model_end = time.time()
            logger.debug(f"FNLLM AZURE DEBUG: self.model() call completed in {model_end - model_start:.2f} seconds")
        else:
            model_start = time.time()
            logger.debug(f"FNLLM AZURE DEBUG: Calling self.model() with history (length: {len(history)})")
            response = await self.model(prompt, history=history, **kwargs)
            model_end = time.time()
            logger.debug(f"FNLLM AZURE DEBUG: self.model() call completed in {model_end - model_start:.2f} seconds")

        response_creation_start = time.time()
        logger.debug(f"FNLLM AZURE DEBUG: Creating BaseModelResponse")
        result = BaseModelResponse(
            output=BaseModelOutput(
                content=response.output.content,
                full_response=response.output.raw_model.to_dict(),
            ),
            parsed_response=response.parsed_json,
            history=response.history,
            cache_hit=response.cache_hit,
            tool_calls=response.tool_calls,
            metrics=response.metrics,
        )
        response_creation_end = time.time()
        logger.debug(f"FNLLM AZURE DEBUG: BaseModelResponse creation completed in {response_creation_end - response_creation_start:.2f} seconds")

        fnllm_azure_end = time.time()
        logger.debug(f"FNLLM AZURE DEBUG: Total achat completed in {fnllm_azure_end - fnllm_azure_start:.2f} seconds")
        return result

    async def achat_stream(
        self, prompt: str, history: list | None = None, **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Stream Chat with the Model using the given prompt.

        Args:
            prompt: The prompt to chat with.
            history: The conversation history.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            A generator that yields strings representing the response.
        """
        if history is None:
            response = await self.model(prompt, stream=True, **kwargs)
        else:
            response = await self.model(prompt, history=history, stream=True, **kwargs)
        async for chunk in response.output.content:
            if chunk is not None:
                yield chunk

    def chat(self, prompt: str, history: list | None = None, **kwargs) -> ModelResponse:
        """
        Chat with the Model using the given prompt.

        Args:
            prompt: The prompt to chat with.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            The response from the Model.
        """
        return run_coroutine_sync(self.achat(prompt, history=history, **kwargs))

    def chat_stream(
        self, prompt: str, history: list | None = None, **kwargs
    ) -> Generator[str, None]:
        """
        Stream Chat with the Model using the given prompt.

        Args:
            prompt: The prompt to chat with.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            A generator that yields strings representing the response.
        """
        msg = "chat_stream is not supported for synchronous execution"
        raise NotImplementedError(msg)


class AzureOpenAIEmbeddingFNLLM:
    """An Azure OpenAI Embedding Model provider using the fnllm library."""

    model: FNLLMEmbeddingLLM

    def __init__(
        self,
        *,
        name: str,
        config: LanguageModelConfig,
        callbacks: WorkflowCallbacks | None = None,
        cache: PipelineCache | None = None,
    ) -> None:
        model_config = _create_openai_config(config, azure=True)
        error_handler = _create_error_handler(callbacks) if callbacks else None
        model_cache = _create_cache(cache, name)
        client = create_openai_client(model_config)
        self.model = create_openai_embeddings_llm(
            model_config,
            client=client,
            cache=model_cache,
            events=FNLLMEvents(error_handler) if error_handler else None,
        )
        self.config = config

    async def aembed_batch(self, text_list: list[str], **kwargs) -> list[list[float]]:
        """
        Embed the given text using the Model.

        Args:
            text: The text to embed.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            The embeddings of the text.
        """
        response = await self.model(text_list, **kwargs)
        if response.output.embeddings is None:
            msg = "No embeddings found in response"
            raise ValueError(msg)
        embeddings: list[list[float]] = response.output.embeddings
        return embeddings

    async def aembed(self, text: str, **kwargs) -> list[float]:
        """
        Embed the given text using the Model.

        Args:
            text: The text to embed.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            The embeddings of the text.
        """
        response = await self.model([text], **kwargs)
        if response.output.embeddings is None:
            msg = "No embeddings found in response"
            raise ValueError(msg)
        embeddings: list[float] = response.output.embeddings[0]
        return embeddings

    def embed_batch(self, text_list: list[str], **kwargs) -> list[list[float]]:
        """
        Embed the given text using the Model.

        Args:
            text: The text to embed.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            The embeddings of the text.
        """
        return run_coroutine_sync(self.aembed_batch(text_list, **kwargs))

    def embed(self, text: str, **kwargs) -> list[float]:
        """
        Embed the given text using the Model.

        Args:
            text: The text to embed.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            The embeddings of the text.
        """
        return run_coroutine_sync(self.aembed(text, **kwargs))
