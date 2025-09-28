from typing import Generator, Any, Dict, NamedTuple, Union, Protocol, Optional
import hashlib
import json


class LLMResponse(NamedTuple):
    """
    A standardized response object from an LLM provider.
    """

    content: str
    model: str
    provider: str
    usage: Dict[str, Any]
    metadata: Dict[str, Any]


class LLMProvider(Protocol):
    """
    Protocol for LLM providers - defines required interface without inheritance.
    """

    def generate_response(
        self,
        prompt: str,
        model: str = None,
        stream: bool = False,
        timeout: int = None,
        **kwargs,
    ) -> Union[Generator[LLMResponse, None, None], LLMResponse]:
        """
        Sends a prompt to the LLM and returns the response.
        """
        ...


def get_cache_key(prompt: str, model: str, **kwargs) -> str:
    """Creates a unique cache key for a given request."""
    # Simplified cache key generation without unnecessary SHA256 hashing
    request_data = {
        "prompt": prompt[:100],  # Only use first 100 chars for key
        "model": model,
        "kwargs": sorted(kwargs.items()),
    }
    request_str = json.dumps(request_data)
    return f"llm_cache:{hash(request_str)}"


def validate_response(response: LLMResponse) -> LLMResponse:
    """
    Basic validation for LLM response - only check essential fields.

    Args:
        response: The LLMResponse object to validate.

    Returns:
        The same response if valid.

    Raises:
        ValueError: If the response is invalid.
    """
    from fileintel.core.validation import (
        validate_llm_response_content,
        validate_llm_response_model,
    )

    validate_llm_response_content(response.content)
    validate_llm_response_model(response.model)
    return response


async def handle_cached_response(cache, cache_key: str) -> Optional[LLMResponse]:
    """Check cache for existing response and return it if found."""
    cached_response = await cache.get_async(cache_key)
    if cached_response:
        response_data = json.loads(cached_response)
        return LLMResponse(**response_data)
    return None


async def cache_response(cache, cache_key: str, response: LLMResponse):
    """Cache the LLM response for future use."""
    await cache.set_async(cache_key, json.dumps(response._asdict()))
