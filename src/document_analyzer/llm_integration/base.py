from abc import ABC, abstractmethod
from typing import Generator, Any, Dict, NamedTuple, Union
import hashlib
import json
from ..core.exceptions import LLMException
from ..storage.cache import CacheInterface


class LLMResponse(NamedTuple):
    """
    A standardized response object from an LLM provider.
    """

    content: str
    model: str
    cost: float
    metadata: Dict[str, Any]


class LLMProvider(ABC):
    """
    Abstract base class for all LLM providers.
    """

    def __init__(self, cache: CacheInterface = None):
        self.cache = cache

    def _get_cache_key(self, prompt: str, model: str, **kwargs) -> str:
        """Creates a unique cache key for a given request."""
        # Create a stable string representation of the request
        request_data = {
            "prompt": prompt,
            "model": model,
            "kwargs": sorted(kwargs.items()),
        }
        request_str = json.dumps(request_data)

        # Hash the string to create a key
        return f"llm_cache:{hashlib.sha256(request_str.encode()).hexdigest()}"

    @abstractmethod
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
        This method will be decorated by subclasses to handle caching.
        """
        pass

    def validate_response(self, response: LLMResponse):
        """
        Validates the response from the LLM.

        Args:
            response: The LLMResponse object to validate.

        Raises:
            LLMException: If the response is invalid.
        """
        if not response.content or not response.content.strip():
            raise LLMException("LLM response content is empty.")
        if not response.model:
            raise LLMException("LLM response is missing model information.")
