import anthropic
import json
from typing import Generator, Union
from .base import LLMProvider, LLMResponse
from ..core.config import settings
from .rate_limiter import RateLimiter
from ..storage.cache import CacheInterface, RedisCache

anthropic_rate_limiter = RateLimiter(
    max_requests=settings.get("llm.anthropic.rate_limit", 15), per_seconds=60
)


class AnthropicProvider(LLMProvider):
    """
    An LLM provider for Anthropic models.
    """

    def __init__(self, cache: CacheInterface = None):
        super().__init__(cache or RedisCache())
        self.client = anthropic.Anthropic(api_key=settings.get("llm.anthropic_api_key"))

    @anthropic_rate_limiter
    def generate_response(
        self,
        prompt: str,
        model: str = None,
        stream: bool = False,
        timeout: int = None,
        **kwargs,
    ) -> Union[Generator[LLMResponse, None, None], LLMResponse]:
        """
        Sends a prompt to the Anthropic API and returns the response.
        """
        model = model or settings.get("llm.model", "claude-3-opus-20240229")
        timeout = timeout or settings.get("llm.timeout", 60)

        if stream:
            # Caching is not supported for streaming responses
            return self._generate_stream_response(
                prompt, model, timeout=timeout, **kwargs
            )

        # Check cache first
        cache_key = self._get_cache_key(prompt, model, timeout=timeout, **kwargs)
        cached_response = self.cache.get(cache_key)
        if cached_response:
            print("Returning cached response.")
            response_data = json.loads(cached_response)
            return LLMResponse(**response_data)

        # If not in cache, generate response
        response = self._generate_standard_response(
            prompt, model, timeout=timeout, **kwargs
        )

        # Save to cache
        self.cache.set(cache_key, json.dumps(response._asdict()))

        return response

    def _generate_standard_response(
        self, prompt: str, model: str, timeout: int, **kwargs
    ) -> LLMResponse:
        response = self.client.messages.create(
            model=model,
            max_tokens=4096,  # Anthropic requires max_tokens
            messages=[{"role": "user", "content": prompt}],
            timeout=timeout,
            **kwargs,
        )

        content = response.content[0].text
        cost = self._calculate_cost(response.usage)

        llm_response = LLMResponse(
            content=content,
            model=response.model,
            cost=cost,
            metadata={
                "usage": response.usage.dict(),
                "stop_reason": response.stop_reason,
            },
        )
        self.validate_response(llm_response)
        return llm_response

    def _generate_stream_response(
        self, prompt: str, model: str, timeout: int, **kwargs
    ) -> Generator[LLMResponse, None, None]:
        with self.client.messages.stream(
            model=model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
            timeout=timeout,
            **kwargs,
        ) as stream:
            for chunk in stream.text_stream:
                yield LLMResponse(
                    content=chunk,
                    model=model,  # The model is not in the stream chunks
                    cost=0,  # Cost calculation for streams is more complex
                    metadata={},
                )

    def _calculate_cost(self, usage) -> float:
        # This is a simplified cost calculation. A real implementation would
        # have a more sophisticated model with up-to-date pricing.
        prompt_cost = (usage.input_tokens / 1000) * 0.005
        completion_cost = (usage.output_tokens / 1000) * 0.025
        return prompt_cost + completion_cost
