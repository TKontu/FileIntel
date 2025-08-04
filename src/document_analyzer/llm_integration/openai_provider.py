import openai
import json
import logging
from typing import Generator, Union
from .base import LLMProvider, LLMResponse
from ..core.config import settings
from .rate_limiter import RateLimiter
from ..storage.cache import CacheInterface, RedisCache

logger = logging.getLogger(__name__)

openai_rate_limiter = RateLimiter(
    max_requests=settings.get("llm.openai.rate_limit", 30),
    per_seconds=60
)

class OpenAIProvider(LLMProvider):
    """
    An LLM provider for OpenAI models.
    """

    def __init__(self, cache: CacheInterface = None):
        super().__init__(cache or RedisCache())
        
        api_key = settings.get("llm.openai.api_key", "dummy-key")
        base_url = settings.get("llm.openai.base_url")
        
        logger.info(f"Initializing OpenAIProvider with base_url: {base_url}")
        
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    @openai_rate_limiter
    def generate_response(
        self,
        prompt: str,
        model: str = None,
        stream: bool = False,
        timeout: int = None,
        **kwargs,
    ) -> Union[Generator[LLMResponse, None, None], LLMResponse]:
        """
        Sends a prompt to the OpenAI API and returns the response.
        """
        model = model or settings.get("llm.model", "gpt-3.5-turbo")
        timeout = timeout or settings.get("output.timeout", 60)

        if stream:
            # Caching is not supported for streaming responses
            return self._generate_stream_response(prompt, model, timeout=timeout, **kwargs)

        # Check cache first
        cache_key = self._get_cache_key(prompt, model, timeout=timeout, **kwargs)
        cached_response = self.cache.get(cache_key)
        if cached_response:
            print("Returning cached response.")
            response_data = json.loads(cached_response)
            return LLMResponse(**response_data)

        # If not in cache, generate response
        response = self._generate_standard_response(prompt, model, timeout=timeout, **kwargs)

        # Save to cache
        self.cache.set(cache_key, json.dumps(response._asdict()))

        return response

    def _generate_standard_response(self, prompt: str, model: str, timeout: int, **kwargs) -> LLMResponse:
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            timeout=timeout,
            **kwargs,
        )
        
        content = response.choices[0].message.content
        cost = self._calculate_cost(response.usage)
        
        llm_response = LLMResponse(
            content=content,
            model=response.model,
            cost=cost,
            metadata={"usage": response.usage.dict()}
        )
        self.validate_response(llm_response)
        return llm_response

    def _generate_stream_response(self, prompt: str, model: str, timeout: int, **kwargs) -> Generator[LLMResponse, None, None]:
        stream = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            timeout=timeout,
            **kwargs,
        )
        
        for chunk in stream:
            content = chunk.choices[0].delta.content or ""
            yield LLMResponse(
                content=content,
                model=chunk.model,
                cost=0,  # Cost calculation for streams is more complex
                metadata={"finish_reason": chunk.choices[0].finish_reason}
            )

    def _calculate_cost(self, usage) -> float:
        # This is a simplified cost calculation. A real implementation would
        # have a more sophisticated model with up-to-date pricing.
        prompt_cost = (usage.prompt_tokens / 1000) * 0.0015  # Example cost for GPT-3.5 Turbo
        completion_cost = (usage.completion_tokens / 1000) * 0.002
        return prompt_cost + completion_cost