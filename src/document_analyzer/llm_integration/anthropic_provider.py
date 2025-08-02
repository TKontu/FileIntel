from .base import LLMProvider
from ..core.config import settings
import anthropic

class AnthropicProvider(LLMProvider):
    def __init__(self):
        self.client = anthropic.Anthropic()

    def get_response(self, prompt: str) -> str:
        """
        Sends a prompt to the Anthropic API and returns the response.
        """
        response = self.client.completions.create(
            model=settings.llm.model,
            prompt=f"\n\nHuman: {prompt}\n\nAssistant:",
            max_tokens_to_sample=settings.llm.max_tokens,
            temperature=settings.llm.temperature,
        )
        return response.completion
