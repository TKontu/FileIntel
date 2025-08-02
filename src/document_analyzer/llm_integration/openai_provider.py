from .base import LLMProvider
from ..core.config import settings
import openai

class OpenAIProvider(LLMProvider):
    def __init__(self):
        self.client = openai.OpenAI(
            base_url=settings.llm.base_url,
            api_key=settings.llm.api_key,
        )

    def get_response(self, prompt: str) -> str:
        """
        Sends a prompt to a compatible OpenAI API endpoint and returns the response.
        """
        response = self.client.chat.completions.create(
            model=settings.llm.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=settings.llm.max_tokens,
            temperature=settings.llm.temperature,
        )
        return response.choices[0].message.content
