from abc import ABC, abstractmethod
from typing import List
from openai import OpenAI
from ..core.config import settings


class EmbeddingProvider(ABC):
    @abstractmethod
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Converts a list of texts into a list of embeddings."""
        pass


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self):
        self.client = OpenAI(
            base_url=settings.llm.openai.base_url,
            api_key=settings.llm.openai.api_key,
        )
        self.model = settings.rag.embedding_model

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Gets embeddings from OpenAI."""
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
        )
        return [embedding.embedding for embedding in response.data]
