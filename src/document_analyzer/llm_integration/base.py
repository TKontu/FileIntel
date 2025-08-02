from abc import ABC, abstractmethod

class LLMProvider(ABC):
    @abstractmethod
    def get_response(self, prompt: str) -> str:
        """
        Sends a prompt to the LLM and returns the response.
        """
        pass
