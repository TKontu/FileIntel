"""
OpenAI Provider - Backward compatibility wrapper.

Redirects to UnifiedLLMProvider with OpenAI configuration.
"""

import logging
from fileintel.llm_integration.unified_provider import (
    UnifiedLLMProvider,
    LLMProviderType,
)
from fileintel.core.config import get_config, Settings
from fileintel.storage.postgresql_storage import PostgreSQLStorage

logger = logging.getLogger(__name__)


class OpenAIProvider:
    """
    Backward compatibility wrapper for OpenAI provider.
    Redirects to UnifiedLLMProvider with OpenAI configuration.
    """

    def __init__(self, storage: PostgreSQLStorage = None, cache=None):
        # Create config that forces OpenAI provider
        config = get_config()

        # Ensure OpenAI provider is selected
        if config.llm.provider != "openai":
            logger.warning(
                f"Config specifies {config.llm.provider} but OpenAIProvider was requested. Using OpenAI."
            )
            # Create a modified config copy for OpenAI
            config_dict = config.model_dump()
            config_dict["llm"]["provider"] = "openai"
            config = Settings.model_validate(config_dict)

        self._unified_provider = UnifiedLLMProvider(config=config, storage=storage)
        logger.info("OpenAIProvider initialized (using UnifiedLLMProvider)")

    def generate_response(
        self,
        prompt: str,
        model: str = None,
        stream: bool = False,
        timeout: int = None,
        **kwargs,
    ):
        """
        Generate response using unified provider.
        Direct sync call - no longer needs async conversion.
        """
        return self._unified_provider.generate_response(prompt, model, **kwargs)

    def __getattr__(self, name):
        """Delegate any other method calls to the unified provider."""
        return getattr(self._unified_provider, name)
