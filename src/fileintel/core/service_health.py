"""Service health checking and startup coordination."""

import asyncio
import logging
from typing import Optional
from dataclasses import dataclass

import httpx


logger = logging.getLogger(__name__)


@dataclass
class ServiceHealthConfig:
    """Configuration for service health checks."""

    timeout: float = 5.0  # Single check timeout
    max_retries: int = 12  # Total attempts
    retry_delay: float = 5.0  # Seconds between retries
    startup_message: str = "Service starting up..."


class ServiceHealthChecker:
    """Check and wait for external services to become ready.

    Used before expensive operations (indexing, queries) to ensure
    services are responsive and avoid timeouts during processing.
    """

    def __init__(self, config: Optional[ServiceHealthConfig] = None):
        self.config = config or ServiceHealthConfig()

    async def wait_for_service(
        self,
        url: str,
        service_name: str,
        expected_status: int = 200,
    ) -> bool:
        """Wait for service to become healthy with retries.

        Args:
            url: Health check endpoint URL (e.g., http://host:port/health)
            service_name: Human-readable name for logging
            expected_status: Expected HTTP status code

        Returns:
            True if service is healthy, False if timed out
        """
        logger.info(f"Checking {service_name} health at {url}")

        async with httpx.AsyncClient() as client:
            for attempt in range(1, self.config.max_retries + 1):
                try:
                    response = await client.get(
                        url,
                        timeout=self.config.timeout,
                    )

                    if response.status_code == expected_status:
                        logger.info(f"✓ {service_name} is healthy (attempt {attempt})")
                        return True

                    logger.warning(
                        f"{service_name} returned status {response.status_code} "
                        f"(expected {expected_status}), retrying..."
                    )

                except (httpx.ConnectError, httpx.TimeoutException) as e:
                    logger.warning(
                        f"{service_name} not ready (attempt {attempt}/{self.config.max_retries}): {e}"
                    )

                except Exception as e:
                    logger.error(f"Unexpected error checking {service_name}: {e}")

                # Wait before retry (unless last attempt)
                if attempt < self.config.max_retries:
                    logger.info(
                        f"{self.config.startup_message} "
                        f"Waiting {self.config.retry_delay}s before retry..."
                    )
                    await asyncio.sleep(self.config.retry_delay)

        logger.error(
            f"✗ {service_name} did not become healthy after "
            f"{self.config.max_retries} attempts ({self.config.max_retries * self.config.retry_delay}s)"
        )
        return False

    async def check_llm_service(self, base_url: str) -> bool:
        """Check LLM service (vLLM) health.

        Args:
            base_url: Base URL like http://host:port/v1

        Returns:
            True if healthy, False otherwise
        """
        # Try health endpoint first, fallback to models endpoint
        health_url = base_url.rstrip('/v1').rstrip('/') + '/health'

        if await self.wait_for_service(health_url, "LLM service", expected_status=200):
            return True

        # Some vLLM deployments don't have /health, try /v1/models
        models_url = base_url.rstrip('/') + '/models'
        logger.info("Health endpoint unavailable, trying models endpoint...")
        return await self.wait_for_service(models_url, "LLM service (via models)", expected_status=200)

    async def check_embedding_service(self, base_url: str) -> bool:
        """Check embedding service health.

        Args:
            base_url: Base URL like http://host:port/v1

        Returns:
            True if healthy, False otherwise
        """
        health_url = base_url.rstrip('/v1').rstrip('/') + '/health'

        if await self.wait_for_service(health_url, "Embedding service", expected_status=200):
            return True

        # Fallback to models endpoint
        models_url = base_url.rstrip('/') + '/models'
        logger.info("Health endpoint unavailable, trying models endpoint...")
        return await self.wait_for_service(models_url, "Embedding service (via models)", expected_status=200)


async def ensure_services_ready(
    llm_base_url: Optional[str] = None,
    embedding_base_url: Optional[str] = None,
    config: Optional[ServiceHealthConfig] = None,
) -> bool:
    """Convenience function to check multiple services.

    Args:
        llm_base_url: LLM service base URL (if needed)
        embedding_base_url: Embedding service base URL (if needed)
        config: Health check configuration

    Returns:
        True if all required services are healthy
    """
    checker = ServiceHealthChecker(config)

    checks = []
    if llm_base_url:
        checks.append(checker.check_llm_service(llm_base_url))
    if embedding_base_url:
        checks.append(checker.check_embedding_service(embedding_base_url))

    if not checks:
        logger.warning("No services to check")
        return True

    results = await asyncio.gather(*checks)
    return all(results)
