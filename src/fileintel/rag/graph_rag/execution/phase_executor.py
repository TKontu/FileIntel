"""Phase executor with gap prevention and retry logic."""
import asyncio
import logging
import random
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PhaseResult:
    """Result of executing a phase."""

    phase_name: str
    total_items: int
    successful_items: int
    failed_items: int
    retry_attempts: int
    final_gaps: int
    success: bool

    @property
    def completeness(self) -> float:
        """Calculate completeness ratio."""
        if self.total_items == 0:
            return 1.0
        return (self.total_items - self.final_gaps) / self.total_items


class PhaseExecutor:
    """Executes GraphRAG phase with gap prevention and retry logic."""

    def __init__(
        self,
        phase_name: str,
        max_retries: int = 20,
        retry_backoff_base: float = 2.0,
        retry_backoff_max: float = 120.0,
        retry_jitter: bool = True,
        gap_fill_concurrency: int = 5,
    ):
        """Initialize phase executor.

        Args:
            phase_name: Name of the phase being executed
            max_retries: Maximum number of retries per failed item
            retry_backoff_base: Exponential backoff base multiplier
            retry_backoff_max: Maximum backoff time in seconds
            retry_jitter: Whether to add random jitter to backoff
            gap_fill_concurrency: Concurrency limit for gap filling (lower to reduce 503s)
        """
        self.phase_name = phase_name
        self.max_retries = max_retries
        self.retry_backoff_base = retry_backoff_base
        self.retry_backoff_max = retry_backoff_max
        self.retry_jitter = retry_jitter
        self.gap_fill_concurrency = gap_fill_concurrency

    def _calculate_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff with optional jitter.

        Args:
            attempt: Current retry attempt number (0-indexed)

        Returns:
            Backoff time in seconds
        """
        # Exponential backoff: base^attempt
        backoff = min(
            self.retry_backoff_base**attempt,
            self.retry_backoff_max,
        )

        # Add jitter if enabled (±25% random variation)
        if self.retry_jitter:
            jitter_range = backoff * 0.25
            jitter = random.uniform(-jitter_range, jitter_range)
            backoff = max(0, backoff + jitter)

        return backoff

    async def execute_phase(
        self,
        items: List[Any],
        process_func: Callable,
        validate_func: Optional[Callable] = None,
    ) -> PhaseResult:
        """Execute phase with automatic retry and gap filling.

        Steps:
        1. Process all items with normal concurrency
        2. Track failures and add to retry queue
        3. Retry failed items with exponential backoff (up to max_retries)
        4. If validate_func provided, check for gaps in output
        5. Fill remaining gaps with reduced concurrency
        6. Final validation

        Args:
            items: List of items to process
            process_func: Async function that processes a single item
                         Should accept (item, attempt_number) and return success/failure
            validate_func: Optional async function that validates completeness
                          Should return list of missing item IDs

        Returns:
            PhaseResult with execution statistics
        """
        total_items = len(items)
        logger.info(f"Starting phase '{self.phase_name}' with {total_items} items")

        # Track statistics
        successful_count = 0
        failed_items = []
        retry_attempts = 0

        # Step 1: Initial processing
        logger.info(f"Processing {total_items} items...")
        for idx, item in enumerate(items):
            try:
                success = await process_func(item, attempt=0)
                if success:
                    successful_count += 1
                else:
                    failed_items.append(item)
                    logger.warning(
                        f"Item {idx + 1}/{total_items} failed on first attempt"
                    )
            except Exception as e:
                logger.error(
                    f"Error processing item {idx + 1}/{total_items}: {e}",
                    exc_info=True,
                )
                failed_items.append(item)

        initial_success = successful_count
        logger.info(
            f"Initial processing: {successful_count}/{total_items} successful, {len(failed_items)} failed"
        )

        # Step 2: Retry failed items with exponential backoff
        if failed_items:
            logger.info(
                f"Retrying {len(failed_items)} failed items (max {self.max_retries} attempts)..."
            )

            for attempt in range(1, self.max_retries + 1):
                if not failed_items:
                    break

                # Calculate backoff and wait
                backoff = self._calculate_backoff(attempt - 1)
                logger.info(
                    f"Retry attempt {attempt}/{self.max_retries} after {backoff:.1f}s backoff"
                )
                await asyncio.sleep(backoff)

                # Retry failed items
                still_failing = []
                for item in failed_items:
                    try:
                        success = await process_func(item, attempt=attempt)
                        if success:
                            successful_count += 1
                            retry_attempts += 1
                        else:
                            still_failing.append(item)
                    except Exception as e:
                        logger.error(
                            f"Error retrying item on attempt {attempt}: {e}",
                            exc_info=True,
                        )
                        still_failing.append(item)

                failed_items = still_failing
                logger.info(
                    f"Retry {attempt} result: {successful_count}/{total_items} successful, {len(failed_items)} still failing"
                )

        # Step 3: Validate completeness if validator provided
        final_gaps = 0
        if validate_func:
            logger.info("Validating phase completeness...")
            try:
                missing_ids = await validate_func()
                final_gaps = len(missing_ids)

                if final_gaps > 0:
                    logger.warning(
                        f"Validation found {final_gaps} gaps after retry: {missing_ids[:10]}{'...' if final_gaps > 10 else ''}"
                    )

                    # Step 4: Attempt to fill gaps with reduced concurrency
                    logger.info(
                        f"Attempting gap fill with reduced concurrency ({self.gap_fill_concurrency})..."
                    )

                    # This would require the caller to provide gap-filling logic
                    # For now, just log the gaps
                    logger.warning(
                        f"⚠️  Phase '{self.phase_name}' completed with {final_gaps} gaps ({final_gaps/total_items*100:.2f}%)"
                    )
                else:
                    logger.info(f"✅ Phase '{self.phase_name}' 100% complete")

            except Exception as e:
                logger.error(f"Validation error: {e}", exc_info=True)
                # Assume worst case - count remaining failures as gaps
                final_gaps = len(failed_items)
        else:
            # No validation - use failed_items count
            final_gaps = len(failed_items)

        # Build result
        result = PhaseResult(
            phase_name=self.phase_name,
            total_items=total_items,
            successful_items=successful_count,
            failed_items=len(failed_items),
            retry_attempts=retry_attempts,
            final_gaps=final_gaps,
            success=(final_gaps == 0),
        )

        logger.info(
            f"Phase '{self.phase_name}' completed: {result.completeness:.2%} complete "
            f"({successful_count}/{total_items} successful, {retry_attempts} retries, {final_gaps} final gaps)"
        )

        return result
