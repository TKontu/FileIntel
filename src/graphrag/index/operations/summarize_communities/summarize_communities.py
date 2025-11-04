# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing create_community_reports and load_strategy methods definition."""

import logging
from collections.abc import Callable

import pandas as pd

import graphrag.data_model.schemas as schemas
from graphrag.cache.pipeline_cache import PipelineCache
from graphrag.callbacks.noop_workflow_callbacks import NoopWorkflowCallbacks
from graphrag.callbacks.workflow_callbacks import WorkflowCallbacks
from graphrag.config.enums import AsyncType
from graphrag.index.operations.summarize_communities.typing import (
    CommunityReport,
    CommunityReportsStrategy,
    CreateCommunityReportsStrategyType,
)
from graphrag.index.operations.summarize_communities.utils import (
    get_levels,
)
from graphrag.index.utils.derive_from_rows import derive_from_rows
from graphrag.logger.progress import progress_ticker

logger = logging.getLogger(__name__)


async def summarize_communities(
    nodes: pd.DataFrame,
    communities: pd.DataFrame,
    local_contexts,
    level_context_builder: Callable,
    callbacks: WorkflowCallbacks,
    cache: PipelineCache,
    strategy: dict,
    max_input_length: int,
    async_mode: AsyncType = AsyncType.AsyncIO,
    num_threads: int = 4,
    existing_reports: pd.DataFrame | None = None,
):
    """Generate community summaries."""
    # Initialize reports list with existing reports if provided (for resume capability)
    reports: list[CommunityReport | None] = []
    existing_community_ids = set()

    if existing_reports is not None and not existing_reports.empty:
        # Validate existing reports have required columns
        required_cols = [schemas.COMMUNITY_ID, schemas.COMMUNITY_LEVEL]
        missing_cols = set(required_cols) - set(existing_reports.columns)
        if missing_cols:
            logger.warning(
                f"Existing reports missing required columns {missing_cols}, ignoring partial results for safety"
            )
            existing_reports = None
        else:
            # Convert existing reports DataFrame to list of CommunityReport objects
            for _, row in existing_reports.iterrows():
                # Ensure type consistency: convert community ID to int to avoid type mismatch
                community_id = int(row.get(schemas.COMMUNITY_ID))

                # Create base report (TypedDict fields only)
                report = CommunityReport(
                    community=community_id,
                    level=row.get(schemas.COMMUNITY_LEVEL),
                    title=row.get("title", ""),
                    summary=row.get("summary", ""),
                    full_content=row.get("full_content", ""),
                    full_content_json=row.get("full_content_json", ""),
                    rank=row.get("rank", 0.0),
                    rating_explanation=row.get("rating_explanation", ""),
                    findings=row.get("findings", []),
                )

                # Preserve ID if it exists (TypedDict allows extra fields at runtime)
                if "id" in row and pd.notna(row["id"]) and row["id"] != "":
                    report["id"] = row["id"]  # type: ignore

                reports.append(report)
                existing_community_ids.add(community_id)

            logger.info(f"Resume mode: Loaded {len(existing_community_ids)} existing community reports, will skip these during summarization")

    # Calculate total work for progress tracking (accounting for skipped communities)
    total_work = len(local_contexts) - len(existing_community_ids) if existing_community_ids else len(local_contexts)
    tick = progress_ticker(callbacks.progress, total_work)

    strategy_exec = load_strategy(strategy["type"])
    strategy_config = {**strategy}

    community_hierarchy = (
        communities.explode("children")
        .rename({"children": "sub_community"}, axis=1)
        .loc[:, ["community", "level", "sub_community"]]
    ).dropna()

    levels = get_levels(nodes)

    # CRITICAL FIX: Build level contexts incrementally to use newly-generated reports
    # This maintains the hierarchical design where parent levels use child summaries
    for i, level in enumerate(levels):
        # Rebuild context for this level with all reports generated so far
        level_context = level_context_builder(
            pd.DataFrame(reports),
            community_hierarchy_df=community_hierarchy,
            local_context_df=local_contexts,
            level=level,
            max_context_tokens=max_input_length,
        )

        # Filter out communities that already have reports (resume capability)
        if existing_community_ids:
            original_count = len(level_context)
            # Use .copy() to avoid SettingWithCopyWarning
            level_context = level_context[
                ~level_context[schemas.COMMUNITY_ID].isin(existing_community_ids)
            ].copy()
            filtered_count = len(level_context)
            skipped_count = original_count - filtered_count

            if skipped_count > 0:
                logger.info(
                    f"Resume mode: Skipping {skipped_count} already-completed communities at level {level}, "
                    f"processing {filtered_count} remaining communities"
                )

        # Skip this level if all communities already have reports
        if level_context.empty:
            logger.info(f"Resume mode: All communities at level {level} already have reports, skipping")
            continue

        async def run_generate(record):
            result = await _generate_report(
                strategy_exec,
                community_id=record[schemas.COMMUNITY_ID],
                community_level=record[schemas.COMMUNITY_LEVEL],
                community_context=record[schemas.CONTEXT_STRING],
                callbacks=callbacks,
                cache=cache,
                strategy=strategy_config,
            )
            tick()
            return result

        local_reports = await derive_from_rows(
            level_context,
            run_generate,
            callbacks=NoopWorkflowCallbacks(),
            num_threads=num_threads,
            async_type=async_mode,
            progress_msg=f"level {level} summarize communities progress: ",
        )
        reports.extend([lr for lr in local_reports if lr is not None])

    return pd.DataFrame(reports)


async def _generate_report(
    runner: CommunityReportsStrategy,
    callbacks: WorkflowCallbacks,
    cache: PipelineCache,
    strategy: dict,
    community_id: int,
    community_level: int,
    community_context: str,
) -> CommunityReport | None:
    """Generate a report for a single community."""
    return await runner(
        community_id,
        community_context,
        community_level,
        callbacks,
        cache,
        strategy,
    )


def load_strategy(
    strategy: CreateCommunityReportsStrategyType,
) -> CommunityReportsStrategy:
    """Load strategy method definition."""
    match strategy:
        case CreateCommunityReportsStrategyType.graph_intelligence:
            from graphrag.index.operations.summarize_communities.strategies import (
                run_graph_intelligence,
            )

            return run_graph_intelligence
        case _:
            msg = f"Unknown strategy: {strategy}"
            raise ValueError(msg)
