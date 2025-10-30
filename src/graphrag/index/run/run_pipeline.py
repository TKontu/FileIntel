# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Different methods to run the pipeline."""

import json
import logging
import re
import time
from collections.abc import AsyncIterable
from dataclasses import asdict
from typing import Any

import pandas as pd

from graphrag.callbacks.workflow_callbacks import WorkflowCallbacks
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.index.run.checkpoint_manager import CheckpointManager
from graphrag.index.run.utils import create_run_context
from graphrag.index.typing.context import PipelineRunContext
from graphrag.index.typing.pipeline import Pipeline
from graphrag.index.typing.pipeline_run_result import PipelineRunResult
from graphrag.storage.pipeline_storage import PipelineStorage
from graphrag.utils.api import create_cache_from_config, create_storage_from_config
from graphrag.utils.storage import load_table_from_storage, write_table_to_storage

logger = logging.getLogger(__name__)


async def run_pipeline(
    pipeline: Pipeline,
    config: GraphRagConfig,
    callbacks: WorkflowCallbacks,
    is_update_run: bool = False,
    additional_context: dict[str, Any] | None = None,
    input_documents: pd.DataFrame | None = None,
) -> AsyncIterable[PipelineRunResult]:
    """Run all workflows using a simplified pipeline."""
    root_dir = config.root_dir

    input_storage = create_storage_from_config(config.input.storage)
    output_storage = create_storage_from_config(config.output)
    cache = create_cache_from_config(config.cache, root_dir)

    # load existing state in case any workflows are stateful
    state_json = await output_storage.get("context.json")
    state = json.loads(state_json) if state_json else {}

    if additional_context:
        state.setdefault("additional_context", {}).update(additional_context)

    if is_update_run:
        logger.info("Running incremental indexing.")

        update_storage = create_storage_from_config(config.update_index_output)
        # we use this to store the new subset index, and will merge its content with the previous index
        update_timestamp = time.strftime("%Y%m%d-%H%M%S")
        timestamped_storage = update_storage.child(update_timestamp)
        delta_storage = timestamped_storage.child("delta")
        # copy the previous output to a backup folder, so we can replace it with the update
        # we'll read from this later when we merge the old and new indexes
        previous_storage = timestamped_storage.child("previous")
        await _copy_previous_output(output_storage, previous_storage)

        state["update_timestamp"] = update_timestamp

        # if the user passes in a df directly, write directly to storage so we can skip finding/parsing later
        if input_documents is not None:
            await write_table_to_storage(input_documents, "documents", delta_storage)
            pipeline.remove("load_update_documents")

        context = create_run_context(
            input_storage=input_storage,
            output_storage=delta_storage,
            previous_storage=previous_storage,
            cache=cache,
            callbacks=callbacks,
            state=state,
        )

    else:
        logger.info("Running standard indexing.")

        # if the user passes in a df directly, write directly to storage so we can skip finding/parsing later
        if input_documents is not None:
            await write_table_to_storage(input_documents, "documents", output_storage)
            pipeline.remove("load_input_documents")

        context = create_run_context(
            input_storage=input_storage,
            output_storage=output_storage,
            cache=cache,
            callbacks=callbacks,
            state=state,
        )

    async for table in _run_pipeline(
        pipeline=pipeline,
        config=config,
        context=context,
    ):
        yield table


async def run_pipeline_with_resume(
    pipeline: Pipeline,
    config: GraphRagConfig,
    callbacks: WorkflowCallbacks,
    is_update_run: bool = False,
    additional_context: dict[str, Any] | None = None,
    input_documents: pd.DataFrame | None = None,
    enable_resume: bool = True,
    validate_checkpoints: bool = True,
) -> AsyncIterable[PipelineRunResult]:
    """
    Run pipeline with checkpoint resume capability.

    This function wraps the standard run_pipeline with checkpoint detection,
    allowing automatic resume from the last completed workflow step.

    Args:
        pipeline: The pipeline to execute
        config: GraphRAG configuration
        callbacks: Workflow callbacks
        is_update_run: Whether this is an incremental update run
        additional_context: Additional context to pass to workflows
        input_documents: Optional pre-loaded documents
        enable_resume: Whether to enable checkpoint detection and resume
        validate_checkpoints: Whether to validate checkpoint consistency

    Yields:
        PipelineRunResult for each workflow
    """
    root_dir = config.root_dir

    input_storage = create_storage_from_config(config.input.storage)
    output_storage = create_storage_from_config(config.output)
    cache = create_cache_from_config(config.cache, root_dir)

    # load existing state in case any workflows are stateful
    state_json = await output_storage.get("context.json")
    state = json.loads(state_json) if state_json else {}

    if additional_context:
        state.setdefault("additional_context", {}).update(additional_context)

    if is_update_run:
        logger.info("Running incremental indexing.")

        update_storage = create_storage_from_config(config.update_index_output)
        update_timestamp = time.strftime("%Y%m%d-%H%M%S")
        timestamped_storage = update_storage.child(update_timestamp)
        delta_storage = timestamped_storage.child("delta")
        previous_storage = timestamped_storage.child("previous")
        await _copy_previous_output(output_storage, previous_storage)

        state["update_timestamp"] = update_timestamp

        if input_documents is not None:
            await write_table_to_storage(input_documents, "documents", delta_storage)
            pipeline.remove("load_update_documents")

        context = create_run_context(
            input_storage=input_storage,
            output_storage=delta_storage,
            previous_storage=previous_storage,
            cache=cache,
            callbacks=callbacks,
            state=state,
        )

    else:
        logger.info("Running standard indexing.")

        if input_documents is not None:
            await write_table_to_storage(input_documents, "documents", output_storage)
            pipeline.remove("load_input_documents")

        context = create_run_context(
            input_storage=input_storage,
            output_storage=output_storage,
            cache=cache,
            callbacks=callbacks,
            state=state,
        )

    # NEW: Find resume point if enabled
    resume_idx = 0
    last_completed = None

    if enable_resume:
        logger.info("🔍 Checking for existing checkpoints...")
        checkpoint_mgr = CheckpointManager()

        # Get workflow names from pipeline
        workflow_names = [name for name, _ in pipeline.workflows]

        resume_idx, last_completed = await checkpoint_mgr.find_resume_point(
            workflow_names, context.output_storage
        )

        if resume_idx > 0:
            logger.info(
                f"📍 Resuming from workflow #{resume_idx} (after '{last_completed}')"
            )

            # Validate checkpoints if requested
            if validate_checkpoints:
                logger.info("🔍 Validating checkpoint consistency...")
                validation = await checkpoint_mgr.validate_checkpoint_chain(
                    workflow_names, context.output_storage, resume_idx
                )

                if not validation["valid"]:
                    logger.error(
                        f"❌ Checkpoint validation failed: {validation['issues']}"
                    )
                    logger.error(
                        "   Falling back to full rebuild for data consistency"
                    )
                    resume_idx = 0
                else:
                    logger.info("✓ Checkpoint validation passed")
        else:
            logger.info("🆕 Starting fresh - no checkpoints found")
    else:
        logger.info("ℹ️ Checkpoint resume disabled - starting from beginning")

    # Store resume metadata in state
    state["resume_enabled"] = enable_resume
    state["resume_from_index"] = resume_idx
    state["last_completed_workflow"] = last_completed

    # Execute pipeline starting from resume point
    async for table in _run_pipeline_from_index(
        pipeline=pipeline,
        config=config,
        context=context,
        start_index=resume_idx,
    ):
        yield table


async def _run_pipeline_from_index(
    pipeline: Pipeline,
    config: GraphRagConfig,
    context: PipelineRunContext,
    start_index: int = 0,
) -> AsyncIterable[PipelineRunResult]:
    """
    Execute pipeline starting from a specific workflow index.

    This is a refactored version of _run_pipeline that supports skipping
    completed workflows when resuming from a checkpoint.

    Args:
        pipeline: The pipeline to execute
        config: GraphRAG configuration
        context: Pipeline run context
        start_index: Workflow index to start from (0-based)

    Yields:
        PipelineRunResult for each workflow
    """
    start_time = time.time()
    last_workflow = "<startup>"

    try:
        await _dump_json(context)

        if start_index > 0:
            logger.info(
                f"Executing pipeline from workflow #{start_index} "
                f"(skipping {start_index} completed workflows)..."
            )
        else:
            logger.info("Executing pipeline...")

        workflows = list(pipeline.run())

        for idx, (name, workflow_function) in enumerate(workflows):
            # Skip completed workflows
            if idx < start_index:
                logger.info(f"⏭ Skipping workflow #{idx}: {name} (checkpoint exists)")
                # Still yield result to maintain API contract
                yield PipelineRunResult(
                    workflow=name,
                    result={"skipped": True, "reason": "checkpoint exists"},
                    state=context.state,
                    errors=None,
                )
                continue

            last_workflow = name
            logger.info(f"▶ Executing workflow #{idx}: {name}")

            context.callbacks.workflow_start(name, None)
            work_time = time.time()

            result = await workflow_function(config, context)

            context.callbacks.workflow_end(name, result)
            elapsed = time.time() - work_time

            logger.info(f"✓ Completed workflow: {name} ({elapsed:.1f}s)")

            yield PipelineRunResult(
                workflow=name, result=result.result, state=context.state, errors=None
            )
            context.stats.workflows[name] = {"overall": elapsed}

            if result.stop:
                logger.info("Halting pipeline at workflow request")
                break

        context.stats.total_runtime = time.time() - start_time
        logger.info("Indexing pipeline complete.")
        await _dump_json(context)

    except Exception as e:
        logger.exception("error running workflow %s", last_workflow)
        yield PipelineRunResult(
            workflow=last_workflow, result=None, state=context.state, errors=[e]
        )


async def _run_pipeline(
    pipeline: Pipeline,
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> AsyncIterable[PipelineRunResult]:
    start_time = time.time()

    last_workflow = "<startup>"

    try:
        await _dump_json(context)

        logger.info("Executing pipeline...")
        for name, workflow_function in pipeline.run():
            last_workflow = name
            context.callbacks.workflow_start(name, None)
            work_time = time.time()
            result = await workflow_function(config, context)
            context.callbacks.workflow_end(name, result)
            yield PipelineRunResult(
                workflow=name, result=result.result, state=context.state, errors=None
            )
            context.stats.workflows[name] = {"overall": time.time() - work_time}
            if result.stop:
                logger.info("Halting pipeline at workflow request")
                break

        context.stats.total_runtime = time.time() - start_time
        logger.info("Indexing pipeline complete.")
        await _dump_json(context)

    except Exception as e:
        logger.exception("error running workflow %s", last_workflow)
        yield PipelineRunResult(
            workflow=last_workflow, result=None, state=context.state, errors=[e]
        )


async def _dump_json(context: PipelineRunContext) -> None:
    """Dump the stats and context state to the storage."""
    await context.output_storage.set(
        "stats.json", json.dumps(asdict(context.stats), indent=4, ensure_ascii=False)
    )
    # Dump context state, excluding additional_context
    temp_context = context.state.pop(
        "additional_context", None
    )  # Remove reference only, as object size is uncertain
    try:
        state_blob = json.dumps(context.state, indent=4, ensure_ascii=False)
    finally:
        if temp_context:
            context.state["additional_context"] = temp_context

    await context.output_storage.set("context.json", state_blob)


async def _copy_previous_output(
    storage: PipelineStorage,
    copy_storage: PipelineStorage,
):
    for file in storage.find(re.compile(r"\.parquet$")):
        base_name = file[0].replace(".parquet", "")
        table = await load_table_from_storage(base_name, storage)
        await write_table_to_storage(table, base_name, copy_storage)
