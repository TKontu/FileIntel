# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""
Checkpoint detection and management for GraphRAG indexing pipeline.

This module provides checkpoint detection capabilities to enable automatic
resume from the last successful workflow step, avoiding the need to restart
multi-day indexing operations from scratch after failures.
"""

import logging
from typing import Any

import pandas as pd

from graphrag.storage.pipeline_storage import PipelineStorage
from graphrag.utils.storage import load_table_from_storage

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages checkpoint detection and workflow resume logic."""

    # Define expected outputs for each workflow
    # Maps workflow name -> list of parquet files it produces
    WORKFLOW_OUTPUTS = {
        "load_input_documents": ["documents.parquet"],
        "create_base_text_units": ["text_units.parquet"],
        "create_final_documents": ["documents.parquet"],  # overwrites
        "extract_graph": ["entities.parquet", "relationships.parquet"],
        "finalize_graph": ["entities.parquet", "relationships.parquet"],  # overwrites with additional columns
        "extract_covariates": ["covariates.parquet"],  # optional workflow
        "create_communities": ["communities.parquet"],
        "create_final_text_units": ["text_units.parquet"],  # overwrites with additional columns
        "create_community_reports": ["community_reports.parquet"],
        "generate_text_embeddings": [
            # Adds embedding columns to existing files
            "entities.parquet",
            "relationships.parquet",
            "text_units.parquet",
            "community_reports.parquet",
        ],
    }

    # Define required columns for validation
    # Maps workflow name -> {filename: [required_columns]}
    WORKFLOW_REQUIRED_COLUMNS = {
        "create_base_text_units": {
            "text_units.parquet": ["id", "text", "n_tokens", "document_ids"],
        },
        "extract_graph": {
            "entities.parquet": ["id", "title", "type", "description", "text_unit_ids"],
            "relationships.parquet": ["id", "source", "target", "text_unit_ids"],
        },
        "finalize_graph": {
            "entities.parquet": ["degree"],  # additional column added by finalize_graph
            "relationships.parquet": ["weight"],  # additional column added by finalize_graph
        },
        "create_communities": {
            "communities.parquet": [
                "id",
                "community",
                "level",
                "entity_ids",
                "relationship_ids",
            ],
        },
        "create_final_text_units": {
            "text_units.parquet": ["entity_ids"],  # additional column added
        },
        "create_community_reports": {
            "community_reports.parquet": [
                "id",
                "community",
                "level",
                "title",
                "summary",
                "full_content",
            ],
        },
        "generate_text_embeddings": {
            "entities.parquet": ["description_embedding"],
            "text_units.parquet": ["text_embedding"],
        },
    }

    # Minimum row count thresholds (helps detect incomplete processing)
    MIN_ROW_COUNTS = {
        "entities.parquet": 1,  # At least 1 entity
        "relationships.parquet": 0,  # Relationships are optional
        "communities.parquet": 1,  # At least 1 community
        "community_reports.parquet": 1,  # At least 1 report
        "text_units.parquet": 1,  # At least 1 text unit
        "documents.parquet": 1,  # At least 1 document
    }

    async def check_workflow_completion(
        self, workflow_name: str, storage: PipelineStorage
    ) -> dict[str, Any]:
        """
        Check if a workflow has been completed successfully.

        Args:
            workflow_name: Name of the workflow to check
            storage: Storage instance to check for output files

        Returns:
            Dictionary with completion status:
            {
                "completed": bool,      # All files exist and are valid
                "partial": bool,        # Some files exist but incomplete
                "missing_files": list,  # Files that don't exist
                "invalid_files": list,  # Files that exist but are invalid
                "row_counts": dict,     # Row counts for each file
            }
        """
        result: dict[str, Any] = {
            "completed": False,
            "partial": False,
            "missing_files": [],
            "invalid_files": [],
            "row_counts": {},
        }

        expected_files = self.WORKFLOW_OUTPUTS.get(workflow_name, [])
        if not expected_files:
            logger.warning(f"No expected outputs defined for workflow: {workflow_name}")
            return result

        required_columns = self.WORKFLOW_REQUIRED_COLUMNS.get(workflow_name, {})

        # Check each expected output file
        for filename in expected_files:
            table_name = filename.replace(".parquet", "")

            # Check file existence
            if not await storage.has(filename):
                result["missing_files"].append(filename)
                continue

            # Validate file structure and content
            try:
                df = await load_table_from_storage(table_name, storage)
                row_count = len(df)
                result["row_counts"][filename] = row_count

                # Check for empty results (likely incomplete)
                min_rows = self.MIN_ROW_COUNTS.get(filename, 0)
                if row_count < min_rows:
                    result["invalid_files"].append(
                        f"{filename} (only {row_count} rows, expected >={min_rows})"
                    )
                    continue

                # Check required columns exist
                if filename in required_columns:
                    missing_cols = set(required_columns[filename]) - set(df.columns)
                    if missing_cols:
                        result["invalid_files"].append(
                            f"{filename} (missing columns: {missing_cols})"
                        )
                        continue

            except Exception as e:
                result["invalid_files"].append(f"{filename} (error: {str(e)[:100]})")
                continue

        # Determine completion status
        has_missing = len(result["missing_files"]) > 0
        has_invalid = len(result["invalid_files"]) > 0

        if not has_missing and not has_invalid:
            # All files exist and are valid
            result["completed"] = True
        elif has_invalid or (has_missing and result["row_counts"]):
            # Some files exist but there are issues
            result["partial"] = True
        # else: no files exist, not started (completed=False, partial=False)

        return result

    async def find_resume_point(
        self, workflow_names: list[str], storage: PipelineStorage
    ) -> tuple[int, str | None]:
        """
        Find the last completed workflow and return where to resume.

        Args:
            workflow_names: List of workflow names in execution order
            storage: Storage instance to check for checkpoints

        Returns:
            Tuple of (resume_index, last_completed_workflow):
            - resume_index: Index of workflow to resume from (0-based)
            - last_completed_workflow: Name of last completed workflow (or None)
        """
        last_completed_idx = -1
        last_completed_name = None

        logger.info(
            f"🔍 Checking for existing checkpoints across {len(workflow_names)} workflows..."
        )

        for idx, name in enumerate(workflow_names):
            status = await self.check_workflow_completion(name, storage)

            if status["completed"]:
                last_completed_idx = idx
                last_completed_name = name
                row_summary = ", ".join(
                    f"{k}: {v}" for k, v in status["row_counts"].items()
                )
                logger.info(f"  ✓ Checkpoint found: {name} completed ({row_summary})")
            elif status["partial"]:
                logger.warning(
                    f"  ⚠ Partial checkpoint: {name} incomplete "
                    f"(missing: {status['missing_files']}, "
                    f"invalid: {status['invalid_files']})"
                )
                logger.warning(f"  → Will restart from this step to ensure consistency")
                break
            else:
                logger.info(f"  ○ Checkpoint not found: {name} not started")
                break

        resume_idx = last_completed_idx + 1

        if resume_idx == 0:
            logger.info("🆕 No checkpoints found - starting fresh pipeline execution")
        else:
            logger.info(
                f"📍 Resume point: workflow #{resume_idx} "
                f"(after '{last_completed_name}')"
            )

        return resume_idx, last_completed_name

    async def validate_checkpoint_chain(
        self, workflow_names: list[str], storage: PipelineStorage, resume_idx: int
    ) -> dict[str, Any]:
        """
        Validate that all checkpoints before resume point form a valid chain.

        Args:
            workflow_names: List of workflow names
            storage: Storage instance
            resume_idx: Index where resume will start

        Returns:
            Dictionary with validation results:
            {
                "valid": bool,
                "issues": list[str],
                "warnings": list[str]
            }
        """
        result = {
            "valid": True,
            "issues": [],
            "warnings": [],
        }

        # Only validate workflows before resume point
        workflows_to_check = workflow_names[:resume_idx]

        for workflow_name in workflows_to_check:
            status = await self.check_workflow_completion(workflow_name, storage)

            if not status["completed"]:
                result["valid"] = False
                result["issues"].append(
                    f"Workflow '{workflow_name}' marked as checkpoint but is incomplete"
                )

            if status["invalid_files"]:
                result["valid"] = False
                result["issues"].append(
                    f"Workflow '{workflow_name}' has invalid files: "
                    f"{status['invalid_files']}"
                )

        # Additional cross-workflow validation
        if resume_idx > 0:
            # Check text_units -> document_ids linkage
            try:
                if await storage.has("text_units.parquet"):
                    text_units = await load_table_from_storage("text_units", storage)
                    if "document_ids" not in text_units.columns:
                        result["valid"] = False
                        result["issues"].append(
                            "text_units.parquet missing document_ids column"
                        )
                    elif text_units["document_ids"].isna().all():
                        result["warnings"].append(
                            "text_units.parquet has all null document_ids"
                        )
            except Exception as e:
                result["warnings"].append(f"Could not validate text_units: {e}")

            # Check entities -> text_unit_ids linkage
            try:
                if await storage.has("entities.parquet"):
                    entities = await load_table_from_storage("entities", storage)
                    if "text_unit_ids" not in entities.columns:
                        result["valid"] = False
                        result["issues"].append(
                            "entities.parquet missing text_unit_ids column"
                        )
                    elif entities["text_unit_ids"].isna().all():
                        result["warnings"].append(
                            "entities.parquet has all null text_unit_ids"
                        )
            except Exception as e:
                result["warnings"].append(f"Could not validate entities: {e}")

        if result["valid"]:
            logger.info("✓ Checkpoint validation passed")
        else:
            logger.error(f"✗ Checkpoint validation failed: {result['issues']}")

        if result["warnings"]:
            for warning in result["warnings"]:
                logger.warning(f"  ⚠ {warning}")

        return result
