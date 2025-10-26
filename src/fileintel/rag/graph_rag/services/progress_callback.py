"""
GraphRAG progress tracking callback.

Provides clean progress logging for GraphRAG indexing workflows without
verbose per-LLM-call logging from fnllm.
"""

import logging
from typing import List, Optional
from graphrag.callbacks.workflow_callbacks import WorkflowCallbacks
from graphrag.index.typing.pipeline_run_result import PipelineRunResult
from graphrag.logger.progress import Progress

logger = logging.getLogger(__name__)


class GraphRAGProgressCallback(WorkflowCallbacks):
    """
    Custom callback for tracking GraphRAG indexing progress.

    Logs workflow-level progress instead of per-LLM-call details:
    - "GraphRAG: Entity extraction (2/8 workflows, 25%)"
    - "GraphRAG: Community detection (5/8 workflows, 62%)"
    - "GraphRAG: Completed 8/8 workflows (100%)"
    """

    def __init__(self, collection_id: str):
        """
        Initialize progress callback.

        Args:
            collection_id: Collection identifier for logging context
        """
        self.collection_id = collection_id
        self.workflow_names: List[str] = []
        self.completed_workflows: List[str] = []
        self.current_workflow: Optional[str] = None
        self.total_workflows = 0

    def pipeline_start(self, names: List[str]) -> None:
        """
        Called at pipeline start with list of all workflow names.

        Args:
            names: List of workflow names to be executed
        """
        try:
            self.workflow_names = names
            self.total_workflows = len(names)
            self.completed_workflows = []

            logger.info(
                f"GraphRAG indexing started: {self.total_workflows} workflows "
                f"for collection {self.collection_id}"
            )
            logger.debug(f"GraphRAG workflows: {', '.join(names)}")
        except Exception as e:
            logger.error(f"GraphRAG callback error in pipeline_start: {e}", exc_info=True)
            # Don't break pipeline - continue execution

    def workflow_start(self, name: str, instance: object) -> None:
        """
        Called when a workflow starts.

        Args:
            name: Workflow name
            instance: Workflow instance
        """
        try:
            self.current_workflow = name

            # Clean up workflow name for display
            display_name = self._format_workflow_name(name)
            # Current workflow index (completed + 1 for the one starting now)
            current_index = len(self.completed_workflows) + 1
            percentage = (current_index / self.total_workflows) * 100 if self.total_workflows > 0 else 0

            logger.info(
                f"GraphRAG: {display_name} ({current_index}/{self.total_workflows} workflows, {percentage:.0f}%)"
            )
        except Exception as e:
            logger.error(f"GraphRAG callback error in workflow_start: {e}", exc_info=True)
            # Don't break pipeline - continue execution

    def workflow_end(self, name: str, instance: object) -> None:
        """
        Called when a workflow completes.

        Args:
            name: Workflow name
            instance: Workflow instance
        """
        try:
            if name not in self.completed_workflows:
                self.completed_workflows.append(name)

            self.current_workflow = None
        except Exception as e:
            logger.error(f"GraphRAG callback error in workflow_end: {e}", exc_info=True)
            # Don't break pipeline - continue execution

    def progress(self, progress: Progress) -> None:
        """
        Called during workflow progress updates.

        We suppress these to avoid verbose logging - workflow start/end is sufficient.

        Args:
            progress: Progress information
        """
        # Intentionally suppress per-operation progress to keep logs clean
        # workflow_start/workflow_end provides enough visibility
        pass

    def pipeline_end(self, results: List[PipelineRunResult]) -> None:
        """
        Called when entire pipeline completes.

        Args:
            results: Pipeline execution results
        """
        try:
            logger.info(
                f"GraphRAG: Completed {len(self.completed_workflows)}/{self.total_workflows} workflows (100%)"
            )

            # Log high-level results summary
            success_count = sum(1 for r in results if r.errors is None or len(r.errors) == 0)
            if success_count < len(results):
                logger.warning(
                    f"GraphRAG: {len(results) - success_count}/{len(results)} workflows had errors"
                )
        except Exception as e:
            logger.error(f"GraphRAG callback error in pipeline_end: {e}", exc_info=True)
            # Don't break pipeline - continue execution

    def _format_workflow_name(self, name: str) -> str:
        """
        Format workflow name for user-friendly display.

        Args:
            name: Raw workflow name (e.g., "create_final_entities")

        Returns:
            Formatted name (e.g., "Entity extraction")
        """
        # Map common workflow names to readable descriptions
        name_map = {
            "create_base_text_units": "Text unit creation",
            "create_base_extracted_entities": "Entity extraction",
            "create_base_entity_graph": "Entity graph building",
            "create_final_entities": "Entity refinement",
            "create_final_nodes": "Graph node creation",
            "create_final_communities": "Community detection",
            "create_final_community_reports": "Community summarization",
            "create_base_documents": "Document processing",
            "create_final_relationships": "Relationship extraction",
            "create_final_text_units": "Text unit finalization",
        }

        return name_map.get(name, name.replace("_", " ").title())
