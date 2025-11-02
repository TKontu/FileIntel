"""Validates completeness of GraphRAG indices."""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CompletenessReport:
    """Completeness report for a single phase."""

    phase: str
    total_items: int
    complete_items: int
    missing_items: int
    completeness: float
    missing_ids: List[Any]
    details_by_level: Optional[Dict[int, Dict[str, int]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "phase": self.phase,
            "total_items": self.total_items,
            "complete_items": self.complete_items,
            "missing_items": self.missing_items,
            "completeness": self.completeness,
            "missing_ids": self.missing_ids,
        }
        if self.details_by_level is not None:
            result["details_by_level"] = self.details_by_level
        return result


class CompletenessValidator:
    """Validates completeness of GraphRAG indices."""

    def __init__(self, workspace_path: Path):
        """Initialize validator.

        Args:
            workspace_path: Path to GraphRAG workspace directory
        """
        self.workspace_path = Path(workspace_path)

    def validate_entity_descriptions(self) -> CompletenessReport:
        """Validate entity descriptions are complete.

        Returns:
            CompletenessReport with missing entity IDs
        """
        entities_file = self.workspace_path / "entities.parquet"

        if not entities_file.exists():
            logger.warning(f"Entities file not found: {entities_file}")
            return CompletenessReport(
                phase="extract_graph",
                total_items=0,
                complete_items=0,
                missing_items=0,
                completeness=0.0,
                missing_ids=[],
            )

        df = pd.read_parquet(entities_file)
        total = len(df)

        # Check for null or empty descriptions
        if "description" in df.columns:
            missing_mask = df["description"].isna() | (
                df["description"].astype(str).str.strip() == ""
            )
            missing_df = df[missing_mask]
            missing_ids = missing_df["id"].tolist() if "id" in df.columns else []
        else:
            missing_df = df
            missing_ids = df["id"].tolist() if "id" in df.columns else []

        missing = len(missing_df)
        complete = total - missing
        completeness = complete / total if total > 0 else 0.0

        logger.info(
            f"Entity descriptions: {complete}/{total} complete ({completeness:.2%})"
        )

        return CompletenessReport(
            phase="extract_graph",
            total_items=total,
            complete_items=complete,
            missing_items=missing,
            completeness=completeness,
            missing_ids=missing_ids,
        )

    def validate_community_reports(self) -> CompletenessReport:
        """Validate community reports are complete.

        Returns:
            CompletenessReport with missing community IDs and hierarchy level details
        """
        # Community reports are stored in community_reports.parquet, not communities.parquet
        reports_file = self.workspace_path / "community_reports.parquet"

        if not reports_file.exists():
            logger.warning(f"Community reports file not found: {reports_file}")
            return CompletenessReport(
                phase="create_community_reports",
                total_items=0,
                complete_items=0,
                missing_items=0,
                completeness=0.0,
                missing_ids=[],
            )

        df = pd.read_parquet(reports_file)
        total = len(df)

        # Check for null or empty full_content
        if "full_content" in df.columns:
            missing_mask = df["full_content"].isna() | (
                df["full_content"].astype(str).str.strip() == ""
            )
            missing_df = df[missing_mask]
            # Use 'community' column for IDs (maps to community_id)
            missing_ids = (
                missing_df["community"].tolist() if "community" in df.columns else []
            )
        else:
            missing_df = df
            missing_ids = df["community"].tolist() if "community" in df.columns else []

        missing = len(missing_df)
        complete = total - missing
        completeness = complete / total if total > 0 else 0.0

        # Calculate details by hierarchy level
        # Level information is in communities.parquet, need to join
        details_by_level = None
        communities_file = self.workspace_path / "communities.parquet"

        if communities_file.exists():
            try:
                communities_df = pd.read_parquet(communities_file)

                # Join reports with communities to get level information
                if "community" in df.columns and "community" in communities_df.columns and "level" in communities_df.columns:
                    # Merge on community ID
                    # Use suffixes to avoid column name conflicts since both dataframes have "level"
                    merged_df = df.merge(
                        communities_df[["community", "level"]],
                        on="community",
                        how="left",
                        suffixes=("_report", "_community")
                    )

                    # After merge, level from communities will be "level_community" if conflict, else "level"
                    level_col = "level_community" if "level_community" in merged_df.columns else "level"

                    if level_col in merged_df.columns:
                        details_by_level = {}
                        for level in sorted(merged_df[level_col].dropna().unique()):
                            level_df = merged_df[merged_df[level_col] == level]
                            level_total = len(level_df)

                            if "full_content" in level_df.columns:
                                level_missing_mask = level_df["full_content"].isna() | (
                                    level_df["full_content"].astype(str).str.strip() == ""
                                )
                                level_missing = level_missing_mask.sum()
                            else:
                                level_missing = level_total

                            level_complete = level_total - level_missing

                            details_by_level[int(level)] = {
                                "total": int(level_total),
                                "complete": int(level_complete),
                                "missing": int(level_missing),
                                "completeness": (
                                    level_complete / level_total if level_total > 0 else 0.0
                                ),
                            }
            except Exception as e:
                logger.warning(f"Could not calculate hierarchy details: {e}")

        logger.info(
            f"Community reports: {complete}/{total} complete ({completeness:.2%})"
        )

        return CompletenessReport(
            phase="create_community_reports",
            total_items=total,
            complete_items=complete,
            missing_items=missing,
            completeness=completeness,
            missing_ids=missing_ids,
            details_by_level=details_by_level,
        )

    def validate_all(self) -> Dict[str, CompletenessReport]:
        """Validate all phases and return comprehensive report.

        Returns:
            Dictionary mapping phase name to CompletenessReport
        """
        reports = {
            "extract_graph": self.validate_entity_descriptions(),
            "create_community_reports": self.validate_community_reports(),
        }

        # Calculate overall completeness
        total_items = sum(r.total_items for r in reports.values())
        complete_items = sum(r.complete_items for r in reports.values())
        overall_completeness = complete_items / total_items if total_items > 0 else 0.0

        logger.info(
            f"Overall completeness: {complete_items}/{total_items} ({overall_completeness:.2%})"
        )

        return reports
