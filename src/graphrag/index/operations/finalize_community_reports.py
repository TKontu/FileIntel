# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""All the steps to transform final entities."""

from uuid import uuid4

import pandas as pd

from graphrag.data_model.schemas import COMMUNITY_REPORTS_FINAL_COLUMNS


def finalize_community_reports(
    reports: pd.DataFrame,
    communities: pd.DataFrame,
) -> pd.DataFrame:
    """All the steps to transform final community reports."""
    # Merge with communities to add shared fields
    community_reports = reports.merge(
        communities.loc[:, ["community", "parent", "children", "size", "period"]],
        on="community",
        how="left",
        copy=False,
    )

    community_reports["community"] = community_reports["community"].astype(int)
    community_reports["human_readable_id"] = community_reports["community"]

    # CRITICAL FIX: Preserve existing IDs when resuming from partial results
    # Only generate new IDs for reports that don't already have one
    if "id" in community_reports.columns:
        # Fill missing IDs with new UUIDs (for new reports)
        community_reports["id"] = community_reports["id"].apply(
            lambda x: x if pd.notna(x) and x != "" else uuid4().hex
        )
    else:
        # No existing IDs, generate for all (backwards compatible)
        community_reports["id"] = [uuid4().hex for _ in range(len(community_reports))]

    return community_reports.loc[
        :,
        COMMUNITY_REPORTS_FINAL_COLUMNS,
    ]
