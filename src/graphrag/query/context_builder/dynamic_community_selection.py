# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Algorithm to dynamically select relevant communities with respect to a query."""

import asyncio
import logging
from collections import Counter
from copy import deepcopy
from time import time
from typing import Any

import tiktoken

from graphrag.data_model.community import Community
from graphrag.data_model.community_report import CommunityReport
from graphrag.language_model.protocol.base import ChatModel
from graphrag.query.context_builder.rate_prompt import RATE_QUERY
from graphrag.query.context_builder.rate_relevancy import rate_relevancy

logger = logging.getLogger(__name__)


class DynamicCommunitySelection:
    """Dynamic community selection to select community reports that are relevant to the query.

    Any community report with a rating EQUAL or ABOVE the rating_threshold is considered relevant.
    """

    def __init__(
        self,
        community_reports: list[CommunityReport],
        communities: list[Community],
        model: ChatModel,
        token_encoder: tiktoken.Encoding,
        rate_query: str = RATE_QUERY,
        use_summary: bool = False,
        threshold: int = 1,
        keep_parent: bool = False,
        num_repeats: int = 1,
        max_level: int = 2,
        starting_level: int = 0,  # Allow configuring the starting level
        concurrent_coroutines: int = 25,  # Match vLLM capacity and system config
        model_params: dict[str, Any] | None = None,
    ):
        self.model = model
        self.token_encoder = token_encoder
        self.rate_query = rate_query
        self.num_repeats = num_repeats
        self.use_summary = use_summary
        self.threshold = threshold
        self.keep_parent = keep_parent
        self.max_level = max_level
        self.starting_level = starting_level
        self.semaphore = asyncio.Semaphore(concurrent_coroutines)
        self.model_params = model_params if model_params else {}

        self.reports = {report.community_id: report for report in community_reports}
        self.communities = {community.short_id: community for community in communities}

        # mapping from level to communities
        self.levels: dict[str, list[str]] = {}

        for community in communities:
            if community.level not in self.levels:
                self.levels[community.level] = []
            if community.short_id in self.reports:
                self.levels[community.level].append(community.short_id)

        # start from configured starting level (default 0 for backward compatibility)
        starting_level_key = str(self.starting_level)
        if starting_level_key in self.levels:
            self.starting_communities = self.levels[starting_level_key]
        else:
            # Fallback to level 0 if starting level doesn't exist
            logger.warning(f"Starting level {self.starting_level} not found, falling back to level 0")
            self.starting_communities = self.levels.get("0", [])

    async def select(self, query: str) -> tuple[list[CommunityReport], dict[str, Any]]:
        """
        Select relevant communities with respect to the query.

        Args:
            query: the query to rate against
        """
        start = time()
        queue = deepcopy(self.starting_communities)
        level = self.starting_level  # Start from configured starting level

        ratings = {}  # store the ratings for each community
        llm_info: dict[str, Any] = {
            "llm_calls": 0,
            "prompt_tokens": 0,
            "output_tokens": 0,
        }
        relevant_communities = set()

        # Log drift search initialization
        logger.info(f"Drift search starting at level {self.starting_level} with {len(queue)} communities")
        logger.info(f"Query: {query}")
        logger.info(f"Threshold: {self.threshold}, Max level: {self.max_level}")

        while queue:
            gather_results = await asyncio.gather(*[
                rate_relevancy(
                    query=query,
                    description=(
                        self.reports[community].summary
                        if self.use_summary
                        else self.reports[community].full_content
                    ),
                    model=self.model,
                    token_encoder=self.token_encoder,
                    rate_query=self.rate_query,
                    num_repeats=self.num_repeats,
                    semaphore=self.semaphore,
                    **self.model_params,
                )
                for community in queue
            ])

            # Log current level exploration
            logger.info(f"Level {level}: Evaluating {len(queue)} communities")

            communities_to_rate = []
            level_relevant = []
            level_rejected = []

            for community, result in zip(queue, gather_results, strict=True):
                rating = result["rating"]
                community_title = self.reports[community].title if community in self.reports else community

                logger.info(
                    f"  Community '{community_title}' (ID: {community}) - Rating: {rating}/{self.threshold}"
                )

                ratings[community] = rating
                llm_info["llm_calls"] += result["llm_calls"]
                llm_info["prompt_tokens"] += result["prompt_tokens"]
                llm_info["output_tokens"] += result["output_tokens"]

                if rating >= self.threshold:
                    relevant_communities.add(community)
                    level_relevant.append((community_title, rating))

                    # find children nodes of the current node and append them to the queue
                    # TODO check why some sub_communities are NOT in report_df
                    if community in self.communities:
                        comm_obj = self.communities[community]
                        logger.info(f"    Community {community} has children: {comm_obj.children if hasattr(comm_obj, 'children') else 'NO CHILDREN ATTR'}")

                        children_count = 0
                        missing_children = []
                        for child in self.communities[community].children:
                            # Convert child to string to match report keys (reports are keyed by string IDs)
                            child_str = str(child)
                            if child_str in self.reports:
                                communities_to_rate.append(child_str)
                                children_count += 1
                            else:
                                missing_children.append(child)
                                logger.info(
                                    f"    ⚠ Child community {child} (str: {child_str}) not found in reports (total reports: {len(self.reports)})"
                                )

                        if missing_children:
                            logger.info(f"    Missing children IDs: {missing_children}")
                            logger.info(f"    Available report IDs sample: {list(self.reports.keys())[:10]}")

                        if children_count > 0:
                            logger.info(f"    → Added {children_count} children to explore at level {level + 1}")
                        else:
                            logger.info(f"    ⚠ No children added (had {len(comm_obj.children) if hasattr(comm_obj, 'children') else 0} children, but none in reports)")
                    else:
                        logger.info(f"    ⚠ Community {community} not found in self.communities dict (total: {len(self.communities)})")

                    # remove parent node if the current node is deemed relevant
                    if not self.keep_parent and community in self.communities:
                        relevant_communities.discard(self.communities[community].parent)
                else:
                    level_rejected.append((community_title, rating))

            # Log level summary
            if level_relevant:
                logger.info(f"Level {level} summary: {len(level_relevant)} relevant, {len(level_rejected)} rejected")
                logger.info(f"  Relevant: {[f'{title} ({rating})' for title, rating in level_relevant[:5]]}")
            else:
                logger.info(f"Level {level} summary: No communities met threshold ({self.threshold})")
            queue = communities_to_rate
            level += 1
            if (
                (len(queue) == 0)
                and (len(relevant_communities) == 0)
                and (str(level) in self.levels)
                and (level <= self.max_level)
            ):
                logger.info(
                    f"⚠ FALLBACK: No relevant communities found yet. "
                    f"Adding all {len(self.levels[str(level)])} communities at level {level} to rate."
                )
                # append all communities at the next level to queue
                queue = self.levels[str(level)]

        community_reports = [
            self.reports[community] for community in relevant_communities
        ]
        end = time()

        # Log final summary
        logger.info("=" * 80)
        logger.info("DRIFT SEARCH COMPLETED")
        logger.info("=" * 80)
        logger.info(f"Duration: {int(end - start)}s")
        logger.info(f"Selected {len(relevant_communities)} out of {len(self.reports)} total communities")
        logger.info(f"Rating distribution: {dict(sorted(Counter(ratings.values()).items()))}")
        logger.info(f"LLM calls: {llm_info['llm_calls']}, Prompt tokens: {llm_info['prompt_tokens']}, Output tokens: {llm_info['output_tokens']}")

        # Group selected communities by level
        selected_by_level = {}
        for community_id in relevant_communities:
            if community_id in self.communities:
                comm_level = self.communities[community_id].level
                if comm_level not in selected_by_level:
                    selected_by_level[comm_level] = []
                title = self.reports[community_id].title if community_id in self.reports else community_id
                selected_by_level[comm_level].append(title)

        logger.info("\nSelected communities by level:")
        for lvl in sorted(selected_by_level.keys()):
            logger.info(f"  Level {lvl}: {len(selected_by_level[lvl])} communities")
            for title in selected_by_level[lvl][:3]:  # Show first 3
                logger.info(f"    - {title}")
            if len(selected_by_level[lvl]) > 3:
                logger.info(f"    ... and {len(selected_by_level[lvl]) - 3} more")
        logger.info("=" * 80)

        logger.debug(
            "dynamic community selection (took: %ss)\n"
            "\trating distribution %s\n"
            "\t%s out of %s community reports are relevant\n"
            "\tprompt tokens: %s, output tokens: %s",
            int(end - start),
            dict(sorted(Counter(ratings.values()).items())),
            len(relevant_communities),
            len(self.reports),
            llm_info["prompt_tokens"],
            llm_info["output_tokens"],
        )

        llm_info["ratings"] = ratings
        return community_reports, llm_info
