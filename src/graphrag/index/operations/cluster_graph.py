# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing cluster_graph, apply_clustering and run_layout methods definition."""

import logging
import os

import networkx as nx
from graspologic.partition import hierarchical_leiden

from graphrag.index.utils.stable_lcc import stable_largest_connected_component

Communities = list[tuple[int, int, int, list[str]]]


logger = logging.getLogger(__name__)


# Feature flag to enable new pyramid hierarchy algorithm
USE_PYRAMID_HIERARCHY = os.environ.get("GRAPHRAG_USE_PYRAMID_HIERARCHY", "true").lower() == "true"

# Base resolution multiplier for pyramid hierarchy (configurable)
# Higher values = more fine-grained base communities (15.0 recommended for ~75K entities)
BASE_RESOLUTION_MULTIPLIER = float(os.environ.get("GRAPHRAG_BASE_RESOLUTION_MULTIPLIER", "15.0"))

# Consolidation scaling factor for pyramid hierarchy (configurable)
# Lower values = steeper consolidation (fewer levels), higher values = gentler (more levels)
# 0.6 = gentle (8-10 levels), 0.4 = moderate (6-8 levels), 0.3 = steep (5-6 levels)
CONSOLIDATION_SCALING_FACTOR = float(os.environ.get("GRAPHRAG_CONSOLIDATION_SCALING_FACTOR", "0.4"))

# Adaptive scaling: automatically adjust multiplier based on graph size
# When enabled, overrides BASE_RESOLUTION_MULTIPLIER with calculated value
USE_ADAPTIVE_SCALING = os.environ.get("GRAPHRAG_USE_ADAPTIVE_SCALING", "true").lower() == "true"
ADAPTIVE_TARGET_BASE_SIZE = int(os.environ.get("GRAPHRAG_ADAPTIVE_TARGET_BASE_SIZE", "20"))


def cluster_graph(
    graph: nx.Graph,
    max_cluster_size: int,
    use_lcc: bool,
    seed: int | None = None,
    resolution: float = 1.0,
) -> Communities:
    """Apply a hierarchical clustering algorithm to a graph."""
    if len(graph.nodes) == 0:
        logger.warning("Graph has no nodes")
        return []

    # Choose clustering algorithm based on feature flag
    if USE_PYRAMID_HIERARCHY:
        node_id_to_community_map, parent_mapping = _compute_pyramid_communities(
            graph=graph,
            max_cluster_size=max_cluster_size,
            use_lcc=use_lcc,
            seed=seed,
            resolution=resolution,
        )
    else:
        node_id_to_community_map, parent_mapping = _compute_leiden_communities(
            graph=graph,
            max_cluster_size=max_cluster_size,
            use_lcc=use_lcc,
            seed=seed,
            resolution=resolution,
        )

    levels = sorted(node_id_to_community_map.keys())

    # Pyramid hierarchy: already built with Level 0 = root (compatible with queries)
    # hierarchical_leiden: needs inversion for compatibility
    if not USE_PYRAMID_HIERARCHY:
        # Invert hierarchical_leiden levels: 0 (coarse/root) → max, max (fine) → 0
        max_level = max(levels) if levels else 0
        inverted_map = {}
        for leiden_level, communities in node_id_to_community_map.items():
            graphrag_level = max_level - leiden_level
            inverted_map[graphrag_level] = communities
        node_id_to_community_map = inverted_map
        levels = sorted(node_id_to_community_map.keys())

    clusters: dict[int, dict[int, list[str]]] = {}
    for level in levels:
        result = {}
        clusters[level] = result
        for node_id, raw_community_id in node_id_to_community_map[level].items():
            community_id = raw_community_id
            if community_id not in result:
                result[community_id] = []
            result[community_id].append(node_id)

    results: Communities = []
    for level in clusters:
        for cluster_id, nodes in clusters[level].items():
            # Get parent from hierarchy, defaulting to -1 if this is a root community
            parent_id = parent_mapping.get(cluster_id, -1)
            results.append((level, cluster_id, parent_id, nodes))
    return results


# Taken from graph_intelligence & adapted
def _compute_leiden_communities(
    graph: nx.Graph | nx.DiGraph,
    max_cluster_size: int,
    use_lcc: bool,
    seed: int | None = None,
    resolution: float = 1.0,
) -> tuple[dict[int, dict[str, int]], dict[int, int]]:
    """Return Leiden root communities and their hierarchy mapping."""
    if use_lcc:
        graph = stable_largest_connected_component(graph)

    logger.info(
        f"Running hierarchical Leiden clustering with max_cluster_size={max_cluster_size}, resolution={resolution}"
    )

    community_mapping = hierarchical_leiden(
        graph, max_cluster_size=max_cluster_size, random_seed=seed, resolution=resolution
    )
    results: dict[int, dict[str, int]] = {}
    hierarchy: dict[int, int] = {}
    for partition in community_mapping:
        results[partition.level] = results.get(partition.level, {})
        results[partition.level][partition.node] = partition.cluster

        hierarchy[partition.cluster] = (
            partition.parent_cluster if partition.parent_cluster is not None else -1
        )

    return results, hierarchy


def _compute_pyramid_communities(
    graph: nx.Graph | nx.DiGraph,
    max_cluster_size: int,
    use_lcc: bool,
    seed: int | None = None,
    resolution: float = 1.0,
    base_resolution_multiplier: float | None = None,
    consolidation_scaling_factor: float | None = None,
) -> tuple[dict[int, dict[str, int]], dict[int, int]]:
    """Build proper bottom-up pyramid hierarchy with query-compatible level numbering.

    Creates a pyramid structure:
    - Level 0: ROOT/TOP - Coarse (few large communities, entire corpus) - USED FOR QUERIES
    - Level N: BASE - Fine-grained (many small communities, ~15-25 entities each)

    This matches GraphRAG query expectations where Level 0 is the starting point.

    Parameters same as _compute_leiden_communities, but:
    - resolution: Controls base granularity (higher = more fine-grained)
    - max_cluster_size: Target size for base communities
    """
    from graspologic.partition import leiden

    # Handle edge cases before LCC extraction
    if len(graph.nodes()) == 0:
        logger.warning("Graph has no nodes")
        return {}, {}

    if len(graph.nodes()) == 1:
        # Single node graph - create trivial hierarchy
        single_node = list(graph.nodes())[0]
        logger.info("Single node graph, creating trivial hierarchy")
        return {0: {single_node: 0}}, {0: -1}  # Level 0, community 0, no parent

    if use_lcc and len(graph.nodes()) > 1:
        graph = stable_largest_connected_component(graph)

        # Check if graph became empty after LCC extraction
        if len(graph.nodes()) == 0:
            logger.warning("Graph became empty after LCC extraction")
            return {}, {}

    num_nodes = len(graph.nodes())

    logger.info(
        f"Building pyramid hierarchy: {num_nodes} nodes, resolution={resolution}, target_base_size={max_cluster_size}"
    )

    # Step 1: Create fine-grained base using high resolution for small communities
    # Target: 15-25 entities per community (~3000-5000 communities for 75K entities)
    # Use configurable multiplier (parameter or env var) - adjust via GRAPHRAG_BASE_RESOLUTION_MULTIPLIER env var
    # For resolution=1.0 with multiplier=15.0: base_resolution=15.0 creates ~3000-5000 communities for 75K entities
    if base_resolution_multiplier is None:
        # Use adaptive scaling if enabled
        if USE_ADAPTIVE_SCALING:
            # Hybrid adaptive scaling to achieve target base community size (~20 entities each)
            #
            # The relationship between graph size and optimal multiplier is non-linear due to
            # how Leiden clustering scales with graph structure and connectivity.
            #
            # Validated empirical results:
            # - Small (560 nodes):  multiplier 1.28 → 29 communities,   avg 19.3 entities ✓
            # - Large (75K nodes):  multiplier 170  → 3851 communities, avg 19.6 entities ✓
            #
            # Implementation uses hybrid approach:
            # - Small graphs (< 2000 nodes): Linear formula
            # - Large graphs (≥ 2000 nodes): Power law formula
            target_communities = num_nodes / ADAPTIVE_TARGET_BASE_SIZE

            # For very small graphs (< 2000 nodes), use simpler linear scaling
            # Leiden clustering behaves non-linearly at small scales, making power law unreliable
            if num_nodes < 2000:
                # Linear approximation: multiplier ≈ num_nodes / 437
                # Empirically calibrated: 560 nodes → 1.28, 1000 nodes → 2.29, 2000 nodes → 4.58
                # Targets ~20 entities per base community
                base_resolution_multiplier = num_nodes / 437.0
            else:
                # Power law: multiplier = k × (num_nodes ^ α)
                # Derived from empirical testing:
                # - 85K nodes, multiplier 170 → ~3900 communities, ~19.5 avg entities (optimal)
                # - 2K nodes, multiplier 8 → ~100 communities, ~20 avg entities (target)
                # α = 0.7856, k = 0.022665
                base_resolution_multiplier = 0.022665 * (num_nodes ** 0.7856)

            # Clamp to reasonable bounds and warn about extreme values
            raw_multiplier = base_resolution_multiplier
            base_resolution_multiplier = max(2.0, min(base_resolution_multiplier, 500.0))

            # Validation warnings for edge cases
            if raw_multiplier < 1.0:
                logger.warning(
                    f"Very low multiplier ({raw_multiplier:.2f}) for small graph ({num_nodes} nodes). "
                    f"Clamped to minimum 2.0. Communities may be larger than target."
                )
            elif raw_multiplier > 500.0:
                logger.warning(
                    f"Very high multiplier ({raw_multiplier:.1f}) for large graph ({num_nodes} nodes). "
                    f"Clamped to maximum 500.0. Consider adjusting power law constants if needed."
                )

            logger.info(
                f"Adaptive scaling: {num_nodes} nodes, target {ADAPTIVE_TARGET_BASE_SIZE} entities/community "
                f"→ multiplier={base_resolution_multiplier:.1f} (target {target_communities:.0f} communities)"
            )
        else:
            base_resolution_multiplier = BASE_RESOLUTION_MULTIPLIER
    if consolidation_scaling_factor is None:
        consolidation_scaling_factor = CONSOLIDATION_SCALING_FACTOR

    base_resolution = resolution * base_resolution_multiplier
    base_communities = leiden(
        graph,
        resolution=base_resolution,
        random_seed=seed,
        use_modularity=True,
        trials=3  # Run multiple trials for better quality
    )

    # Build all levels first (storing them temporarily)
    temp_levels: list[dict[str, int]] = [base_communities]
    hierarchy: dict[int, int] = {}

    num_base = len(set(base_communities.values()))
    avg_base_size = num_nodes / num_base if num_base > 0 else 0

    logger.info(
        f"Temp Level 0 (base): {num_base} communities, avg {avg_base_size:.1f} entities/community"
    )

    # Build consolidation levels using metagraph approach
    current_communities = base_communities.copy()
    current_cluster_id_offset = max(base_communities.values()) + 1
    temp_level = 1

    # Target: 1-3 communities at top level for optimal global queries
    # Using 0.2 power gives gentler consolidation: 3000^0.2 ≈ 4.7 → 4 communities
    # This creates more gradual pyramid levels instead of collapsing too quickly
    target_top = max(3, int(num_base ** 0.2))

    while len(set(current_communities.values())) > target_top and temp_level < 15:
        # Build metagraph where communities are nodes
        metagraph = _build_community_metagraph(graph, current_communities)

        if len(metagraph.nodes()) <= 1:
            logger.info(f"Cannot consolidate further at temp level {temp_level}, stopping")
            break

        # Consolidate with decreasing resolution (broader groupings at higher levels)
        # Use configurable scaling factor to control pyramid steepness
        # Lower factor = steeper (fewer levels), higher factor = gentler (more levels)
        # temp_level=1: factor^1 × base, temp_level=2: factor^2 × base, etc.
        # factor=0.4: steeper consolidation (6-8 levels)
        # factor=0.6: gentler consolidation (8-10 levels)
        consolidation_resolution = base_resolution * (consolidation_scaling_factor ** temp_level)

        try:
            parent_communities = leiden(
                metagraph,
                resolution=consolidation_resolution,
                random_seed=seed,
                use_modularity=True,
                trials=1
            )
        except Exception as e:
            logger.warning(f"Leiden failed at temp level {temp_level}: {e}, stopping consolidation")
            break

        # Map original nodes to parent communities
        next_level = {}

        # First, create unique parent community IDs for successfully clustered communities
        unique_parent_communities = set(parent_communities.values())
        parent_id_map = {}
        for idx, parent_comm in enumerate(sorted(unique_parent_communities)):
            parent_id_map[parent_comm] = current_cluster_id_offset + idx

        # Track next available ID for isolated communities
        next_singleton_id = current_cluster_id_offset + len(parent_id_map)
        isolated_comm_to_parent = {}  # Track isolated community → singleton parent mapping

        # Map each node to its parent community
        for node, child_comm in current_communities.items():
            parent_comm = parent_communities.get(child_comm)

            if parent_comm is not None:
                # Community was successfully clustered in metagraph
                parent_comm_id = parent_id_map[parent_comm]
                next_level[node] = parent_comm_id

                # Track hierarchy (child -> parent mapping)
                if child_comm not in hierarchy:
                    hierarchy[child_comm] = parent_comm_id
            else:
                # Community not found in metagraph clustering (isolated community)
                # This can happen if a community has no edges to other communities
                # Each isolated community gets its own unique singleton parent
                if child_comm not in isolated_comm_to_parent:
                    isolated_comm_to_parent[child_comm] = next_singleton_id
                    next_singleton_id += 1

                singleton_parent_id = isolated_comm_to_parent[child_comm]
                next_level[node] = singleton_parent_id

                if child_comm not in hierarchy:
                    hierarchy[child_comm] = singleton_parent_id

        temp_levels.append(next_level)

        # Log statistics
        num_communities = len(set(next_level.values()))
        avg_size = num_nodes / num_communities if num_communities > 0 else 0

        logger.info(
            f"Temp Level {temp_level}: {num_communities} communities, avg {avg_size:.1f} entities/community"
        )

        # Check if we made progress
        if num_communities >= len(set(current_communities.values())):
            logger.warning(f"No consolidation at temp level {temp_level} ({num_communities} communities), stopping")
            break

        # Prepare next iteration
        current_communities = next_level
        # Update offset to be after all IDs used in this level
        current_cluster_id_offset = max(next_level.values()) + 1
        temp_level += 1

    # Step 2: Reverse levels so Level 0 = top (root), Level N = base
    # This makes it compatible with GraphRAG queries which start at Level 0
    max_temp_level = len(temp_levels) - 1
    results: dict[int, dict[str, int]] = {}

    for temp_lvl, communities in enumerate(temp_levels):
        # Reverse: temp 0 (base) → Level max, temp max (top) → Level 0
        final_level = max_temp_level - temp_lvl
        results[final_level] = communities

        num_communities = len(set(communities.values()))
        avg_size = num_nodes / num_communities if num_communities > 0 else 0
        level_type = "ROOT" if final_level == 0 else f"L{final_level}"

        logger.info(
            f"Level {final_level} ({level_type}): {num_communities} communities, avg {avg_size:.1f} entities/community"
        )

    logger.info(f"Built {len(temp_levels)} level pyramid (Level 0 = ROOT for queries, Level {max_temp_level} = BASE)")

    return results, hierarchy


def _build_community_metagraph(
    graph: nx.Graph,
    node_to_community: dict[str, int]
) -> nx.Graph:
    """Build metagraph where communities are nodes, connected by inter-community edges.

    Edge weights = number of connections between communities in original graph.
    """
    metagraph = nx.Graph()

    # Get unique communities
    communities = set(node_to_community.values())
    metagraph.add_nodes_from(communities)

    # Count inter-community edges
    edge_weights: dict[tuple[int, int], float] = {}

    for u, v in graph.edges():
        comm_u = node_to_community.get(u)
        comm_v = node_to_community.get(v)

        # Only count edges between different communities
        if comm_u is not None and comm_v is not None and comm_u != comm_v:
            edge = tuple(sorted([comm_u, comm_v]))
            edge_weights[edge] = edge_weights.get(edge, 0.0) + 1.0

    # Add weighted edges
    for (comm_u, comm_v), weight in edge_weights.items():
        metagraph.add_edge(comm_u, comm_v, weight=weight)

    return metagraph
