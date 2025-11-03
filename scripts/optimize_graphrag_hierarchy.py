#!/usr/bin/env python3
"""
GraphRAG Hierarchy Optimizer

Analyzes existing graph data and helps optimize clustering parameters for pyramid hierarchy.
Can be run standalone to test different parameters without full re-indexing.

Usage:
    python scripts/optimize_graphrag_hierarchy.py --collection-id <id>
    python scripts/optimize_graphrag_hierarchy.py --relationships-file <path>
    python scripts/optimize_graphrag_hierarchy.py --relationships-file <path> --resolution 1.0 --multiplier 15.0
    python scripts/optimize_graphrag_hierarchy.py --relationships-file <path> --test-range
"""

import argparse
import os
import sys
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import networkx as nx

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from graphrag.index.operations.cluster_graph import (
    _compute_pyramid_communities,
    _build_community_metagraph,
)
from graspologic.partition import leiden

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_relationships(file_path: str) -> pd.DataFrame:
    """Load relationships parquet file."""
    logger.info(f"Loading relationships from: {file_path}")
    df = pd.read_parquet(file_path)
    logger.info(f"Loaded {len(df)} relationships")
    return df


def create_graph_from_relationships(relationships: pd.DataFrame) -> nx.Graph:
    """Create NetworkX graph from relationships dataframe."""
    logger.info("Creating graph from relationships...")

    # Create graph with weighted edges
    graph = nx.Graph()

    for _, row in relationships.iterrows():
        source = row.get('source', row.get('source_id'))
        target = row.get('target', row.get('target_id'))
        weight = row.get('weight', 1.0)

        if pd.notna(source) and pd.notna(target):
            if graph.has_edge(source, target):
                # Sum weights for duplicate edges
                graph[source][target]['weight'] += weight
            else:
                graph.add_edge(source, target, weight=weight)

    num_nodes = len(graph.nodes())
    num_edges = len(graph.edges())
    logger.info(f"Created graph: {num_nodes} nodes, {num_edges} edges")

    return graph


def analyze_hierarchy(
    graph: nx.Graph,
    resolution: float = 1.0,
    base_resolution_multiplier: float = 15.0,
    max_cluster_size: int = 25,
    consolidation_scaling_factor: float = 0.4,
    seed: Optional[int] = None
) -> dict:
    """
    Analyze pyramid hierarchy with given parameters.

    Returns detailed statistics about the resulting hierarchy.
    """
    num_nodes = len(graph.nodes())

    logger.info("=" * 80)
    logger.info(f"Testing Parameters:")
    logger.info(f"  Resolution: {resolution}")
    logger.info(f"  Base Resolution Multiplier: {base_resolution_multiplier}")
    logger.info(f"  Consolidation Scaling Factor: {consolidation_scaling_factor}")
    logger.info(f"  Max Cluster Size: {max_cluster_size}")
    logger.info(f"  Effective Base Resolution: {resolution * base_resolution_multiplier}")
    logger.info(f"  Total Nodes: {num_nodes}")
    logger.info("=" * 80)

    # Run pyramid hierarchy algorithm
    try:
        node_to_community_map, parent_mapping = _compute_pyramid_communities(
            graph=graph,
            max_cluster_size=max_cluster_size,
            use_lcc=True,
            seed=seed,
            resolution=resolution,
            base_resolution_multiplier=base_resolution_multiplier,
            consolidation_scaling_factor=consolidation_scaling_factor,
        )
    except Exception as e:
        logger.error(f"Failed to compute pyramid: {e}")
        return {"error": str(e)}

    # Analyze results
    levels = sorted(node_to_community_map.keys(), reverse=True)

    stats = {
        "num_nodes": num_nodes,
        "resolution": resolution,
        "base_resolution_multiplier": base_resolution_multiplier,
        "max_cluster_size": max_cluster_size,
        "effective_base_resolution": resolution * base_resolution_multiplier,
        "num_levels": len(levels),
        "levels": {}
    }

    logger.info("\n" + "=" * 80)
    logger.info("HIERARCHY RESULTS:")
    logger.info("=" * 80)

    for level in levels:
        communities = node_to_community_map[level]
        unique_communities = set(communities.values())
        num_communities = len(unique_communities)
        avg_size = num_nodes / num_communities if num_communities > 0 else 0

        level_type = "ROOT" if level == 0 else "BASE" if level == max(levels) else f"L{level}"

        stats["levels"][level] = {
            "num_communities": num_communities,
            "avg_size": avg_size,
            "type": level_type
        }

        logger.info(f"Level {level} ({level_type}): {num_communities} communities, avg {avg_size:.1f} entities/community")

    logger.info("=" * 80)

    # Quality assessment
    base_level = max(levels)
    base_communities = stats["levels"][base_level]["num_communities"]
    base_avg_size = stats["levels"][base_level]["avg_size"]
    root_communities = stats["levels"][0]["num_communities"]

    # Target: 3000-5000 base communities with 15-25 entities each
    target_base_min = 3000
    target_base_max = 5000
    target_size_min = 15
    target_size_max = 25

    quality_issues = []

    if base_communities < target_base_min:
        quality_issues.append(f"❌ Base has too few communities ({base_communities} < {target_base_min})")
    elif base_communities > target_base_max:
        quality_issues.append(f"⚠️  Base has many communities ({base_communities} > {target_base_max})")
    else:
        logger.info(f"✅ Base community count is optimal ({base_communities})")

    if base_avg_size < target_size_min:
        quality_issues.append(f"⚠️  Base communities too small (avg {base_avg_size:.1f} < {target_size_min})")
    elif base_avg_size > target_size_max:
        quality_issues.append(f"❌ Base communities too large (avg {base_avg_size:.1f} > {target_size_max})")
    else:
        logger.info(f"✅ Base community size is optimal (avg {base_avg_size:.1f})")

    if root_communities > 5:
        quality_issues.append(f"⚠️  Root has many communities ({root_communities} > 5)")
    else:
        logger.info(f"✅ Root community count is optimal ({root_communities})")

    if len(levels) < 5:
        quality_issues.append(f"⚠️  Few pyramid levels ({len(levels)} < 5)")
    elif len(levels) > 10:
        quality_issues.append(f"⚠️  Many pyramid levels ({len(levels)} > 10)")
    else:
        logger.info(f"✅ Pyramid depth is optimal ({len(levels)} levels)")

    if quality_issues:
        logger.warning("\nQuality Issues:")
        for issue in quality_issues:
            logger.warning(f"  {issue}")
    else:
        logger.info("\n✅ All quality checks passed!")

    stats["quality_issues"] = quality_issues

    return stats


def recommend_parameters(num_nodes: int, current_stats: dict) -> dict:
    """
    Recommend optimal parameters based on graph size and current results.
    """
    logger.info("\n" + "=" * 80)
    logger.info("PARAMETER RECOMMENDATIONS:")
    logger.info("=" * 80)

    # Target base communities: ~3000-5000 for 75K entities
    # Scale proportionally for different graph sizes
    target_base = max(3000, int(num_nodes / 20))  # ~20 entities per base community

    current_base = current_stats["levels"][max(current_stats["levels"].keys())]["num_communities"]

    # Estimate needed multiplier adjustment
    current_multiplier = current_stats["base_resolution_multiplier"]
    current_resolution = current_stats["resolution"]

    # Simple linear scaling estimate
    if current_base > 0:
        scaling_factor = target_base / current_base
        recommended_multiplier = current_multiplier * (scaling_factor ** 0.5)  # Square root for gentler adjustment
    else:
        recommended_multiplier = 15.0

    recommendations = {
        "current": {
            "resolution": current_resolution,
            "multiplier": current_multiplier,
            "base_communities": current_base
        },
        "recommended": {
            "resolution": current_resolution,
            "multiplier": round(recommended_multiplier, 1),
            "expected_base_communities": target_base
        }
    }

    logger.info(f"Current Configuration:")
    logger.info(f"  GRAPHRAG_LEIDEN_RESOLUTION={current_resolution}")
    logger.info(f"  GRAPHRAG_BASE_RESOLUTION_MULTIPLIER={current_multiplier}")
    logger.info(f"  → Base Communities: {current_base}")

    logger.info(f"\nRecommended Configuration:")
    logger.info(f"  GRAPHRAG_LEIDEN_RESOLUTION={current_resolution}")
    logger.info(f"  GRAPHRAG_BASE_RESOLUTION_MULTIPLIER={recommended_multiplier:.1f}")
    logger.info(f"  → Expected Base Communities: ~{target_base}")

    logger.info("=" * 80)

    return recommendations


def test_parameter_range(
    graph: nx.Graph,
    resolution_values: list[float] = None,
    multiplier_values: list[float] = None,
    max_cluster_size_values: list[int] = None,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Test multiple parameter combinations and return results as DataFrame.
    """
    if resolution_values is None:
        resolution_values = [0.8, 1.0, 1.2]

    if multiplier_values is None:
        multiplier_values = [10.0, 15.0, 20.0]

    if max_cluster_size_values is None:
        max_cluster_size_values = [25]  # Usually doesn't need variation for pyramid

    total_combinations = len(resolution_values) * len(multiplier_values) * len(max_cluster_size_values)
    logger.info(f"\nTesting parameter combinations:")
    logger.info(f"  Resolutions: {resolution_values}")
    logger.info(f"  Multipliers: {multiplier_values}")
    logger.info(f"  Max Cluster Sizes: {max_cluster_size_values}")
    logger.info(f"  Total combinations: {total_combinations}")

    results = []

    for resolution in resolution_values:
        for multiplier in multiplier_values:
            for max_cluster_size in max_cluster_size_values:
                # Temporarily set environment variable for the test
                os.environ["GRAPHRAG_BASE_RESOLUTION_MULTIPLIER"] = str(multiplier)

                stats = analyze_hierarchy(graph, resolution, multiplier, max_cluster_size, seed)

                if "error" not in stats:
                    base_level = max(stats["levels"].keys())
                    root_level = 0

                    results.append({
                        "resolution": resolution,
                        "multiplier": multiplier,
                        "max_cluster_size": max_cluster_size,
                        "num_levels": stats["num_levels"],
                        "base_communities": stats["levels"][base_level]["num_communities"],
                        "base_avg_size": stats["levels"][base_level]["avg_size"],
                        "root_communities": stats["levels"][root_level]["num_communities"],
                        "quality_score": len(stats["quality_issues"])  # Lower is better
                    })

    df = pd.DataFrame(results)
    df = df.sort_values("quality_score")

    logger.info("\n" + "=" * 80)
    logger.info("TOP 5 PARAMETER COMBINATIONS (by quality):")
    logger.info("=" * 80)
    print(df.head(10).to_string(index=False))

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Optimize GraphRAG pyramid hierarchy parameters"
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--collection-id",
        help="Collection ID (will look for relationships.parquet in standard location)"
    )
    input_group.add_argument(
        "--relationships-file",
        help="Path to relationships.parquet file"
    )

    # Parameter options
    parser.add_argument(
        "--resolution",
        type=float,
        default=1.0,
        help="Leiden resolution parameter (default: 1.0)"
    )
    parser.add_argument(
        "--multiplier",
        type=float,
        default=15.0,
        help="Base resolution multiplier (default: 15.0)"
    )
    parser.add_argument(
        "--max-cluster-size",
        type=int,
        default=25,
        help="Max cluster size parameter (default: 25, note: not directly used by pyramid algorithm)"
    )
    parser.add_argument(
        "--consolidation-factor",
        type=float,
        default=0.4,
        help="Consolidation scaling factor - lower=steeper (fewer levels), higher=gentler (more levels). 0.3=steep, 0.4=moderate, 0.6=gentle (default: 0.4)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility"
    )

    # Analysis options
    parser.add_argument(
        "--test-range",
        action="store_true",
        help="Test range of parameters to find optimal configuration"
    )
    parser.add_argument(
        "--output-csv",
        help="Save test results to CSV file"
    )

    args = parser.parse_args()

    # Determine input file
    if args.collection_id:
        # Try multiple possible base paths (host vs container)
        possible_base_paths = [
            os.environ.get("GRAPHRAG_INDEX_PATH"),  # From env var
            os.environ.get("HOST_GRAPHRAG_PATH"),  # From Docker env
            "/data",  # Docker container path
            "/mnt/fileintel/graphrag_index",  # Common host mount
            "./graphrag_indices",  # Local relative path
        ]

        relationships_file = None
        for base_path in possible_base_paths:
            if base_path:
                candidate = os.path.join(
                    base_path,
                    "graphrag_indices",
                    args.collection_id,
                    "output",
                    "relationships.parquet"
                )
                if os.path.exists(candidate):
                    relationships_file = candidate
                    logger.info(f"Found relationships file at: {relationships_file}")
                    break

        if not relationships_file:
            # Try without 'graphrag_indices' subdirectory (if base path already includes it)
            for base_path in possible_base_paths:
                if base_path:
                    candidate = os.path.join(
                        base_path,
                        args.collection_id,
                        "output",
                        "relationships.parquet"
                    )
                    if os.path.exists(candidate):
                        relationships_file = candidate
                        logger.info(f"Found relationships file at: {relationships_file}")
                        break

        if not relationships_file:
            logger.error(f"Could not find relationships.parquet for collection {args.collection_id}")
            logger.error(f"Tried base paths: {[p for p in possible_base_paths if p]}")
            logger.error(f"You can specify the file directly with --relationships-file")
            sys.exit(1)
    else:
        relationships_file = args.relationships_file

    if not os.path.exists(relationships_file):
        logger.error(f"Relationships file not found: {relationships_file}")
        sys.exit(1)

    # Load data and create graph
    relationships = load_relationships(relationships_file)
    graph = create_graph_from_relationships(relationships)

    if args.test_range:
        # Test multiple parameter combinations
        results_df = test_parameter_range(graph, seed=args.seed)

        if args.output_csv:
            results_df.to_csv(args.output_csv, index=False)
            logger.info(f"\nResults saved to: {args.output_csv}")
    else:
        # Single parameter test
        # Set environment variable to match command line arg
        os.environ["GRAPHRAG_BASE_RESOLUTION_MULTIPLIER"] = str(args.multiplier)

        stats = analyze_hierarchy(
            graph,
            args.resolution,
            args.multiplier,
            args.max_cluster_size,
            args.consolidation_factor,
            args.seed
        )

        if "error" not in stats:
            recommend_parameters(len(graph.nodes()), stats)


if __name__ == "__main__":
    main()
