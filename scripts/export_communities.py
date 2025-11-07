#!/usr/bin/env python3
"""
Export GraphRAG communities to a markdown file.

Usage:
    python scripts/export_communities.py <collection_id> [output_file]
    python scripts/export_communities.py 6525aacb-55b1-4a88-aaaa-a4211d03beba
    python scripts/export_communities.py 6525aacb-55b1-4a88-aaaa-a4211d03beba communities_export.md
"""

import sys
import pandas as pd
from pathlib import Path
from datetime import datetime


def export_communities(collection_id: str, output_file: str = None, base_path: str = "/mnt/fileintel/graphrag_index"):
    """Export all communities to a markdown file."""

    # Default output filename
    if output_file is None:
        output_file = f"communities_{collection_id[:8]}.md"

    # Read parquet file
    parquet_path = Path(base_path) / collection_id / "output" / "community_reports.parquet"

    if not parquet_path.exists():
        print(f"Error: Community reports not found at {parquet_path}")
        sys.exit(1)

    df = pd.read_parquet(parquet_path)

    # Sort by level and title
    df = df.sort_values(['level', 'title'])

    # Generate markdown
    lines = []
    lines.append(f"# GraphRAG Communities Export")
    lines.append(f"")
    lines.append(f"**Collection ID:** `{collection_id}`")
    lines.append(f"**Exported:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Total Communities:** {len(df)}")
    lines.append(f"")

    # Summary table
    lines.append("## Distribution by Level")
    lines.append("")
    lines.append("| Level | Count |")
    lines.append("|-------|-------|")
    for level, count in df['level'].value_counts().sort_index().items():
        lines.append(f"| {level} | {count} |")
    lines.append("")

    # Communities by level
    for level in sorted(df['level'].unique()):
        level_communities = df[df['level'] == level]
        lines.append(f"## Level {level} Communities ({len(level_communities)})")
        lines.append("")

        for idx, row in level_communities.iterrows():
            lines.append(f"### {row['title']}")
            lines.append(f"")
            lines.append(f"**Community ID:** `{row['community']}`")
            lines.append(f"")

            # Truncate summary to 300 chars
            summary = row['summary']
            if len(summary) > 300:
                summary = summary[:297] + "..."
            lines.append(f"**Summary:** {summary}")
            lines.append(f"")
            lines.append("---")
            lines.append("")

    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"✓ Exported {len(df)} communities to {output_file}")
    print(f"  Levels: {df['level'].min()} - {df['level'].max()}")
    print(f"  File size: {Path(output_file).stat().st_size / 1024:.1f} KB")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/export_communities.py <collection_id> [output_file]")
        print("\nExample:")
        print("  python scripts/export_communities.py 6525aacb-55b1-4a88-aaaa-a4211d03beba")
        print("  python scripts/export_communities.py 6525aacb-55b1-4a88-aaaa-a4211d03beba my_communities.md")
        sys.exit(1)

    collection_id = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    export_communities(collection_id, output_file)


if __name__ == "__main__":
    main()
