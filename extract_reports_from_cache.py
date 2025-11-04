#!/usr/bin/env python3
"""
Extract community reports from GraphRAG cache and build partial community_reports.parquet
"""
import json
import os
import re
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

def extract_community_id_from_prompt(prompt_text: str) -> int | None:
    """Extract community ID by parsing the entities section."""
    try:
        # Look for the community pattern in entity descriptions
        # Entities are in format: human_readable_id,title,description,degree
        lines = prompt_text.split('\n')

        # Find the entities section
        in_entities = False
        for line in lines:
            if line.strip() == '-----Entities-----':
                in_entities = True
                continue
            if in_entities and line.strip() == '-----Relationships-----':
                break
            if in_entities and line.strip() and not line.startswith('human_readable_id'):
                # Parse the first entity's human_readable_id which is the community number
                parts = line.split(',', 1)
                if parts:
                    try:
                        community_id = int(parts[0])
                        return community_id
                    except ValueError:
                        continue

        return None
    except Exception as e:
        return None

def load_cached_report(cache_file: Path, communities_map: dict) -> dict | None:
    """Load a cached report and extract community ID."""
    try:
        with open(cache_file, 'r') as f:
            data = json.load(f)

        # Extract community ID from prompt
        if 'input' not in data or 'messages' not in data['input']:
            return None

        prompt_text = data['input']['messages'][0]['content']
        community_id = extract_community_id_from_prompt(prompt_text)

        if community_id is None:
            return None

        # Verify this community exists
        if community_id not in communities_map:
            return None

        # Extract the report from the LLM response
        if 'result' not in data or 'choices' not in data['result']:
            return None

        content = data['result']['choices'][0]['message']['content']
        report_json = json.loads(content)

        community_level = communities_map[community_id]

        return {
            'community': community_id,
            'level': community_level,
            'title': report_json.get('title', ''),
            'summary': report_json.get('summary', ''),
            'full_content': content,  # Keep the original JSON string
            'full_content_json': content,
            'rank': float(report_json.get('rating', 0.0)),
            'rating_explanation': report_json.get('rating_explanation', ''),
            'findings': report_json.get('findings', []),
        }
    except Exception as e:
        return None

def main():
    """Main function."""
    index_dir = Path('/mnt/fileintel/graphrag_index/6525aacb-55b1-4a88-aaaa-a4211d03beba')
    cache_dir = index_dir / 'cache' / 'community_reporting'
    output_dir = index_dir / 'output'

    print("="*80)
    print("EXTRACTING COMMUNITY REPORTS FROM CACHE")
    print("="*80)

    # Load communities to create a mapping
    print("\n1. Loading communities...")
    communities_df = pd.read_parquet(output_dir / 'communities.parquet')
    communities_map = dict(zip(communities_df['community'], communities_df['level']))

    print(f"   ✓ Total communities: {len(communities_df)}")
    print(f"   ✓ Level distribution: {communities_df['level'].value_counts().sort_index().to_dict()}")

    # Get all cache files
    cache_files = list(cache_dir.glob('chat_create_community_report_*'))
    print(f"\n2. Found {len(cache_files)} cached report files")

    # Process cache files
    print(f"\n3. Extracting reports from cache...")
    reports = []
    failed_count = 0
    community_ids_found = set()

    for cache_file in tqdm(cache_files, desc="Processing"):
        report = load_cached_report(cache_file, communities_map)
        if report:
            reports.append(report)
            community_ids_found.add(report['community'])
        else:
            failed_count += 1

    print(f"\n   ✓ Successfully extracted: {len(reports)} reports")
    print(f"   ✗ Failed to parse: {failed_count} files")
    print(f"   ✓ Unique communities: {len(community_ids_found)}")

    if not reports:
        print("\n   ERROR: No reports could be extracted!")
        return

    # Create DataFrame
    print(f"\n4. Creating community_reports DataFrame...")
    reports_df = pd.DataFrame(reports)

    # Check for duplicates
    duplicates = reports_df[reports_df.duplicated(subset=['community'], keep=False)]
    if not duplicates.empty:
        print(f"   ⚠ Warning: Found {len(duplicates)} duplicate community reports")
        # Keep first occurrence
        reports_df = reports_df.drop_duplicates(subset=['community'], keep='first')

    print(f"   ✓ Final report count: {len(reports_df)}")
    print(f"   ✓ Coverage: {len(reports_df) / len(communities_df) * 100:.1f}%")
    print(f"   ✓ Level distribution: {reports_df['level'].value_counts().sort_index().to_dict()}")

    # Save to parquet
    output_path = output_dir / 'community_reports_partial.parquet'
    reports_df.to_parquet(output_path)
    print(f"\n5. Saved partial results to: {output_path}")

    # Analysis
    print(f"\n" + "="*80)
    print("ANALYSIS: Missing Communities")
    print("="*80)

    missing_communities = communities_df[~communities_df['community'].isin(community_ids_found)]
    print(f"\nMissing {len(missing_communities)} communities ({len(missing_communities) / len(communities_df) * 100:.1f}%)")
    print(f"Missing by level: {missing_communities['level'].value_counts().sort_index().to_dict()}")

    print(f"\n" + "="*80)
    print("SUCCESS: Partial community_reports.parquet created!")
    print("="*80)
    print(f"\nYou can now test resume capability by:")
    print(f"1. Copying to: {output_dir / 'community_reports.parquet'}")
    print(f"2. Re-running the create_community_reports workflow")
    print(f"3. It will automatically resume and fill in the missing {len(missing_communities)} reports")

if __name__ == '__main__':
    main()
