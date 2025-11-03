# GraphRAG Hierarchy Optimizer

A standalone script to analyze and optimize GraphRAG pyramid hierarchy parameters without running full re-indexing.

## Features

- ✅ Analyze existing graph data from relationships.parquet
- ✅ Test single parameter combinations
- ✅ Test ranges of parameters to find optimal configuration
- ✅ Quality assessment and recommendations
- ✅ Export results to CSV for analysis
- ✅ Fast testing without full workflow execution

## Usage

### 1. Test Current Parameters

Analyze with current production settings:

```bash
python scripts/optimize_graphrag_hierarchy.py \
  --collection-id 6525aacb-55b1-4a88-aaaa-a4211d03beba
```

### 2. Test Specific Parameters

Test a specific resolution and multiplier:

```bash
python scripts/optimize_graphrag_hierarchy.py \
  --relationships-file /mnt/fileintel/graphrag_index/graphrag_indices/6525aacb-55b1-4a88-aaaa-a4211d03beba/output/relationships.parquet \
  --resolution 1.0 \
  --multiplier 15.0
```

### 3. Find Optimal Parameters (Recommended)

Test multiple parameter combinations to find the best one:

```bash
python scripts/optimize_graphrag_hierarchy.py \
  --relationships-file /mnt/fileintel/graphrag_index/graphrag_indices/6525aacb-55b1-4a88-aaaa-a4211d03beba/output/relationships.parquet \
  --test-range \
  --output-csv /tmp/hierarchy_optimization.csv
```

This will test:
- **Resolutions:** 0.5, 0.8, 1.0, 1.2, 1.5
- **Multipliers:** 5.0, 10.0, 15.0, 20.0, 25.0
- **Total:** 25 combinations

Results sorted by quality score (lower = better).

### 4. Quick Test from Docker Container

If running inside Docker:

```bash
# From inside celery-graphrag-gevent container
python /home/appuser/app/scripts/optimize_graphrag_hierarchy.py \
  --collection-id 6525aacb-55b1-4a88-aaaa-a4211d03beba \
  --test-range
```

## Output Example

```
================================================================================
Testing Parameters:
  Resolution: 1.0
  Base Resolution Multiplier: 15.0
  Effective Base Resolution: 15.0
  Total Nodes: 75637
================================================================================

================================================================================
HIERARCHY RESULTS:
================================================================================
Level 4 (BASE): 3421 communities, avg 22.1 entities/community
Level 3 (L3): 854 communities, avg 88.6 entities/community
Level 2 (L2): 187 communities, avg 404.5 entities/community
Level 1 (L1): 38 communities, avg 1990.4 entities/community
Level 0 (ROOT): 4 communities, avg 18909.2 entities/community
================================================================================

✅ Base community count is optimal (3421)
✅ Base community size is optimal (avg 22.1)
✅ Root community count is optimal (4)
✅ Pyramid depth is optimal (5 levels)

✅ All quality checks passed!

================================================================================
PARAMETER RECOMMENDATIONS:
================================================================================
Current Configuration:
  GRAPHRAG_LEIDEN_RESOLUTION=1.0
  GRAPHRAG_BASE_RESOLUTION_MULTIPLIER=15.0
  → Base Communities: 3421

Recommended Configuration:
  GRAPHRAG_LEIDEN_RESOLUTION=1.0
  GRAPHRAG_BASE_RESOLUTION_MULTIPLIER=15.0
  → Expected Base Communities: ~3781
================================================================================
```

## Quality Criteria

The script evaluates hierarchy quality based on:

### Base Level (Fine-grained)
- ✅ **Optimal:** 3,000-5,000 communities
- ✅ **Optimal Size:** 15-25 entities/community
- ❌ **Too Few:** < 3,000 communities (increase multiplier)
- ⚠️ **Too Many:** > 5,000 communities (decrease multiplier)

### Root Level (Coarse)
- ✅ **Optimal:** 1-5 communities
- ⚠️ **Too Many:** > 5 communities (issue with consolidation)

### Pyramid Depth
- ✅ **Optimal:** 5-10 levels
- ⚠️ **Too Few:** < 5 levels (too aggressive consolidation)
- ⚠️ **Too Many:** > 10 levels (too many levels)

## Parameter Tuning Guide

### If Base Has Too Few Communities:
**Problem:** Base has 300 communities instead of 3,000
**Solution:** Increase `GRAPHRAG_BASE_RESOLUTION_MULTIPLIER`

```bash
# Current: 5.0 → Try: 15.0
GRAPHRAG_BASE_RESOLUTION_MULTIPLIER=15.0
```

### If Base Communities Are Too Large:
**Problem:** Average 200 entities/community instead of 20
**Solution:** Increase `GRAPHRAG_BASE_RESOLUTION_MULTIPLIER`

```bash
# Increase multiplier for more fine-grained clustering
GRAPHRAG_BASE_RESOLUTION_MULTIPLIER=20.0
```

### If Too Many Root Communities:
**Problem:** Root has 10 communities instead of 3
**Solution:** Check consolidation logic (may need code adjustment)

### If Wrong Number of Levels:
**Problem:** Only 2-3 levels or 15+ levels
**Solution:** Adjust resolution or check graph connectivity

## Advanced Usage

### Custom Resolution/Multiplier Ranges

Edit the script to test custom ranges:

```python
# In test_parameter_range() function:
resolution_values = [0.8, 1.0, 1.2, 1.5, 2.0]
multiplier_values = [10.0, 12.0, 15.0, 18.0, 20.0]
```

### Reproducible Results

Use `--seed` for consistent clustering:

```bash
python scripts/optimize_graphrag_hierarchy.py \
  --collection-id <id> \
  --seed 42 \
  --test-range
```

### Export and Analyze

Export results for external analysis:

```bash
python scripts/optimize_graphrag_hierarchy.py \
  --collection-id <id> \
  --test-range \
  --output-csv results.csv

# Analyze with pandas, Excel, etc.
```

## Workflow Integration

### Typical Optimization Process:

1. **Run Optimizer:**
   ```bash
   python scripts/optimize_graphrag_hierarchy.py \
     --collection-id <id> \
     --test-range
   ```

2. **Review Results:**
   - Check quality scores
   - Review top recommendations
   - Choose optimal parameters

3. **Update Configuration:**
   ```bash
   # stack.env
   GRAPHRAG_BASE_RESOLUTION_MULTIPLIER=<optimal_value>
   ```

4. **Test Parameters:**
   ```bash
   # Verify with single test
   python scripts/optimize_graphrag_hierarchy.py \
     --collection-id <id> \
     --resolution 1.0 \
     --multiplier <optimal_value>
   ```

5. **Deploy:**
   ```bash
   # Rebuild and restart
   docker-compose -f docker-compose.prod.yml build
   docker-compose -f docker-compose.prod.yml up -d

   # Delete old clustering
   rm /mnt/.../output/create_final_communities.parquet
   rm /mnt/.../output/create_final_community_reports.parquet

   # Re-run indexing
   ```

## Performance

- **Single test:** ~10-30 seconds for 75K entities
- **Range test (25 combinations):** ~5-15 minutes for 75K entities
- Much faster than full re-indexing (which takes hours)

## Limitations

- Uses same algorithms as production pipeline
- Results may vary slightly due to Leiden randomness (use --seed for consistency)
- Does not test community report generation (only clustering)
- Requires relationships.parquet file (graph must already be extracted)

## Troubleshooting

### "Relationships file not found"
- Check collection ID is correct
- Ensure graph has been extracted (extract_graph workflow completed)
- Verify GRAPHRAG_INDEX_PATH environment variable

### "Import errors"
- Ensure you're in the correct Python environment
- Check that graphrag dependencies are installed
- Run from project root directory

### "Quality issues persist"
- Try wider parameter ranges
- Check graph connectivity (isolated nodes/communities)
- Review graph extraction settings

## Example Output for Range Testing

```
TOP 5 PARAMETER COMBINATIONS (by quality):
   resolution  multiplier  num_levels  base_communities  base_avg_size  root_communities  quality_score
0         1.0        15.0           5              3421           22.1                 4              0
1         1.2        15.0           6              4102           18.4                 3              0
2         1.0        20.0           6              4856           15.6                 5              0
3         0.8        15.0           5              2789           27.1                 3              1
4         1.5        12.0           5              3654           20.7                 6              1
```

Quality score of 0 = all quality checks passed!
