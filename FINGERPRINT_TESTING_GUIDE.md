# Content Fingerprinting - Testing & Deployment Guide

## Implementation Complete! ✅

All code changes have been implemented in small, testable increments.

---

## Step-by-Step Deployment

### Step 1: Run Database Migration

```bash
# Run the migration to add content_fingerprint column
python scripts/run_migration.py
```

**Verify migration:**
```bash
# Connect to PostgreSQL
psql -U $DB_USER -d $DB_NAME

# Check the new column exists
\d documents

# You should see:
# content_fingerprint | character varying(36) | | |
```

---

### Step 2: Restart Application

Restart your application to load the updated models and code:

```bash
# If using Docker
docker-compose restart api worker

# Or restart your application server
```

---

### Step 3: Test New Upload with Fingerprinting

**Test Case 1: Upload a new file**

```bash
# Upload a test PDF
curl -X POST http://localhost:8000/api/v2/collections/{collection_id}/documents \
  -F "file=@test.pdf"
```

**Expected behavior:**
- File is uploaded successfully
- `content_fingerprint` is calculated and stored
- Response includes the document_id

**Check database:**
```sql
SELECT id, original_filename, content_fingerprint, created_at
FROM documents
ORDER BY created_at DESC
LIMIT 1;
```

You should see a UUID in the `content_fingerprint` column.

---

**Test Case 2: Upload the same file again (duplicate detection)**

```bash
# Upload the SAME file again
curl -X POST http://localhost:8000/api/v2/collections/{collection_id}/documents \
  -F "file=@test.pdf"
```

**Expected behavior:**
- Duplicate is detected immediately
- Response shows `"duplicate": true`
- No new document is created
- Returns existing document ID

**Check logs:**
```
Duplicate detected: test.pdf (fingerprint: 8f3d2c1b-...) already exists as document abc123
```

---

**Test Case 3: Upload same file with different name**

```bash
# Rename file and upload
cp test.pdf renamed_test.pdf
curl -X POST http://localhost:8000/api/v2/collections/{collection_id}/documents \
  -F "file=@renamed_test.pdf"
```

**Expected behavior:**
- Still detected as duplicate (content-based, not name-based)
- Same fingerprint as original
- Returns existing document

---

### Step 4: Test MinerU Caching

**Test Case 4: Upload PDF and process with MinerU**

```bash
# Upload a PDF that will be processed with MinerU
curl -X POST http://localhost:8000/api/v2/collections/{collection_id}/documents \
  -F "file=@sample.pdf"

# Trigger processing (if not automatic)
# Check logs for:
# "Processing sample.pdf with self-hosted MinerU API"
# "Saving MinerU output to cache for fingerprint 8f3d2c1b-..."
```

**Check cache directory:**
```bash
ls -la /home/appuser/app/mineru_outputs/

# You should see a directory named with the fingerprint:
# 8f3d2c1b-4a5e-5678-9abc-def123456789/
#   8f3d2c1b-4a5e-5678-9abc-def123456789.md
#   8f3d2c1b-4a5e-5678-9abc-def123456789_content_list.json
#   ...
```

---

**Test Case 5: Reprocess same file (cache hit)**

```bash
# Delete document from collection (keeps cache)
curl -X DELETE http://localhost:8000/api/v2/documents/{document_id}

# Upload same file again
curl -X POST http://localhost:8000/api/v2/collections/{collection_id}/documents \
  -F "file=@sample.pdf"
```

**Expected behavior:**
- Same fingerprint is calculated
- MinerU cache is checked: **CACHE HIT!**
- No API call to MinerU
- Processing completes instantly

**Check logs:**
```
Loading MinerU output from cache for fingerprint 8f3d2c1b-...
```

**Performance:** Should be 10-100x faster than initial processing!

---

### Step 5: Backfill Existing Documents

**Dry run first (recommended):**
```bash
python scripts/backfill_content_fingerprints.py --dry-run
```

Review output to see what would be updated.

**Run actual backfill:**
```bash
python scripts/backfill_content_fingerprints.py
```

**Expected output:**
```
===========================================================
CONTENT FINGERPRINT BACKFILL SCRIPT
===========================================================
Fetching documents from database...
Found 150 total documents
Documents without fingerprint: 150

[1/150] Processing: report.pdf (abc-123)
  Reading file: /home/appuser/app/uploads/uuid123.pdf
  Calculated fingerprint: 8f3d2c1b-4a5e-5678-9abc-def123456789
  ✓ Updated document with fingerprint

...

===========================================================
BACKFILL COMPLETE
===========================================================
Total documents processed: 150
  ✓ Updated:               148
  ⊘ Skipped (no file path): 0
  ⊘ Skipped (file missing): 2
  ✗ Errors:                0
===========================================================
```

**Verify backfill:**
```sql
SELECT
  COUNT(*) as total,
  COUNT(content_fingerprint) as with_fingerprint,
  COUNT(*) - COUNT(content_fingerprint) as without_fingerprint
FROM documents;
```

All documents should now have fingerprints!

---

## Testing Checklist

- [ ] Database migration completed successfully
- [ ] New uploads receive `content_fingerprint`
- [ ] Duplicate detection works (same file, same collection)
- [ ] Duplicate detection works (same file, different collection)
- [ ] Duplicate detection works (same content, different filename)
- [ ] MinerU cache saves outputs with fingerprint-based naming
- [ ] MinerU cache hit works (instant processing)
- [ ] Backfill script completes without errors
- [ ] All existing documents have fingerprints

---

## Monitoring & Metrics

### Query 1: Deduplication Rate

```sql
-- How many unique files vs total uploads?
SELECT
  COUNT(DISTINCT content_fingerprint) as unique_files,
  COUNT(*) as total_documents,
  COUNT(*) - COUNT(DISTINCT content_fingerprint) as duplicates,
  ROUND(100.0 * (COUNT(*) - COUNT(DISTINCT content_fingerprint)) / COUNT(*), 2) as duplicate_percentage
FROM documents
WHERE content_fingerprint IS NOT NULL;
```

### Query 2: Most Duplicated Files

```sql
-- Which files are uploaded most often?
SELECT
  content_fingerprint,
  COUNT(*) as upload_count,
  STRING_AGG(DISTINCT original_filename, ', ') as filenames,
  STRING_AGG(DISTINCT collection_id, ', ') as collections
FROM documents
WHERE content_fingerprint IS NOT NULL
GROUP BY content_fingerprint
HAVING COUNT(*) > 1
ORDER BY upload_count DESC
LIMIT 10;
```

### Query 3: Cache Coverage

```bash
# Check how many unique fingerprints are cached
ls /home/appuser/app/mineru_outputs/ | wc -l

# Compare to unique documents in DB
psql -U $DB_USER -d $DB_NAME -c "SELECT COUNT(DISTINCT content_fingerprint) FROM documents WHERE content_fingerprint IS NOT NULL;"
```

---

## Troubleshooting

### Issue: Migration fails

**Check:**
```bash
# Verify database connection
psql -U $DB_USER -d $DB_NAME -c "SELECT 1;"

# Check if column already exists
psql -U $DB_USER -d $DB_NAME -c "\d documents"
```

**Solution:**
If column exists, migration already ran. Skip to next step.

---

### Issue: Fingerprint not being calculated

**Check logs:**
```bash
docker logs fileintel-api | grep fingerprint
```

**Look for:**
- Import errors with `fileintel.utils.fingerprint`
- Errors in `generate_content_fingerprint()`

**Verify file exists:**
```python
from fileintel.utils.fingerprint import generate_content_fingerprint
print(generate_content_fingerprint(b"test content"))
# Should print a UUID like: "f3e8a1b2-c3d4-5e6f-7a8b-9c0d1e2f3a4b"
```

---

### Issue: Cache not being used

**Check:**
1. Is `content_fingerprint` populated in database?
   ```sql
   SELECT content_fingerprint FROM documents WHERE id = '{document_id}';
   ```

2. Does cache directory exist?
   ```bash
   ls /home/appuser/app/mineru_outputs/{fingerprint}/
   ```

3. Check processor logs:
   ```bash
   docker logs fileintel-worker | grep "cache"
   ```

**Look for:**
- "Loading MinerU output from cache" (cache hit)
- "Saving MinerU output to cache" (cache save)
- "MinerU cache MISS" (cache miss)

---

### Issue: Backfill script fails

**Common causes:**
1. **File not found**: Document metadata has wrong path
   - Skipped automatically, check logs

2. **Permission denied**: Script can't read files
   - Run with appropriate permissions
   - Check file ownership

3. **Database connection**: Can't connect to DB
   - Verify `.env` file
   - Check `DB_*` environment variables

---

## Performance Benchmarks

### Expected Performance Improvements

| Scenario | Before | After | Improvement |
|----------|--------|-------|-------------|
| **Duplicate Upload** | ~2-5s (processing) | ~100ms (instant) | **20-50x faster** |
| **MinerU Reprocessing** | ~30-120s (API call) | ~1-2s (cache) | **30-120x faster** |
| **Storage (100 duplicates)** | 500MB total | 5MB unique | **100x reduction** |
| **MinerU API Costs** | $X per document | $X per unique | **Saves on duplicates** |

---

## Next Steps

1. **Monitor deduplication rate**
   - Track how many duplicates are caught
   - Measure storage savings

2. **Monitor cache hit rate**
   - Track `cache_hit / (cache_hit + cache_miss)`
   - Aim for >50% hit rate in production

3. **Set cache retention policy** (future enhancement)
   - Currently: cache never expires
   - Consider: LRU eviction or TTL

4. **Add cache warming** (future enhancement)
   - Pre-populate cache with common documents
   - Background refresh for frequently accessed files

5. **Cross-instance caching** (future enhancement)
   - Share cache across multiple instances
   - Distributed cache with Redis/S3

---

## Rollback Plan

If issues arise, you can roll back:

### Rollback Database
```bash
# Downgrade migration
alembic downgrade -1
```

This will:
- Remove `content_fingerprint` column
- Remove index on fingerprint

### Rollback Code
```bash
git revert {commit_hash}
```

Or manually:
- Comment out fingerprint-related code
- Restart application

**Note:** Fingerprint system is **non-breaking** - old code will work even with new column present (it's nullable).

---

## Success Criteria

✅ Implementation is successful when:

1. All new uploads receive fingerprints
2. Duplicate detection works across collections
3. MinerU cache hit rate > 0% (any cache hits indicate success)
4. Backfill completes for existing documents
5. No errors in application logs
6. Performance improvements are measurable

---

**Questions or Issues?**

Check the detailed implementation plan: `FINGERPRINT_IMPLEMENTATION_PLAN.md`

Or create an issue with:
- Error messages from logs
- Database query results
- Steps to reproduce
