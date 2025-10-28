# Content Fingerprint Implementation Plan

## Overview

Implement content-based fingerprinting (UUID v5) to enable:
- **Deterministic IDs**: Same file content → same UUID (always)
- **Global deduplication**: Detect duplicates across all collections
- **MinerU output caching**: Reuse expensive processing results
- **Storage optimization**: Avoid duplicate file storage

## Implementation Summary

**Total Files to Modify:** 7 core files + 1 database migration
**New Files Created:** 2
**Estimated Complexity:** Medium (mostly additive changes, non-breaking)

---

## 1. New Files (Already Created) ✅

### 1.1 `src/fileintel/utils/fingerprint.py`
**Status:** ✅ Created
**Purpose:** Content-based fingerprinting utilities

**Key Functions:**
- `generate_content_fingerprint(content)` - Generate UUID v5 from bytes/Path
- `generate_fingerprint_from_hash(content_hash)` - Generate from existing SHA256
- `verify_fingerprint(content, expected)` - Integrity verification

**Example:**
```python
from fileintel.utils.fingerprint import generate_content_fingerprint

content = Path("report.pdf").read_bytes()
fp = generate_content_fingerprint(content)
# fp = "8f3d2c1b-4a5e-5678-9abc-def123456789"

# Same content, different filename
Path("report.pdf").rename("summary.pdf")
fp2 = generate_content_fingerprint(Path("summary.pdf"))
# fp2 = "8f3d2c1b-4a5e-5678-9abc-def123456789" (identical!)
```

### 1.2 `src/fileintel/document_processing/mineru_cache.py`
**Status:** ✅ Created
**Purpose:** MinerU output caching system

**Key Methods:**
- `has_cache(fingerprint)` - Check if cached output exists
- `load_cached_output(fingerprint)` - Load from cache
- `save_to_cache(fingerprint, results)` - Save to cache
- `get_cache_stats()` - Cache statistics

**Cache Structure:**
```
/home/appuser/app/mineru_outputs/
  8f3d2c1b-4a5e-5678-9abc-def123456789/    ← Fingerprint as directory
    8f3d2c1b-4a5e-5678-9abc-def123456789.md
    8f3d2c1b-4a5e-5678-9abc-def123456789_content_list.json
    8f3d2c1b-4a5e-5678-9abc-def123456789_model.json
    images/...
```

---

## 2. Database Changes

### 2.1 Add `content_fingerprint` Column to Documents Table

**File:** Database migration script
**Changes:** Add new column + index

```sql
-- Migration: Add content_fingerprint column
ALTER TABLE documents
ADD COLUMN content_fingerprint VARCHAR(36);

-- Add index for fast fingerprint lookups
CREATE INDEX idx_documents_content_fingerprint
ON documents(content_fingerprint);

-- Optional: Add uniqueness constraint for strict global deduplication
-- (Only if you want to prevent same content in different collections)
-- ALTER TABLE documents
-- ADD CONSTRAINT unique_content_fingerprint UNIQUE (content_fingerprint);
```

**Impact:** Non-breaking change (column is nullable)

**Migration Script Location:** Create new file
`src/fileintel/migrations/add_content_fingerprint.py` or use Alembic

---

## 3. Core File Modifications

### 3.1 `src/fileintel/storage/models.py`

**Location:** Line ~69-97 (Document class)
**Changes:** Add `content_fingerprint` field to Document model

```python
class Document(Base):
    __tablename__ = "documents"
    id = Column(String, primary_key=True)
    collection_id = Column(
        String, ForeignKey("collections.id"), nullable=False, index=True
    )
    filename = Column(String, nullable=False)
    original_filename = Column(String, nullable=False)
    content_hash = Column(String, nullable=False, index=True)

    # NEW: Content-based fingerprint (UUID v5)
    content_fingerprint = Column(
        String(36),
        nullable=True,  # Nullable for backward compatibility
        index=True,  # Index for fast lookups
        # unique=True  # Uncomment for strict global deduplication
    )

    file_size = Column(Integer, nullable=False)
    mime_type = Column(String, nullable=False)
    document_metadata = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # ... relationships remain unchanged
```

**Lines to modify:** ~69-97
**Estimated changes:** +7 lines

---

### 3.2 `src/fileintel/storage/document_storage.py`

**Changes:**
1. Add `content_fingerprint` parameter to `create_document()` (line ~195)
2. Add `get_document_by_fingerprint()` method (new)
3. Add `get_document_by_fingerprint_global()` method (new)

#### Change 3.2.1: Modify `create_document()` signature

**Location:** Line 195-215
**Before:**
```python
def create_document(
    self,
    filename: str,
    content_hash: str,
    file_size: int,
    mime_type: str,
    collection_id: str,
    original_filename: str = None,
    file_path: str = None,
    metadata: dict = None,
) -> Document:
    document_id = str(uuid.uuid4())  # Random UUID
    # ...
```

**After:**
```python
def create_document(
    self,
    filename: str,
    content_hash: str,
    file_size: int,
    mime_type: str,
    collection_id: str,
    original_filename: str = None,
    file_path: str = None,
    metadata: dict = None,
    content_fingerprint: str = None,  # NEW parameter
) -> Document:
    document_id = str(uuid.uuid4())  # Keep random UUID for backward compat
    # ...
    document = Document(
        id=document_id,
        filename=filename,
        content_hash=content_hash,
        content_fingerprint=content_fingerprint,  # NEW field
        file_size=file_size,
        mime_type=mime_type,
        collection_id=collection_id,
        original_filename=original_filename,
        document_metadata=doc_metadata,
    )
    # ...
```

**Estimated changes:** +2 lines (parameter + field assignment)

#### Change 3.2.2: Add `get_document_by_fingerprint()` method

**Location:** After line 259 (after `get_document_by_filename_and_collection`)
**Add:**
```python
def get_document_by_fingerprint(
    self, fingerprint: str, collection_id: str = None
) -> Document:
    """
    Get document by content fingerprint.

    Args:
        fingerprint: Content fingerprint UUID
        collection_id: Optional collection filter (for scoped lookup)

    Returns:
        Document if found, None otherwise
    """
    query = self.db.query(Document).filter(
        Document.content_fingerprint == fingerprint
    )

    if collection_id:
        # Scoped to collection
        query = query.filter(Document.collection_id == collection_id)
    else:
        # Global lookup (any collection)
        pass

    return query.first()

def get_all_documents_by_fingerprint(self, fingerprint: str) -> List[Document]:
    """
    Get all documents with this fingerprint across all collections.

    Useful for:
    - Checking if content exists anywhere in system
    - Finding duplicate uploads in different collections

    Args:
        fingerprint: Content fingerprint UUID

    Returns:
        List of Documents (may be empty)
    """
    return (
        self.db.query(Document)
        .filter(Document.content_fingerprint == fingerprint)
        .all()
    )
```

**Estimated changes:** +40 lines (2 new methods)

---

### 3.3 `src/fileintel/api/routes/collections_v2.py`

**Location:** `upload_document_to_collection()` function (line 176-279)
**Changes:**
1. Calculate fingerprint after reading file content
2. Check for existing document by fingerprint (global or scoped)
3. Pass fingerprint to `create_document()`

**Before (line 226-246):**
```python
content_hash = hashlib.sha256(content).hexdigest()
file_size = len(content)
mime_type = file.content_type or "application/octet-stream"

# Check for duplicate in this collection
existing_document = storage.get_document_by_hash_and_collection(
    content_hash, collection.id
)

if existing_document:
    logger.info(f"Duplicate detected: {file.filename}")
    document = existing_document
    duplicate_detected = True
else:
    document = storage.create_document(
        filename=unique_filename,
        content_hash=content_hash,
        # ...
    )
```

**After:**
```python
content_hash = hashlib.sha256(content).hexdigest()
file_size = len(content)
mime_type = file.content_type or "application/octet-stream"

# NEW: Calculate content fingerprint
from fileintel.utils.fingerprint import generate_content_fingerprint
content_fingerprint = generate_content_fingerprint(content)

logger.debug(f"File fingerprint: {content_fingerprint}")

# Check for duplicate by fingerprint (globally or per-collection)
# Option A: Global deduplication (any collection)
existing_document = storage.get_document_by_fingerprint(content_fingerprint)

# Option B: Per-collection deduplication (backward compatible)
# existing_document = storage.get_document_by_fingerprint(
#     content_fingerprint, collection_id=collection.id
# )

if existing_document:
    logger.info(
        f"Duplicate detected: {file.filename} (fingerprint: {content_fingerprint}) "
        f"already exists as document {existing_document.id}"
    )
    document = existing_document
    duplicate_detected = True
else:
    document = storage.create_document(
        filename=unique_filename,
        content_hash=content_hash,
        content_fingerprint=content_fingerprint,  # NEW parameter
        file_size=file_size,
        mime_type=mime_type,
        collection_id=collection.id,
        original_filename=file.filename,
        metadata={
            "uploaded_via": "api_v2",
            "original_filename": file.filename,
            "file_path": str(file_path),
        },
    )
    duplicate_detected = False
```

**Estimated changes:** +10 lines (import, fingerprint calculation, pass to create_document)

---

### 3.4 `src/fileintel/document_processing/processors/mineru_selfhosted.py`

**Location:** `read()` method (line 96-165)
**Changes:** Integrate MinerU caching

**Key Integration Points:**

1. **Initialize cache** (line ~102)
2. **Check cache before API call** (line ~106-110)
3. **Save to cache after processing** (line ~113-115)

**Before (line 96-157):**
```python
def read(self, file_path: Path, adapter: logging.LoggerAdapter = None) -> Tuple[List[DocumentElement], Dict[str, Any]]:
    log = adapter or logger
    validate_file_for_processing(file_path, ".pdf")

    try:
        # Process with self-hosted MinerU API
        log.debug(f"Processing {file_path.name} with self-hosted MinerU API")

        mineru_results = self._process_with_selfhosted_api(file_path, log)

        # Save outputs if enabled (for debugging)
        self._save_mineru_outputs(mineru_results, file_path, log)

        # Extract data from response
        markdown_content, json_data = self._extract_results_from_response(mineru_results)

        # Create elements...
```

**After:**
```python
def read(
    self,
    file_path: Path,
    adapter: logging.LoggerAdapter = None,
    content_fingerprint: str = None  # NEW parameter
) -> Tuple[List[DocumentElement], Dict[str, Any]]:
    log = adapter or logger
    validate_file_for_processing(file_path, ".pdf")

    try:
        # NEW: Initialize MinerU cache
        from fileintel.document_processing.mineru_cache import MinerUCache
        mineru_config = self.config.document_processing.mineru
        cache = MinerUCache(mineru_config.output_directory)

        # NEW: Check cache first (if fingerprint provided)
        mineru_results = None
        if content_fingerprint and cache.has_cache(content_fingerprint):
            log.info(f"Loading MinerU output from cache for {content_fingerprint}")
            mineru_results = cache.load_cached_output(content_fingerprint)

        # Process with API if no cache hit
        if not mineru_results:
            log.debug(f"Processing {file_path.name} with self-hosted MinerU API")
            mineru_results = self._process_with_selfhosted_api(file_path, log)

        # Save outputs (for debugging and future cache hits)
        if content_fingerprint:
            # Use fingerprint-based naming
            self._save_mineru_outputs_with_fingerprint(
                mineru_results, content_fingerprint, log, cache
            )
        else:
            # Fallback: use filename-based naming (backward compatible)
            self._save_mineru_outputs(mineru_results, file_path, log)

        # Extract data from response...
```

**Estimated changes:** +20 lines

**Additional Method:** Update `_save_mineru_outputs()` to use cache when fingerprint available

```python
def _save_mineru_outputs_with_fingerprint(
    self,
    mineru_results: Dict[str, Any],
    fingerprint: str,
    log,
    cache: MinerUCache
) -> None:
    """Save MinerU outputs using fingerprint-based naming."""
    if not self.config.document_processing.mineru.save_outputs:
        return

    # Extract markdown and JSON data
    markdown_content, json_data = self._extract_results_from_response(mineru_results)

    # Save to cache (handles directory structure and naming)
    cache.save_to_cache(fingerprint, mineru_results, markdown_content, json_data)
```

**Estimated changes:** +15 lines (new method)

---

### 3.5 `src/fileintel/document_processing/processors/mineru_commercial.py`

**Location:** Similar changes to selfhosted version
**Changes:** Same integration pattern as 3.4
**Estimated changes:** +35 lines (same as selfhosted)

---

### 3.6 `src/fileintel/tasks/document_tasks.py`

**Location:** `process_document()` task and `read_document_with_elements()` (line ~300-500)
**Changes:** Pass `content_fingerprint` through processing pipeline

**Key Points:**
1. Retrieve fingerprint from document record
2. Pass to processor's `read()` method

**Modification in `read_document_with_elements()`:**

**Before (approximately line 400):**
```python
def read_document_with_elements(
    document_path: Path,
    processor_type: str,
    config,
    adapter
) -> Tuple[List[TextElement], Dict[str, Any]]:
    # ... processor selection logic ...

    if processor_type == 'mineru_selfhosted':
        processor = MinerUSelfHostedProcessor(config)
        elements, metadata = processor.read(document_path, adapter)
    # ...
```

**After:**
```python
def read_document_with_elements(
    document_path: Path,
    processor_type: str,
    config,
    adapter,
    content_fingerprint: str = None  # NEW parameter
) -> Tuple[List[TextElement], Dict[str, Any]]:
    # ... processor selection logic ...

    if processor_type == 'mineru_selfhosted':
        processor = MinerUSelfHostedProcessor(config)
        # Pass fingerprint to enable caching
        elements, metadata = processor.read(
            document_path,
            adapter,
            content_fingerprint=content_fingerprint
        )
    # ...
```

**Modification in `process_document()` task:**

**Before:**
```python
@app.task(base=BaseFileIntelTask, bind=True, ...)
def process_document(self, document_id: str, collection_id: str) -> Dict:
    # ... setup ...

    # Read document
    elements, metadata = read_document_with_elements(
        file_path, processor_type, config, adapter
    )
```

**After:**
```python
@app.task(base=BaseFileIntelTask, bind=True, ...)
def process_document(self, document_id: str, collection_id: str) -> Dict:
    # ... setup ...

    # Get document to retrieve fingerprint
    document = storage.get_document(document_id)
    content_fingerprint = document.content_fingerprint if document else None

    # Read document (with caching if fingerprint available)
    elements, metadata = read_document_with_elements(
        file_path,
        processor_type,
        config,
        adapter,
        content_fingerprint=content_fingerprint  # NEW parameter
    )
```

**Estimated changes:** +10 lines

---

### 3.7 `src/fileintel/core/config.py` (Optional)

**Location:** MinerU config section (line ~313-320)
**Changes:** Add caching configuration flags (optional)

```python
class MinerUConfig(BaseModel):
    # ... existing fields ...
    save_outputs: bool = Field(default=False)
    output_directory: str = Field(default="/home/appuser/app/mineru_outputs")

    # NEW: Caching configuration
    enable_cache: bool = Field(
        default=True,
        description="Enable MinerU output caching based on content fingerprints"
    )
    cache_ttl_days: int = Field(
        default=0,
        description="Cache TTL in days (0 = never expire, useful for debugging)"
    )
```

**Estimated changes:** +6 lines (optional)

---

## 4. Migration Strategy

### 4.1 Backward Compatibility

**Design Principle:** Non-breaking changes only

1. **Nullable field**: `content_fingerprint` is nullable (existing documents have NULL)
2. **Fallback behavior**: If fingerprint is NULL, fall back to hash-based deduplication
3. **Gradual adoption**: New uploads get fingerprints; old documents work unchanged

### 4.2 Backfilling Existing Documents

**Script: `scripts/backfill_fingerprints.py`**

```python
"""
Backfill content_fingerprint for existing documents.

Reads files from disk, calculates fingerprints, updates database.
Safe to run multiple times (idempotent).
"""

from pathlib import Path
from fileintel.storage.document_storage import DocumentStorage
from fileintel.utils.fingerprint import generate_content_fingerprint

def backfill_fingerprints():
    storage = DocumentStorage()
    documents = storage.get_all_documents()  # Implement if needed

    updated = 0
    skipped = 0
    errors = 0

    for doc in documents:
        # Skip if already has fingerprint
        if doc.content_fingerprint:
            skipped += 1
            continue

        # Read file from disk
        file_path = Path(doc.document_metadata.get('file_path'))
        if not file_path.exists():
            print(f"File not found for document {doc.id}: {file_path}")
            errors += 1
            continue

        try:
            # Calculate fingerprint
            fingerprint = generate_content_fingerprint(file_path)

            # Update document
            doc.content_fingerprint = fingerprint
            storage.base._safe_commit()

            updated += 1
            print(f"Updated document {doc.id}: {fingerprint}")

        except Exception as e:
            print(f"Error processing document {doc.id}: {e}")
            errors += 1

    print(f"\nBackfill complete:")
    print(f"  Updated: {updated}")
    print(f"  Skipped: {skipped}")
    print(f"  Errors: {errors}")

if __name__ == "__main__":
    backfill_fingerprints()
```

### 4.3 Testing Strategy

**Test Cases:**

1. **Upload same file twice** → Same fingerprint → Duplicate detected
2. **Rename and upload** → Same fingerprint → Reuses existing
3. **MinerU cache hit** → No API call → Fast processing
4. **MinerU cache miss** → API call → Cache saved for next time
5. **Different collections** → Option A (global): dedupe, Option B (scoped): separate documents
6. **Backward compat** → Documents without fingerprints still work
7. **Backfill script** → Existing documents get fingerprints

---

## 5. Implementation Checklist

### Phase 1: Core Infrastructure ✅
- [✅] Create `src/fileintel/utils/fingerprint.py`
- [✅] Create `src/fileintel/document_processing/mineru_cache.py`

### Phase 2: Database
- [ ] Create database migration: add `content_fingerprint` column
- [ ] Run migration on development database
- [ ] Verify column exists with `\d documents` in psql

### Phase 3: Storage Layer
- [ ] Modify `src/fileintel/storage/models.py` (add field to Document class)
- [ ] Modify `src/fileintel/storage/document_storage.py`:
  - [ ] Update `create_document()` signature
  - [ ] Add `get_document_by_fingerprint()` method
  - [ ] Add `get_all_documents_by_fingerprint()` method

### Phase 4: API Layer
- [ ] Modify `src/fileintel/api/routes/collections_v2.py`:
  - [ ] Import fingerprint utility
  - [ ] Calculate fingerprint on upload
  - [ ] Check for duplicates by fingerprint
  - [ ] Pass fingerprint to `create_document()`

### Phase 5: Processing Layer
- [ ] Modify `src/fileintel/tasks/document_tasks.py`:
  - [ ] Update `read_document_with_elements()` signature
  - [ ] Pass fingerprint from document to processor
- [ ] Modify `src/fileintel/document_processing/processors/mineru_selfhosted.py`:
  - [ ] Update `read()` signature
  - [ ] Check cache before API call
  - [ ] Save to cache after processing
- [ ] Modify `src/fileintel/document_processing/processors/mineru_commercial.py`:
  - [ ] Same changes as selfhosted

### Phase 6: Configuration (Optional)
- [ ] Update `src/fileintel/core/config.py` with cache settings

### Phase 7: Testing
- [ ] Unit tests for fingerprint generation
- [ ] Integration test: upload → dedupe → cache
- [ ] Test MinerU cache hit/miss scenarios
- [ ] Test backward compatibility (NULL fingerprints)

### Phase 8: Migration
- [ ] Create backfill script
- [ ] Run backfill on existing documents
- [ ] Verify all documents have fingerprints

---

## 6. Summary Statistics

| Category | Count | Complexity |
|----------|-------|------------|
| **New Files** | 2 | Low |
| **Modified Files** | 7 | Medium |
| **Database Changes** | 1 | Low |
| **New Methods** | 5 | Low |
| **Modified Methods** | 6 | Medium |
| **Total Lines Added** | ~150 | - |
| **Breaking Changes** | 0 | None |

---

## 7. Benefits After Implementation

1. **Deterministic IDs**
   - Same file → Same fingerprint (always)
   - Reproducible across systems

2. **Global Deduplication**
   - Detect duplicates across all collections
   - Save storage space

3. **MinerU Caching**
   - Reuse expensive processing results
   - Avoid redundant API calls
   - Faster reprocessing (different settings)

4. **Cost Savings**
   - No duplicate MinerU API calls for same content
   - Reduced storage for duplicate files

5. **Improved UX**
   - Instant "duplicate detected" feedback
   - Fast reprocessing from cache

---

## 8. Rollout Plan

### Week 1: Infrastructure
- Implement database migration
- Update storage models and methods
- Add unit tests

### Week 2: Integration
- Update API endpoints
- Integrate with MinerU processors
- Add integration tests

### Week 3: Testing
- QA testing in development
- Performance testing (cache hit rates)
- Load testing

### Week 4: Deployment
- Deploy to staging
- Run backfill script
- Monitor for issues
- Deploy to production

---

## 9. Monitoring & Metrics

**Key Metrics to Track:**

1. **Deduplication Rate**
   - `uploads_with_existing_fingerprint / total_uploads`

2. **MinerU Cache Hit Rate**
   - `cache_hits / (cache_hits + cache_misses)`

3. **Storage Savings**
   - `(duplicate_files * avg_file_size) = bytes_saved`

4. **API Cost Savings**
   - `(duplicate_mineru_calls * cost_per_call) = $ saved`

5. **Processing Time**
   - `avg_time_with_cache_hit vs avg_time_cache_miss`

---

## 10. Future Enhancements

1. **Cache Expiration**
   - Implement TTL for cache entries
   - LRU eviction policy

2. **Cache Warming**
   - Pre-populate cache with common documents
   - Background cache refresh

3. **Cross-System Fingerprinting**
   - Share fingerprint database across instances
   - Distributed cache

4. **Advanced Deduplication**
   - Near-duplicate detection (fuzzy hashing)
   - Version tracking (same file, minor changes)

---

**Implementation Owner:** TBD
**Estimated Effort:** 2-4 weeks
**Priority:** High (significant cost/performance benefits)
