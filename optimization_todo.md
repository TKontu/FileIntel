# Two-Tier Chunking Performance Optimization Plan

## 📊 **Current System Analysis**

### **Performance Baseline (100MB PDF)**
- **Processing Time**: ~45 seconds
- **Memory Peak**: ~500MB (within 1GB allocation)
- **CPU Usage**: ~25% (single-core utilization)
- **Bottlenecks**: CPU-bound, not memory-bound

### **Target Performance Goals**
- **Processing Time**: <15 seconds (70% improvement)
- **Memory Peak**: <400MB (20% reduction)
- **CPU Usage**: 60-80% (multi-core utilization)
- **Scalability**: Handle up to 500MB PDFs efficiently

---

## 🎯 **Optimization Phases**

### **Phase 1: Quick Wins (1-2 hours implementation)**
Focus on immediate CPU performance improvements with minimal code changes.

#### **1.1 Parallel Sentence Processing**
**Priority**: 🔴 **HIGH**
**Impact**: 4x faster sentence processing
**Complexity**: Medium
**Files**: `src/fileintel/document_processing/chunking.py`

**Changes**:
```python
# Add new method
def _process_sentences_parallel(self, text_chunks: List[str]) -> List[Dict[str, Any]]:
    """Process multiple text sections in parallel for large documents."""
    from concurrent.futures import ThreadPoolExecutor
    import multiprocessing

    with ThreadPoolExecutor(max_workers=min(4, multiprocessing.cpu_count())) as executor:
        sentence_futures = []
        sentence_id_offset = 0

        for chunk_idx, chunk in enumerate(text_chunks):
            future = executor.submit(self._split_into_sentence_objects, chunk, sentence_id_offset)
            sentence_futures.append(future)
            # Estimate sentence count for next offset (approximate)
            estimated_sentences = len(chunk) // 100  # ~100 chars per sentence
            sentence_id_offset += estimated_sentences

        # Collect results and fix ID sequences
        all_sentences = []
        actual_offset = 0
        for future in sentence_futures:
            chunk_sentences = future.result()
            # Fix sentence IDs to maintain global sequence
            for sentence in chunk_sentences:
                sentence['id'] = actual_offset
                actual_offset += 1
            all_sentences.extend(chunk_sentences)

        return all_sentences

# Add text splitting helper
def _split_text_for_parallel_processing(self, text: str, chunk_size: int = 1_000_000) -> List[str]:
    """Split text into processable chunks at sentence boundaries."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    current_pos = 0

    while current_pos < len(text):
        end_pos = min(current_pos + chunk_size, len(text))

        # Find last sentence boundary before chunk_size
        if end_pos < len(text):
            # Look for sentence ending within last 10% of chunk
            search_start = max(current_pos, end_pos - chunk_size // 10)

            # Find last occurrence of sentence endings
            for pattern in ['. ', '! ', '? ', '.\n', '!\n', '?\n']:
                last_sentence = text.rfind(pattern, search_start, end_pos)
                if last_sentence != -1:
                    end_pos = last_sentence + len(pattern)
                    break

        chunks.append(text[current_pos:end_pos])
        current_pos = end_pos

    return chunks

# Modify process_two_tier_chunking method
def process_two_tier_chunking(self, text: str, page_mappings: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Complete two-tier chunking process with performance optimizations."""

    # Phase 1: Process sentences (optimized for large documents)
    if len(text) > 5_000_000:  # 5MB threshold for parallel processing
        text_chunks = self._split_text_for_parallel_processing(text)
        sentence_objects = self._process_sentences_parallel(text_chunks)

        # Add page mappings to sentences
        if page_mappings:
            for sentence in sentence_objects:
                sentence['page_info'] = self._find_sentence_pages(sentence, text, page_mappings)
    else:
        # Use existing single-threaded approach for smaller documents
        sentence_data = self.process_document_sentences(text, page_mappings)
        sentence_objects = sentence_data['sentences']

    # Phase 2 & 3: Continue with existing vector/graph chunk creation
    vector_chunks = self.create_vector_chunks_from_sentences(
        sentence_objects,
        target_tokens=300,
        max_tokens=self.vector_max_tokens,
        overlap_sentences=3
    )

    graph_chunks = self.create_graph_chunks_from_vector_chunks(
        vector_chunks,
        chunks_per_graph=5,
        overlap_chunks=2,
        target_tokens=1500
    )

    return {
        'sentence_data': {'sentences': sentence_objects, 'total_sentences': len(sentence_objects)},
        'vector_chunks': vector_chunks,
        'graph_chunks': graph_chunks,
        'optimization_stats': {
            'parallel_processing_used': len(text) > 5_000_000,
            'text_chunks_processed': len(text_chunks) if len(text) > 5_000_000 else 1
        }
    }
```

**Testing**:
- [ ] Benchmark with 10MB test document
- [ ] Verify sentence ID uniqueness
- [ ] Confirm page mapping accuracy
- [ ] Measure performance improvement

#### **1.2 Batch Token Counting**
**Priority**: 🟡 **MEDIUM**
**Impact**: 3x faster token counting
**Complexity**: Low
**Files**: `src/fileintel/document_processing/chunking.py`

**Changes**:
```python
def _batch_token_counting(self, texts: List[str]) -> List[int]:
    """Count tokens for multiple texts in batch for better performance."""
    if not texts:
        return []

    # Use BGE tokenizer batch processing if available
    if self.bge_tokenizer:
        try:
            # Process in batches to avoid memory issues
            batch_size = 1000
            all_counts = []

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                encodings = self.bge_tokenizer(
                    batch_texts,
                    return_tensors=None,
                    add_special_tokens=True,
                    truncation=False,
                    padding=False
                )
                batch_counts = [len(encoding) for encoding in encodings['input_ids']]
                all_counts.extend(batch_counts)

            return all_counts
        except Exception as e:
            logger.warning(f"Batch tokenization failed, falling back to individual: {e}")

    # Fallback to individual processing
    return [self._count_tokens(text) for text in texts]

# Modify sentence processing to use batch counting
def _split_into_sentence_objects(self, text: str, start_id: int = 0) -> List[Dict[str, Any]]:
    """Split text into sentence objects with batch token counting."""
    sentences = self._split_into_sentences(text)

    # Batch count tokens for all sentences at once
    sentence_texts = [s for s in sentences if s.strip()]
    token_counts = self._batch_token_counting(sentence_texts)

    sentence_objects = []
    for i, (sentence_text, token_count) in enumerate(zip(sentence_texts, token_counts)):
        sentence_obj = {
            'id': start_id + i,
            'text': sentence_text,
            'token_count': token_count,
            'char_count': len(sentence_text)
        }
        sentence_objects.append(sentence_obj)

    return sentence_objects
```

**Testing**:
- [ ] Compare batch vs individual token counting speed
- [ ] Verify token count accuracy
- [ ] Test with different batch sizes

---

### **Phase 2: Algorithm Optimizations (4-6 hours implementation)**
Focus on reducing computational complexity and memory allocations.

#### **2.1 Lazy Text Reconstruction**
**Priority**: 🟡 **MEDIUM**
**Impact**: 30% memory reduction, faster chunk creation
**Complexity**: Medium
**Files**: `src/fileintel/document_processing/chunking.py`

**Changes**:
```python
class LazyChunkText:
    """Memory-efficient text reconstruction for chunks."""

    def __init__(self, sentence_refs: List[int], sentence_lookup: Dict[int, Dict]):
        self.sentence_refs = sentence_refs
        self.sentence_lookup = sentence_lookup
        self._cached_text = None
        self._cached_length = None

    @property
    def text(self) -> str:
        if self._cached_text is None:
            texts = [self.sentence_lookup[sid]['text'] for sid in self.sentence_refs]
            self._cached_text = ' '.join(texts)
        return self._cached_text

    @property
    def length(self) -> int:
        if self._cached_length is None:
            self._cached_length = len(self.text)
        return self._cached_length

    def __str__(self) -> str:
        return self.text

    def __len__(self) -> int:
        return self.length

# Modify chunk creation to use lazy text
def create_graph_chunks_from_vector_chunks(self, vector_chunks: List[Dict[str, Any]], ...):
    """Create deduplicated graph chunks with lazy text reconstruction."""

    # Create global sentence lookup for this document
    global_sentence_lookup = {}
    for chunk in vector_chunks:
        for sentence in chunk.get('sentences', []):
            global_sentence_lookup[sentence['id']] = sentence

    graph_chunks = []
    # ... existing logic for collecting sentence IDs ...

    # Create chunk with lazy text
    graph_chunk = {
        'id': f'graph_{len(graph_chunks)}',
        'type': 'graph',
        'vector_chunk_ids': [c['id'] for c in chunk_group],
        'unique_sentence_ids': unique_sentence_ids,
        'text': LazyChunkText(unique_sentence_ids, global_sentence_lookup),  # Lazy evaluation
        'sentence_count': len(unique_sentence_ids),
        'token_count': total_tokens,
        # ... rest of chunk data
    }
```

#### **2.2 Optimized Page Aggregation**
**Priority**: 🟢 **LOW**
**Impact**: 50% faster page processing
**Complexity**: Low
**Files**: `src/fileintel/document_processing/chunking.py`

**Changes**:
```python
def _aggregate_pages_efficiently(self, sentences: List[Dict]) -> Dict[str, Any]:
    """Fast page aggregation using set operations and early exit."""
    all_pages = set()

    # Early exit if no sentences
    if not sentences:
        return {'pages': [], 'page_range': None}

    # Collect all pages in single pass
    for sentence in sentences:
        if page_info := sentence.get('page_info', {}).get('pages'):
            all_pages.update(page_info)

    if not all_pages:
        return {'pages': [], 'page_range': None}

    # Sort once at the end
    pages_list = sorted(all_pages)
    page_range = f"{pages_list[0]}-{pages_list[-1]}" if len(pages_list) > 1 else str(pages_list[0])

    return {
        'pages': pages_list,
        'page_range': page_range
    }

# Replace existing page aggregation logic in vector/graph chunk creation
```

---

### **Phase 3: Storage & I/O Optimizations (2-3 hours implementation)**
Focus on database performance and batch operations.

#### **3.1 Batch Database Operations**
**Priority**: 🟡 **MEDIUM**
**Impact**: 5x faster storage operations
**Complexity**: Medium
**Files**: `src/fileintel/storage/document_storage.py`, `src/fileintel/tasks/document_tasks.py`

**Changes**:
```python
# Add to document_storage.py
def add_document_chunks_batch(self, document_id: str, collection_id: str, chunks_data: List[Dict], batch_size: int = 100):
    """Add chunks in batches for better I/O performance."""
    chunk_objects = []

    for i in range(0, len(chunks_data), batch_size):
        batch = chunks_data[i:i + batch_size]
        batch_objects = []

        for j, chunk_data in enumerate(batch):
            chunk_text = self.base._clean_text(chunk_data.get("text", ""))
            if not chunk_text.strip():
                continue

            import uuid
            chunk_id = str(uuid.uuid4())

            metadata = {}
            metadata.update(chunk_data.get("metadata", {}))

            chunk = DocumentChunk(
                id=chunk_id,
                document_id=document_id,
                collection_id=collection_id,
                position=i + j,
                chunk_text=chunk_text,
                chunk_metadata=metadata,
            )

            batch_objects.append(chunk)

        # Batch insert
        self.db.add_all(batch_objects)
        chunk_objects.extend(batch_objects)

        # Commit every batch to avoid large transactions
        self.base._safe_commit()

    return chunk_objects

# Modify document_tasks.py to use batch operations
def process_document_into_chunks(...):
    # ... existing logic ...

    # Store vector chunks in batch
    storage.add_document_chunks_batch(
        actual_document_id, collection_id, chunk_data, batch_size=50
    )

    # Store graph chunks in batch
    if full_chunking_result and 'graph_chunks' in full_chunking_result:
        storage.add_document_chunks_batch(
            actual_document_id, collection_id, graph_chunk_data, batch_size=25
        )
```

#### **3.2 Connection Pooling & Transaction Optimization**
**Priority**: 🟢 **LOW**
**Impact**: 20% faster database operations
**Complexity**: Low

---

### **Phase 4: Advanced Optimizations (1-2 days implementation)**
Optional optimizations for extreme performance requirements.

#### **4.1 Memory Pooling**
**Priority**: 🟢 **LOW** (only for >500MB documents)
**Impact**: Reduced garbage collection overhead
**Complexity**: High

#### **4.2 GPU-Accelerated Tokenization**
**Priority**: 🟢 **LOW** (only if GPU available)
**Impact**: 10x faster tokenization
**Complexity**: High

#### **4.3 Streaming Architecture**
**Priority**: 🟢 **LOW** (only for >1GB documents)
**Impact**: Constant memory usage regardless of document size
**Complexity**: Very High

---

## 🎯 **Implementation Timeline**

### **Week 1: Foundation**
- [ ] **Day 1-2**: Implement Phase 1 optimizations
- [ ] **Day 3**: Testing and benchmarking Phase 1
- [ ] **Day 4-5**: Implement Phase 2 optimizations

### **Week 2: Polish & Validation**
- [ ] **Day 1**: Testing and benchmarking Phase 2
- [ ] **Day 2-3**: Implement Phase 3 optimizations
- [ ] **Day 4-5**: End-to-end testing with real 100MB PDFs

### **Optional Week 3: Advanced Features**
- [ ] Phase 4 optimizations (if needed)
- [ ] Performance tuning and profiling
- [ ] Documentation updates

---

## 📊 **Success Metrics**

### **Performance Benchmarks**
| Document Size | Current Time | Target Time | Target Memory |
|---------------|--------------|-------------|---------------|
| 10MB | ~4.5s | <2s | <50MB |
| 50MB | ~22s | <8s | <250MB |
| 100MB | ~45s | <15s | <400MB |
| 200MB | ~90s | <25s | <600MB |

### **Quality Assurance**
- [ ] Zero data loss during optimization
- [ ] Identical chunking results (deterministic)
- [ ] All existing tests pass
- [ ] Memory usage stays within 1GB limit
- [ ] No performance regression for small documents (<10MB)

---

## 🔧 **Development Guidelines**

### **Code Quality**
- Maintain existing code structure and patterns
- Add comprehensive logging for performance monitoring
- Include fallback mechanisms for compatibility
- Document all configuration parameters

### **Testing Strategy**
```python
# Add performance test suite
class TestChunkingPerformance:
    def test_large_document_processing(self):
        # Test with synthetic 100MB document
        pass

    def test_memory_usage_monitoring(self):
        # Monitor memory during processing
        pass

    def test_parallel_vs_sequential(self):
        # Compare performance improvements
        pass
```

### **Configuration Options**
```yaml
# Add to config/default.yaml
rag:
  chunking:
    performance:
      parallel_processing_threshold: 5000000  # 5MB
      max_parallel_workers: 4
      batch_token_counting: true
      lazy_text_evaluation: true
      database_batch_size: 50
```

---

## 🚨 **Risk Mitigation**

### **Potential Issues**
1. **Thread Safety**: Ensure tokenizers are thread-safe
2. **Memory Spikes**: Monitor batch sizes to prevent OOM
3. **ID Conflicts**: Verify sentence ID uniqueness in parallel processing
4. **Database Locks**: Avoid long-running transactions

### **Rollback Plan**
- Feature flags for each optimization
- A/B testing capability
- Performance regression detection
- Quick disable mechanism for problematic optimizations

---

## 📋 **Next Steps**

1. **Immediate**: Review and approve optimization plan
2. **Setup**: Create performance testing environment with 100MB sample PDFs
3. **Implementation**: Start with Phase 1 parallel processing
4. **Validation**: Benchmark each phase against current implementation
5. **Deployment**: Gradual rollout with monitoring

This optimization plan will transform the system from handling 100MB PDFs "adequately" to processing them "efficiently" while maintaining all the benefits of the two-tier chunking architecture.