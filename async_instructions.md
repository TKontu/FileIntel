# Async Processing Configuration Instructions

## Overview

FileIntel's GraphRAG async batching allows processing multiple document chunks concurrently for 3-5x speedup. This requires proper configuration of both the vLLM server and FileIntel.

## vLLM Server Configuration (Required)

Since your vLLM server runs on another machine (`192.168.0.247:9003`), it must be configured to handle concurrent requests.

### **Current vLLM Startup Command:**
```bash
# On your vLLM server machine (192.168.0.247)
python -m vllm.entrypoints.openai.api_server \
    --model gemma3-4B \
    --host 0.0.0.0 \
    --port 9003 \
    --max-model-len 4096
```

### **Required vLLM Configuration for Async:**
```bash
# Updated startup command with concurrent support
python -m vllm.entrypoints.openai.api_server \
    --model gemma3-4B \
    --host 0.0.0.0 \
    --port 9003 \
    --max-model-len 4096 \
    --max-num-seqs 8 \                    # CRITICAL: Allow 8 concurrent sequences
    --max-parallel-loading-workers 4 \    # Parallel model loading
    --disable-log-requests \               # Reduce log spam (optional)
    --tensor-parallel-size 1              # Single GPU (adjust if multi-GPU)
```

### **Key Parameters:**
- **`--max-num-seqs 8`**: **CRITICAL** - Without this, vLLM processes requests sequentially
- **`--max-parallel-loading-workers 4`**: Improves startup time for concurrent requests
- **`--disable-log-requests`**: Optional - reduces log volume during high concurrency

### **Memory Considerations (RTX 3090 24GB):**
- **Model size**: ~8GB VRAM for gemma3-4B
- **Concurrent sequences**: ~2-4GB additional VRAM for 8 concurrent requests
- **Total usage**: ~12GB VRAM (leaves 12GB free)
- **Safe concurrency**: 8 sequences is optimal for 24GB VRAM

## FileIntel Configuration

### **Enable Async Processing in `config/default.yaml`:**
```yaml
graphrag:
  # Existing settings
  llm_model: "gemma3-4B"
  target_sentences: 18
  overlap_sentences: 2

  # NEW: Async processing configuration
  async_processing:
    enabled: true                    # ON/OFF switch (set to false to disable)
    batch_size: 4                   # Concurrent requests per batch
    max_concurrent_requests: 8      # Must match vLLM --max-num-seqs
    batch_timeout: 30               # Timeout per batch (seconds)
    fallback_to_sequential: true    # Fallback if batching fails
```

### **Configuration Options:**

| Setting | Recommended | Range | Description |
|---------|-------------|-------|-------------|
| `enabled` | `true` | `true/false` | Master switch for async processing |
| `batch_size` | `4` | `1-8` | Chunks processed concurrently per batch |
| `max_concurrent_requests` | `8` | `1-16` | Total concurrent HTTP connections |
| `batch_timeout` | `30` | `10-120` | Seconds before batch times out |
| `fallback_to_sequential` | `true` | `true/false` | Use sequential processing on batch failure |

## Network Configuration

### **HTTP Client Settings:**
The async implementation automatically configures HTTP connection pooling:

```python
# Automatic configuration based on your async settings
httpx.AsyncClient(
    limits=httpx.Limits(
        max_connections=max_concurrent_requests,      # From config
        max_keepalive_connections=batch_size,         # From config
    ),
    timeout=httpx.Timeout(
        connect=5.0,
        read=batch_timeout,        # From config
        write=5.0,
        pool=batch_timeout + 10    # From config + buffer
    )
)
```

## Performance Expectations

### **Before Async (Sequential):**
- **7 chunks × 3 seconds = 21 seconds** total processing time

### **After Async (Batched):**
- **Batch 1**: 4 chunks in parallel = 3 seconds
- **Batch 2**: 3 chunks in parallel = 3 seconds
- **Total**: 6 seconds (**3.5x speedup**)

### **Monitoring:**
```bash
# Check vLLM server logs for concurrent requests
docker logs vllm-container | grep "concurrent"

# Check FileIntel worker logs for batch processing
docker compose logs worker | grep -i "batch"
```

## Troubleshooting

### **If Async Processing Fails:**
1. **Check vLLM configuration**: Ensure `--max-num-seqs 8` is set
2. **Check network connectivity**: Test multiple concurrent requests to vLLM
3. **Disable async**: Set `async_processing.enabled: false` to fall back to sequential
4. **Reduce batch size**: Lower `batch_size` from 4 to 2 if memory issues occur

### **Performance Not Improving:**
1. **vLLM bottleneck**: Increase `--max-num-seqs` on vLLM server
2. **Network latency**: Check network between FileIntel and vLLM servers
3. **VRAM limitation**: Monitor GPU memory usage during concurrent processing

### **VRAM Monitoring:**
```bash
# On vLLM server machine
nvidia-smi -l 1  # Monitor VRAM usage in real-time
```

## Testing Async Configuration

### **Verify vLLM Concurrent Support:**
```bash
# Test multiple concurrent requests to vLLM
curl -X POST http://192.168.0.247:9003/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Test 1"}],"temperature":0}' &
curl -X POST http://192.168.0.247:9003/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Test 2"}],"temperature":0}' &
wait
```

### **Verify FileIntel Async Processing:**
1. **Upload a document** to trigger GraphRAG indexing
2. **Check worker logs** for batch processing messages
3. **Monitor timing** - should see significant speedup vs sequential processing

## Quick Start Checklist

- [ ] Update vLLM startup command with `--max-num-seqs 8`
- [ ] Restart vLLM server with new configuration
- [ ] Add `async_processing` section to `config/default.yaml`
- [ ] Set `async_processing.enabled: true`
- [ ] Test with a document upload and monitor performance
- [ ] If issues occur, set `enabled: false` to disable async processing
