#!/usr/bin/env python3
"""Load test for embedding endpoint - request many embeddings quickly."""

from openai import OpenAI
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure client
client = OpenAI(
    base_url="http://192.168.0.136:9003/v1",
    api_key="ollama"
)

# Generate test texts with unique IDs
test_texts = [
    f"Unique ID {i:05d}: This is test sentence number {i}. It contains sample text for embedding."
    for i in range(10000)  # 100 different texts
]

print(f"Testing with {len(test_texts)} texts")
print(f"Model: bge-large-en")
print("-" * 60)

def validate_results(expected, received, test_name):
    """Check if all embeddings were returned."""
    if received == expected:
        print(f"   ✓ Validation: All {expected} embeddings returned")
        return True
    else:
        print(f"   ✗ Validation FAILED: Expected {expected}, got {received}")
        print(f"   Missing: {expected - received} embeddings")
        return False

# Test 1: Single batch request
print("\n1. Single batch request (all texts at once)...")
start = time.time()
try:
    response = client.embeddings.create(
        model="bge-large-en",
        input=test_texts
    )
    elapsed = time.time() - start
    received = len(response.data)
    print(f"   Success: {received} embeddings in {elapsed:.2f}s")
    print(f"   Throughput: {received/elapsed:.1f} embeddings/sec")
    validate_results(len(test_texts), received, "Single batch")

    # Check embedding dimensions
    if response.data:
        dim = len(response.data[0].embedding)
        print(f"   Embedding dimension: {dim}")
except Exception as e:
    print(f"   ✗ Failed: {e}")

# Test 2: Batches of 10
print("\n2. Sequential batches of 10 texts...")
start = time.time()
batch_size = 2000
total_embeddings = 0
batches_sent = 0
batch_results = []

try:
    for i in range(0, len(test_texts), batch_size):
        batch = test_texts[i:i+batch_size]
        response = client.embeddings.create(
            model="bge-large-en",
            input=batch
        )
        received = len(response.data)
        total_embeddings += received
        batches_sent += 1
        batch_results.append((len(batch), received))

    elapsed = time.time() - start
    print(f"   Success: {total_embeddings} embeddings in {elapsed:.2f}s")
    print(f"   Batches sent: {batches_sent}")
    print(f"   Throughput: {total_embeddings/elapsed:.1f} embeddings/sec")
    validate_results(len(test_texts), total_embeddings, "Sequential batches")

    # Check for any batch mismatches
    mismatches = [(i, exp, got) for i, (exp, got) in enumerate(batch_results) if exp != got]
    if mismatches:
        print(f"   ✗ Batch mismatches detected:")
        for batch_idx, expected, got in mismatches:
            print(f"     Batch {batch_idx}: expected {expected}, got {got}")

except Exception as e:
    print(f"   ✗ Failed: {e}")

# Test 3: Parallel requests (5 concurrent threads)
print("\n3. Parallel requests (5 threads, batches of 10)...")
start = time.time()
total_embeddings = 0
failed_batches = []
batch_results = []

def process_batch(batch_id, batch):
    try:
        response = client.embeddings.create(
            model="bge-large-en",
            input=batch
        )
        return (batch_id, len(batch), len(response.data), None)
    except Exception as e:
        return (batch_id, len(batch), 0, str(e))

try:
    batches = [(i, test_texts[i:i+batch_size]) for i in range(0, len(test_texts), batch_size)]
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_batch, idx, batch) for idx, batch in batches]
        for future in as_completed(futures):
            batch_id, expected, received, error = future.result()
            batch_results.append((batch_id, expected, received, error))
            if error:
                failed_batches.append((batch_id, error))
            else:
                total_embeddings += received

    elapsed = time.time() - start
    print(f"   Success: {total_embeddings} embeddings in {elapsed:.2f}s")
    print(f"   Batches sent: {len(batches)}")
    print(f"   Throughput: {total_embeddings/elapsed:.1f} embeddings/sec")
    validate_results(len(test_texts), total_embeddings, "Parallel batches")

    # Check for any batch mismatches
    mismatches = [(bid, exp, got) for bid, exp, got, err in batch_results if err is None and exp != got]
    if mismatches:
        print(f"   ✗ Batch mismatches detected:")
        for batch_id, expected, got in mismatches:
            print(f"     Batch {batch_id}: expected {expected}, got {got}")

    # Report failed batches
    if failed_batches:
        print(f"   ✗ Failed batches: {len(failed_batches)}")
        for batch_id, error in failed_batches[:3]:  # Show first 3
            print(f"     Batch {batch_id}: {error}")

except Exception as e:
    print(f"   ✗ Failed: {e}")

print("\n" + "=" * 60)
print("Load test complete!")
