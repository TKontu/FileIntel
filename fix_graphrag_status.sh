#!/bin/bash
# Fix GraphRAG index status to allow resume

COLLECTION_ID="6525aacb-55b1-4a88-aaaa-a4211d03beba"

echo "Updating GraphRAG index status to 'building'..."

docker exec -i postgres psql -U user -d fileintel <<EOF
UPDATE graphrag_indices
SET index_status = 'building'
WHERE collection_id = '$COLLECTION_ID';

SELECT
    collection_id,
    index_status,
    documents_count,
    entities_count,
    communities_count,
    created_at
FROM graphrag_indices
WHERE collection_id = '$COLLECTION_ID';
EOF

echo ""
echo "✓ Status updated to 'building'"
echo "You can now run: fileintel graphrag index thesis_sources"
