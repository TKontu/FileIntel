#!/bin/bash
# Check if GraphRAG checkpoint resume feature is installed

echo "=== GraphRAG Resume Feature Version Check ==="
echo ""

# Check current commit
echo "1. Current Git Commit:"
CURRENT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null)
RESUME_COMMIT="19fd087"
if [[ "$CURRENT_COMMIT" == "$RESUME_COMMIT"* ]]; then
    echo "   ✓ On resume commit: $CURRENT_COMMIT"
else
    echo "   Current: $CURRENT_COMMIT"
    echo "   Expected: $RESUME_COMMIT (graphrag resume)"

    # Check if we have the resume commit in history
    if git log --oneline --all | grep -q "$RESUME_COMMIT.*graphrag resume"; then
        echo "   ✓ Resume commit found in history"
    else
        echo "   ✗ Resume commit NOT found - feature may not be available"
    fi
fi
echo ""

# Check checkpoint_manager.py
echo "2. Checkpoint Manager:"
if [ -f "src/graphrag/index/run/checkpoint_manager.py" ]; then
    LINES=$(wc -l < src/graphrag/index/run/checkpoint_manager.py)
    echo "   ✓ checkpoint_manager.py exists ($LINES lines)"
else
    echo "   ✗ checkpoint_manager.py NOT FOUND"
fi
echo ""

# Check config
echo "3. Configuration:"
if grep -q "enable_checkpoint_resume" config/default.yaml 2>/dev/null; then
    echo "   ✓ Config has checkpoint resume settings:"
    grep "enable_checkpoint_resume\|validate_checkpoints" config/default.yaml | sed 's/^/     /'
else
    echo "   ✗ Config missing checkpoint resume settings"
fi
echo ""

# Check task implementation
echo "4. Task Implementation:"
if grep -q "enable_resume" src/fileintel/tasks/graphrag_tasks.py 2>/dev/null; then
    echo "   ✓ build_graphrag_index_task has resume support"
    echo "   Resume parameter passed at line:"
    grep -n "enable_resume=" src/fileintel/tasks/graphrag_tasks.py | head -1 | sed 's/^/     /'
else
    echo "   ✗ Task missing resume support"
fi
echo ""

# Check service
echo "5. GraphRAG Service:"
if grep -q "build_index_with_resume" src/fileintel/rag/graph_rag/services/graphrag_service.py 2>/dev/null; then
    echo "   ✓ GraphRAGService has build_index_with_resume method"
else
    echo "   ✗ Service missing resume method"
fi
echo ""

# Summary
echo "=== Summary ==="
if [ -f "src/graphrag/index/run/checkpoint_manager.py" ] && \
   grep -q "enable_checkpoint_resume" config/default.yaml 2>/dev/null && \
   grep -q "enable_resume" src/fileintel/tasks/graphrag_tasks.py 2>/dev/null; then
    echo "✓ GraphRAG Resume Feature: INSTALLED"
    echo ""
    echo "To use resume on restart:"
    echo "  curl -X POST http://localhost:8000/api/v2/graphrag/index \\"
    echo "    -H 'Content-Type: application/json' \\"
    echo "    -d '{\"collection_id\": \"<your-id>\", \"force_rebuild\": false}'"
else
    echo "✗ GraphRAG Resume Feature: NOT FULLY INSTALLED"
    echo "  Run: git checkout 19fd087"
fi
