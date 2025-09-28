#!/bin/bash
# Script to run FileIntel tests in Docker environment
# Provides cross-platform testing capability

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Default values
TEST_TYPE="unit"
REBUILD=false
VERBOSE=false
COVERAGE=false
CLEANUP=true

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

usage() {
    echo "Usage: $0 [OPTIONS] [TEST_PATTERN]"
    echo ""
    echo "OPTIONS:"
    echo "  -t, --type TYPE        Test type: unit, integration, all (default: unit)"
    echo "  -r, --rebuild          Rebuild Docker images"
    echo "  -v, --verbose          Verbose output"
    echo "  -c, --coverage         Run with coverage report"
    echo "  --no-cleanup           Don't cleanup containers after tests"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "EXAMPLES:"
    echo "  $0                     Run unit tests"
    echo "  $0 -t integration      Run integration tests"
    echo "  $0 -t all -c           Run all tests with coverage"
    echo "  $0 --rebuild -v        Rebuild and run with verbose output"
    echo "  $0 '*alerting*'        Run tests matching pattern"
    exit 1
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

cleanup() {
    if [ "$CLEANUP" = true ]; then
        log_info "Cleaning up Docker containers..."
        cd "$PROJECT_ROOT"
        docker-compose -f docker/test-environment/docker-compose.test.yml down --remove-orphans
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--type)
            TEST_TYPE="$2"
            shift 2
            ;;
        -r|--rebuild)
            REBUILD=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -c|--coverage)
            COVERAGE=true
            shift
            ;;
        --no-cleanup)
            CLEANUP=false
            shift
            ;;
        -h|--help)
            usage
            ;;
        --)
            shift
            break
            ;;
        -*)
            echo "Unknown option: $1"
            usage
            ;;
        *)
            # Treat as test pattern
            TEST_PATTERN="$1"
            shift
            ;;
    esac
done

# Validate test type
case $TEST_TYPE in
    unit|integration|all)
        ;;
    *)
        log_error "Invalid test type: $TEST_TYPE. Must be one of: unit, integration, all"
        usage
        ;;
esac

# Set up cleanup trap
trap cleanup EXIT

log_info "Starting FileIntel Docker Test Environment"
log_info "Test Type: $TEST_TYPE"
log_info "Project Root: $PROJECT_ROOT"

# Change to project root
cd "$PROJECT_ROOT"

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    log_error "Docker is not installed or not in PATH"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    log_error "Docker Compose is not installed or not in PATH"
    exit 1
fi

# Build or rebuild images if needed
if [ "$REBUILD" = true ]; then
    log_info "Rebuilding Docker images..."
    docker-compose -f docker/test-environment/docker-compose.test.yml build --no-cache
else
    log_info "Building Docker images (if needed)..."
    docker-compose -f docker/test-environment/docker-compose.test.yml build
fi

# Start dependencies (database, redis)
log_info "Starting test dependencies..."
docker-compose -f docker/test-environment/docker-compose.test.yml up -d test-db test-redis

# Wait for dependencies to be healthy
log_info "Waiting for dependencies to be ready..."
timeout=60
counter=0
while [ $counter -lt $timeout ]; do
    if docker-compose -f docker/test-environment/docker-compose.test.yml ps test-db | grep -q "healthy" && \
       docker-compose -f docker/test-environment/docker-compose.test.yml ps test-redis | grep -q "healthy"; then
        log_success "Dependencies are ready!"
        break
    fi
    echo -n "."
    sleep 2
    counter=$((counter + 2))
done

if [ $counter -ge $timeout ]; then
    log_error "Dependencies failed to start within $timeout seconds"
    docker-compose -f docker/test-environment/docker-compose.test.yml logs test-db test-redis
    exit 1
fi

# Prepare test command
TEST_CMD=()

if [ -n "$TEST_PATTERN" ]; then
    log_info "Running tests with pattern: $TEST_PATTERN"
    TEST_CMD=("python" "-m" "pytest" "tests/" "-k" "$TEST_PATTERN")
elif [ "$COVERAGE" = true ]; then
    log_info "Running tests with coverage..."
    TEST_CMD=("./test-entrypoint.sh" "coverage")
else
    log_info "Running $TEST_TYPE tests..."
    TEST_CMD=("./test-entrypoint.sh" "$TEST_TYPE")
fi

# Add verbose flag if requested
if [ "$VERBOSE" = true ]; then
    TEST_CMD+=("-v")
fi

# Run the tests
log_info "Executing tests..."
set +e  # Don't exit on test failures

if [ "$VERBOSE" = true ]; then
    docker-compose -f docker/test-environment/docker-compose.test.yml run --rm \
        -e PYTHONPATH=/app/src \
        test-runner "${TEST_CMD[@]}"
else
    docker-compose -f docker/test-environment/docker-compose.test.yml run --rm \
        -e PYTHONPATH=/app/src \
        test-runner "${TEST_CMD[@]}" 2>&1
fi

TEST_EXIT_CODE=$?
set -e

# Show results
if [ $TEST_EXIT_CODE -eq 0 ]; then
    log_success "All tests passed!"
else
    log_error "Some tests failed (exit code: $TEST_EXIT_CODE)"

    # Show logs if tests failed and not in verbose mode
    if [ "$VERBOSE" = false ]; then
        log_info "Showing recent logs..."
        docker-compose -f docker/test-environment/docker-compose.test.yml logs --tail=50 test-runner
    fi
fi

# Copy coverage reports if generated
if [ "$COVERAGE" = true ]; then
    log_info "Copying coverage reports..."
    mkdir -p "$PROJECT_ROOT/test-results"
    docker-compose -f docker/test-environment/docker-compose.test.yml run --rm \
        test-runner sh -c "
        if [ -d /app/htmlcov ]; then
            cp -r /app/htmlcov /app/test-results/ || true
        fi
        if [ -f /app/coverage.xml ]; then
            cp /app/coverage.xml /app/test-results/ || true
        fi
    " 2>/dev/null || true
fi

exit $TEST_EXIT_CODE
