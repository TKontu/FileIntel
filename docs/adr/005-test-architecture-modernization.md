# ADR-005: Test Architecture Modernization

## Status
Accepted

## Context

The transition from job management to Celery required a complete overhaul of the test architecture. The existing test infrastructure was built around the custom job management system and had several issues:

1. **Obsolete Test Infrastructure**: Tests for deleted job management and workflow classes
2. **Missing Modern Patterns**: No test patterns for Celery task testing
3. **Outdated Configuration**: Test configuration still referenced job-based architecture
4. **Incomplete Coverage**: No tests for new v2 API endpoints
5. **Legacy Fixtures**: Test fixtures designed for job management, not task processing

The existing test suite included:
- Tests for deleted JobManager and WorkerRegistry classes
- Workflow tests for removed QuestionWorkflow and AnalysisWorkflow
- Job-based fixtures and mock patterns
- Configuration focused on synchronous job processing

## Decision

We decided to completely modernize the test architecture to support the new Celery-based system while maintaining comprehensive coverage.

### Key Changes:

1. **Remove Obsolete Tests**: Delete tests for deleted functionality
2. **Create Celery Test Patterns**: Establish patterns for testing distributed tasks
3. **Add v2 API Tests**: Comprehensive coverage of new task-based endpoints
4. **Update Test Configuration**: Modern pytest configuration with new markers
5. **Create Task-Focused Fixtures**: Replace job management fixtures with Celery task fixtures

### Test Architecture Principles:

1. **Isolated Task Testing**: Test Celery tasks independently of the distributed system
2. **Comprehensive API Coverage**: Test both v1 (legacy) and v2 (current) endpoints
3. **Mock External Dependencies**: Mock GraphRAG, LLM providers, and storage appropriately
4. **Performance Testing**: Include tests for task performance and resource usage
5. **Integration Testing**: Test actual task execution patterns

## Implementation Details

### Test File Organization:

1. **Removed Obsolete Tests** (5 files):
   - `test_job_manager_retry_system.py`
   - `test_worker_integration.py`
   - `test_job_processor_registry.py`
   - `test_question_processor.py`
   - `test_analysis_processor.py`

2. **New Test Files**:
   - `test_celery_tasks.py`: Unit tests for individual Celery tasks
   - `test_api_v2_endpoints.py`: Integration tests for v2 API
   - `test_graphrag_integration.py`: Updated GraphRAG Celery task tests

3. **Updated Configuration**:
   - `pytest.ini`: New markers and test discovery patterns
   - `run_tests.py`: New test categories and commands
   - `test_celery_config.py`: Celery-specific test configuration

### New Test Fixtures:

```python
# Celery Task Testing
@pytest.fixture
def mock_celery_task():
    """Mock Celery task with standard attributes"""

@pytest.fixture
def mock_celery_result():
    """Mock Celery AsyncResult for task monitoring"""

@pytest.fixture
def sample_task_data():
    """Sample task data for testing"""

# Domain-Specific Fixtures
@pytest.fixture
def mock_document_processing_task():
    """Mock document processing task result"""

@pytest.fixture
def mock_graphrag_task():
    """Mock GraphRAG task result"""

@pytest.fixture
def mock_llm_task():
    """Mock LLM task result"""
```

### Test Categories and Markers:

```ini
# pytest.ini markers
markers =
    unit: Unit tests
    integration: Integration tests (requires database)
    celery: Celery task tests
    graphrag: GraphRAG related tests
    v2_api: V2 API endpoint tests
    workflow: Workflow orchestration tests
```

### Test Execution Commands:

```bash
# New test categories
python run_tests.py celery      # Celery task tests
python run_tests.py api_v2      # V2 API tests
python run_tests.py workflow    # Workflow tests
python run_tests.py graphrag    # GraphRAG tests
```

## Test Patterns

### Celery Task Testing Pattern:

```python
@pytest.mark.celery
class TestDocumentTasks:
    @patch('src.fileintel.tasks.document_tasks.get_storage')
    @patch('src.fileintel.tasks.document_tasks.UnifiedDocumentProcessor')
    def test_process_document_task(self, mock_processor_class, mock_get_storage):
        # Setup mocks
        mock_get_storage.return_value = mock_storage
        mock_processor = Mock()
        mock_processor.process_document.return_value = {...}
        mock_processor_class.return_value = mock_processor

        # Execute task
        result = process_document(file_path, document_id, collection_id)

        # Verify result and calls
        assert result['status'] == 'completed'
        mock_processor.process_document.assert_called_once()
```

### v2 API Testing Pattern:

```python
@pytest.mark.integration
@pytest.mark.v2_api
class TestV2APIEndpoints:
    @patch('src.fileintel.api.dependencies.get_storage')
    def test_create_collection_v2(self, mock_get_storage, client, api_headers):
        # Setup mock
        mock_get_storage.return_value = mock_storage
        mock_storage.create_collection.return_value = Collection(...)

        # Make request
        response = client.post("/api/v2/collections", json={...}, headers=api_headers)

        # Verify response
        assert response.status_code == 200
        assert response.json()["success"] is True
```

### Workflow Testing Pattern:

```python
@pytest.mark.celery
@pytest.mark.workflow
class TestWorkflowTasks:
    def test_task_chaining_pattern(self, mock_celery_task, mock_celery_result):
        # Test Celery groups, chains, and chords
        with patch('celery.chain') as mock_chain:
            mock_chain.return_value.apply_async.return_value = mock_celery_result
            # Verify workflow patterns work correctly
```

## Consequences

### Positive:
1. **Modern Test Infrastructure**: Test patterns aligned with Celery architecture
2. **Comprehensive Coverage**: Tests for all new functionality (v2 API, Celery tasks)
3. **Clear Organization**: Test categories and markers for easy execution
4. **Isolated Testing**: Can test components independently
5. **CI/CD Ready**: Test configuration suitable for automated testing
6. **Performance Testing**: Patterns for testing task performance and resource usage

### Negative:
1. **Learning Curve**: Team needs to understand new test patterns
2. **Setup Complexity**: More sophisticated mocking for distributed systems
3. **Test Maintenance**: Need to maintain tests for both v1 and v2 APIs

### Neutral:
1. **Test Count**: Similar number of tests, but focused on current architecture
2. **Execution Time**: Test execution time similar to previous architecture
3. **Coverage**: Maintains high test coverage while supporting new patterns

## Configuration Updates

### pytest.ini Changes:
- **testpaths**: Updated to include both unit and integration tests
- **markers**: Added Celery, workflow, and v2 API specific markers
- **warnings**: Added filters for Celery-related warnings

### run_tests.py Enhancements:
- **New Commands**: Added celery, api_v2, workflow test categories
- **Updated Help**: Clear documentation of available test commands
- **Parallel Execution**: Support for running test categories in parallel

### Celery Test Configuration:
- **Task Isolation**: Tests execute tasks synchronously for predictability
- **Mock Patterns**: Standardized mocking of Celery task infrastructure
- **Result Handling**: Proper mocking of AsyncResult and task status

## Quality Assurance

### Test Coverage Goals:
- **Unit Tests**: 90%+ coverage for Celery tasks
- **Integration Tests**: 85%+ coverage for API endpoints
- **Workflow Tests**: 80%+ coverage for complex task patterns

### Performance Testing:
- **Task Execution Time**: Verify tasks complete within expected timeframes
- **Memory Usage**: Test memory consumption for large document processing
- **Concurrency**: Test behavior under concurrent task execution

### Error Handling Testing:
- **Task Failures**: Test retry mechanisms and error reporting
- **Network Issues**: Test behavior when external services are unavailable
- **Resource Limits**: Test behavior under resource constraints

## Migration Strategy

### Phase 1: Infrastructure Setup
- ✅ **Create new test configuration and fixtures**
- ✅ **Establish Celery test patterns**
- ✅ **Set up test execution commands**

### Phase 2: Test Migration
- ✅ **Remove obsolete tests for deleted functionality**
- ✅ **Create comprehensive v2 API tests**
- ✅ **Update GraphRAG integration tests**

### Phase 3: Enhancement
- ✅ **Add workflow orchestration tests**
- ✅ **Create performance test patterns**
- ✅ **Establish CI/CD integration**

## Success Metrics

### Coverage Metrics:
- ✅ **Removed 5 obsolete test files**
- ✅ **Added 3 new comprehensive test files**
- ✅ **Created 8 new Celery-specific fixtures**
- ✅ **Updated test configuration with 5 new markers**

### Test Organization:
- ✅ **Clear separation of unit vs integration tests**
- ✅ **Domain-specific test categorization**
- ✅ **Easy test execution with targeted commands**

### Quality Improvements:
- ✅ **Tests aligned with actual architecture**
- ✅ **No tests for non-existent functionality**
- ✅ **Comprehensive coverage of new features**

## Future Considerations

### Continuous Integration:
- Test execution in CI/CD pipelines
- Parallel test execution for faster feedback
- Test result reporting and coverage tracking

### Performance Testing:
- Load testing for high-volume document processing
- Stress testing for concurrent task execution
- Resource usage monitoring during testing

### Test Data Management:
- Standardized test data sets for consistent testing
- Test data cleanup and isolation
- Realistic test scenarios based on production usage

## References

- [pytest Documentation](https://docs.pytest.org/)
- [Celery Testing Documentation](https://docs.celeryproject.org/en/stable/userguide/testing.html)
- [FastAPI Testing](https://fastapi.tiangolo.com/tutorial/testing/)
- [Test Infrastructure Files](../../tests/)
