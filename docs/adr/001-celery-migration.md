# ADR-001: Migration from Job Management to Celery

## Status
Accepted

## Context

FileIntel originally used a custom job management system with Redis queues and background workers to handle asynchronous document processing tasks. However, this approach had several limitations:

1. **Custom Implementation Complexity**: The job management system required significant custom code for task distribution, retry logic, and monitoring
2. **Limited Scalability**: Scaling workers required manual configuration and lacked horizontal scaling capabilities
3. **Poor Monitoring**: Limited visibility into task status, worker health, and queue performance
4. **Maintenance Overhead**: Custom code required ongoing maintenance and debugging
5. **Industry Standards**: Lack of industry-standard patterns for distributed task processing

The existing system consisted of:
- Custom `JobManager` class for task coordination
- Manual worker processes pulling from Redis queues
- Custom retry and error handling logic
- Limited monitoring and management capabilities

## Decision

We decided to migrate from the custom job management system to Celery, a mature distributed task queue framework.

### Key Changes:
1. **Replace JobManager with Celery**: Convert all job processing logic to Celery tasks
2. **Implement Task-Based Architecture**: Organize tasks by domain (document processing, GraphRAG, LLM operations)
3. **Add Advanced Patterns**: Use Celery's groups, chains, and chords for complex workflows
4. **Create V2 API**: New API endpoints that work directly with Celery tasks
5. **Maintain V1 Compatibility**: Keep existing API endpoints for backward compatibility

### Technical Implementation:
- `src/fileintel/celery_config.py`: Centralized Celery configuration
- `src/fileintel/tasks/`: Domain-organized task modules
- `src/fileintel/api/routes/*_v2.py`: Task-based API endpoints
- Updated CLI to use v2 API endpoints

## Consequences

### Positive:
1. **Industry Standard**: Leverages mature, well-tested framework with extensive documentation
2. **Horizontal Scalability**: Easy to scale workers across multiple machines
3. **Rich Monitoring**: Built-in monitoring with Flower web UI and CLI tools
4. **Advanced Patterns**: Groups, chains, and chords enable sophisticated workflows
5. **Better Error Handling**: Automatic retry logic and comprehensive error reporting
6. **Resource Management**: Better memory and concurrency management
7. **Configuration Flexibility**: Queue routing, worker specialization, and resource allocation

### Negative:
1. **Migration Effort**: Required significant refactoring of existing code
2. **Additional Dependency**: Adds Celery as a core dependency
3. **Learning Curve**: Team needs to understand Celery concepts and patterns
4. **API Duplication**: Temporary maintenance of both v1 and v2 APIs

### Neutral:
1. **Redis Dependency**: Still uses Redis as message broker (existing dependency)
2. **Task Organization**: Tasks now organized by domain rather than job types
3. **Testing Changes**: Required updating test infrastructure for task-based testing

## Implementation Timeline

1. **Phase 1**: Create Celery configuration and basic task structure
2. **Phase 2**: Migrate core document processing tasks
3. **Phase 3**: Implement GraphRAG and LLM tasks
4. **Phase 4**: Add workflow orchestration patterns
5. **Phase 5**: Create v2 API endpoints
6. **Phase 6**: Update CLI and documentation
7. **Phase 7**: Deprecate but maintain v1 API for backward compatibility

## Alternatives Considered

1. **Improve Custom System**: Enhance existing job management with better monitoring and scaling
   - Rejected: Would require significant custom development with uncertain outcomes

2. **Other Task Queues** (RQ, Dramatiq, Huey):
   - Rejected: Celery provides the most comprehensive feature set and ecosystem support

3. **Cloud-Based Solutions** (AWS SQS, Google Cloud Tasks):
   - Rejected: Adds cloud dependency and complexity for self-hosted deployments

## References

- [Celery Documentation](https://docs.celeryproject.org/)
- [FileIntel Celery Architecture](../CELERY_ARCHITECTURE.md)
- [API Migration Guide](../API_USAGE.md)
