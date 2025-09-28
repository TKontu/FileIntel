# ADR-003: API Versioning Strategy

## Status
Accepted

## Context

With the migration from job management to Celery, FileIntel needed a strategy for API evolution that would:

1. **Support New Architecture**: Enable task-based API endpoints that work directly with Celery
2. **Maintain Backward Compatibility**: Ensure existing integrations continue to work
3. **Provide Migration Path**: Allow users to transition from old to new API over time
4. **Reduce Maintenance Burden**: Minimize long-term maintenance of deprecated functionality

The existing v1 API was built around the custom job management system and had limitations:
- Synchronous operations that blocked on job completion
- Limited task monitoring capabilities
- No support for advanced workflow patterns
- Tight coupling to custom job management infrastructure

## Decision

We decided to implement a dual-version API strategy with v2 as the primary development focus.

### API Versioning Approach:

1. **v1 API (Legacy - Deprecated)**:
   - Maintain existing endpoints for backward compatibility
   - No new features or significant enhancements
   - Clear deprecation warnings in documentation
   - Eventual removal planned (with advance notice)

2. **v2 API (Primary)**:
   - Task-based endpoints that work directly with Celery
   - Modern REST API design patterns
   - Comprehensive task monitoring and management
   - Authentication required for security
   - All new features implemented in v2 only

### Key Differences:

| Aspect | v1 API | v2 API |
|--------|---------|---------|
| Architecture | Job Management | Celery Tasks |
| Authentication | Optional | Required |
| Task Monitoring | Limited | Comprehensive |
| Workflow Support | Basic | Advanced (Groups, Chains, Chords) |
| Response Format | Custom | Standardized |
| Status | Deprecated | Active Development |

### URL Structure:
- v1: `/api/v1/...` (e.g., `/api/v1/collections`)
- v2: `/api/v2/...` (e.g., `/api/v2/collections`)

## Consequences

### Positive:
1. **Backward Compatibility**: Existing integrations continue to work without changes
2. **Modern API Design**: v2 API follows current best practices and standards
3. **Feature Development**: New features can be designed optimally without v1 constraints
4. **Security**: v2 API enforces authentication by default
5. **Monitoring**: Rich task monitoring and management capabilities in v2
6. **Scalability**: v2 API leverages Celery's distributed processing capabilities

### Negative:
1. **Maintenance Burden**: Temporary maintenance of two API versions
2. **Documentation Complexity**: Need to document both versions clearly
3. **User Confusion**: Users need to understand which version to use
4. **Code Duplication**: Some functionality exists in both versions

### Neutral:
1. **Migration Timeline**: Users can migrate at their own pace
2. **CLI Updates**: CLI updated to use v2 endpoints
3. **Testing**: Need comprehensive tests for both versions

## Implementation Details

### v1 API Maintenance:
- **Status**: Maintained but deprecated
- **Bug Fixes**: Critical security and data integrity issues only
- **New Features**: None - redirect users to v2
- **Documentation**: Clear deprecation notices and migration guidance

### v2 API Features:

1. **Collection Management**:
   ```
   POST /api/v2/collections
   GET /api/v2/collections
   GET /api/v2/collections/{id}
   DELETE /api/v2/collections/{id}
   ```

2. **Document Upload and Processing**:
   ```
   POST /api/v2/collections/{id}/documents
   POST /api/v2/collections/{id}/process
   ```

3. **Task Management**:
   ```
   GET /api/v2/tasks/{task_id}
   POST /api/v2/tasks/{task_id}/cancel
   GET /api/v2/tasks/active
   GET /api/v2/tasks/metrics
   POST /api/v2/tasks/submit
   ```

4. **WebSocket Support**:
   ```
   WS /api/v2/ws/tasks
   ```

### Authentication Strategy:
- **v1**: Optional (for backward compatibility)
- **v2**: Required API key authentication
- **Header Format**: `Authorization: Bearer <api-key>`

### Response Format Standardization:
```json
{
  "success": true,
  "data": { ... },
  "message": "Optional message",
  "timestamp": "2023-12-07T10:30:00Z"
}
```

## Migration Strategy

### Phase 1: Dual Operation (Current)
- Both v1 and v2 APIs operational
- v2 API fully functional with all features
- Clear documentation about which version to use

### Phase 2: Deprecation Notice (Future)
- Formal deprecation announcement for v1 API
- 6-month notice period for users to migrate
- Migration guides and tools provided

### Phase 3: v1 Removal (Future)
- Remove v1 API endpoints
- Cleanup deprecated code
- Focus entirely on v2 development

### Migration Tools:
1. **API Compatibility Layer**: Temporary shims where possible
2. **Migration Scripts**: Tools to help users update integrations
3. **Documentation**: Comprehensive migration guides
4. **Support**: Dedicated support for migration questions

## User Guidance

### For New Users:
- **Use v2 API**: Start with v2 for all new integrations
- **Authentication Setup**: Configure API keys for security
- **Modern Patterns**: Leverage task-based architecture

### For Existing Users:
- **Immediate**: Continue using v1 if it meets your needs
- **Planning**: Plan migration to v2 for long-term support
- **Benefits**: Consider v2 features: better monitoring, scalability, security

### For Developers:
- **New Features**: Implement only in v2 API
- **Bug Fixes**: v1 only for critical issues
- **Documentation**: Keep both versions documented clearly

## Quality Assurance

### Testing Strategy:
1. **v1 API**: Maintain existing tests, no new test development
2. **v2 API**: Comprehensive test suite with integration tests
3. **Compatibility**: Test that v1 endpoints continue to work
4. **Migration**: Test common migration scenarios

### Monitoring:
1. **Usage Metrics**: Track v1 vs v2 API usage
2. **Error Rates**: Monitor both versions for issues
3. **Performance**: Compare performance between versions
4. **Deprecation Timeline**: Track migration progress

## Alternatives Considered

1. **Single Version with Breaking Changes**:
   - Rejected: Would break existing integrations immediately

2. **Gradual Migration within v1**:
   - Rejected: Would require significant v1 refactoring

3. **URL Path Parameters for Versioning**:
   - Rejected: Less clear than URL prefix versioning

4. **Header-Based Versioning**:
   - Rejected: Less discoverable and cacheable than URL versioning

## References

- [API Versioning Best Practices](https://www.troyhunt.com/your-api-versioning-is-wrong-which-is/)
- [REST API Design Guidelines](https://docs.microsoft.com/en-us/azure/architecture/best-practices/api-design)
- [FileIntel API Usage Guide](../API_USAGE.md)
- [Celery Migration ADR](001-celery-migration.md)
