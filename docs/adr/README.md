# Architectural Decision Records (ADRs)

This directory contains Architectural Decision Records (ADRs) for the FileIntel project. ADRs document important architectural decisions, their context, and rationale to prevent future confusion and provide historical context.

## ADR Format

Each ADR follows this structure:
- **Title**: Brief description of the decision
- **Status**: Proposed, Accepted, Deprecated, Superseded
- **Context**: The situation that led to the decision
- **Decision**: What was decided
- **Consequences**: Implications of the decision

## Current ADRs

| Number | Title | Status |
|--------|-------|--------|
| [001](001-celery-migration.md) | Migration from Job Management to Celery | Accepted |
| [002](002-storage-layer-simplification.md) | Storage Layer Simplification | Accepted |
| [003](003-api-versioning-strategy.md) | API Versioning Strategy | Accepted |
| [004](004-graphrag-integration-approach.md) | GraphRAG Integration as Black-Box Dependency | Accepted |
| [005](005-test-architecture-modernization.md) | Test Architecture Modernization | Accepted |

## Guidelines

- ADRs should be numbered sequentially
- Decisions should be documented when they have significant impact
- Status should be updated when decisions change
- ADRs should not be edited after acceptance (create new ADR if needed)
