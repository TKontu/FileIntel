📋 FileIntel CLI - Complete Command Reference

  🚀 Installation & Setup

  The CLI is available through Poetry as a console script:
  # In the project directory:
  poetry install
  poetry run fileintel --help

  📖 Main CLI Structure

  Entry Point: fileintel (defined in pyproject.toml)
  Architecture: Task-based Celery distributed processing
  Framework: Typer with Rich console output

  🎯 Available Commands

  🏠 Root Commands

  fileintel --help                    # Show main help
  fileintel version                   # Show CLI version and architecture info
  fileintel health                    # Check API and task system health

  📁 Collections Management

  fileintel collections --help        # Collections help
  fileintel collections create NAME   # Create new collection
  fileintel collections list          # List all collections
  fileintel collections get ID/NAME   # Get collection details
  fileintel collections delete ID     # Delete collection
  fileintel collections process ID    # Process collection with tasks
  fileintel collections status        # Show collection processing status

  📄 Document Operations

  fileintel documents --help              # Documents help
  fileintel documents upload COLLECTION FILE    # Upload single document
  fileintel documents batch-upload COLLECTION DIR  # Upload directory of files
  fileintel documents status                     # Show document processing status

  ⚙️ Task Management

  fileintel tasks --help              # Task management help
  fileintel tasks list               # List all active tasks
  fileintel tasks get TASK_ID        # Get specific task details
  fileintel tasks cancel TASK_ID     # Cancel running task
  fileintel tasks result TASK_ID     # Get task result
  fileintel tasks wait TASK_ID       # Wait for task completion
  fileintel tasks metrics            # Show task system metrics

  🔍 Query Operations

  fileintel query --help                    # Query help
  fileintel query collection ID QUESTION    # Query collection with question
  fileintel query document ID QUESTION      # Query specific document
  fileintel query status                    # Show query system status

  🕸️ GraphRAG Operations

  fileintel graphrag --help                 # GraphRAG help
  fileintel graphrag index COLLECTION       # Build graph index for collection
  fileintel graphrag query COLLECTION QUESTION  # Query with graph RAG
  fileintel graphrag status COLLECTION      # Check graph index status
  fileintel graphrag task-status TASK_ID    # Check GraphRAG task status
  fileintel graphrag list-tasks             # List all GraphRAG tasks

  🎨 CLI Features

  ✨ Rich Console Output

  - Color-coded status indicators (✓ green success, ✗ red errors)
  - Progress indicators for long-running tasks
  - JSON formatting for detailed outputs
  - Tables for listing operations

  🔧 Task-Based Architecture

  - Asynchronous processing: All operations submit Celery tasks
  - Real-time monitoring: Track task progress and status
  - Distributed execution: Tasks run on worker processes
  - Result persistence: Task results stored and retrievable

  🛡️ Error Handling

  - Graceful failures with informative error messages
  - API connectivity checks before operations
  - Task timeout handling for long operations
  - Retry mechanisms for transient failures

  📊 Example Usage Workflow

  # 1. Check system health
  fileintel health

  # 2. Create a collection
  fileintel collections create "research_papers"

  # 3. Upload documents
  fileintel documents batch-upload research_papers ./papers/

  # 4. Check upload status
  fileintel tasks list

  # 5. Build GraphRAG index
  fileintel graphrag index research_papers

  # 6. Query the collection
  fileintel query collection research_papers "What are the main findings?"

  # 7. Monitor task progress
  fileintel tasks metrics

  🔗 API Integration

  The CLI communicates with:
  - v2 API endpoints at http://localhost:8000/api/v2
  - Celery task system via Redis broker
  - Task monitoring through Flower dashboard

  ⚙️ Configuration

  CLI configuration is handled through:
  - Environment variables (.env file)
  - API base URL: FILEINTEL_API_BASE_URL (default: http://localhost:8000/api/v2)
  - Authentication: API key support via headers

  📋 Help System

  Every command supports --help:
  fileintel --help                     # Main help
  fileintel collections --help         # Collections help
  fileintel collections create --help  # Specific command help
