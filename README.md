# FileIntel

FileIntel is a powerful document analysis system that uses a Retrieval-Augmented Generation (RAG) architecture to provide intelligent answers from your documents. It allows you to create collections of documents, upload various document formats, and then query the entire collection to get synthesized answers based on the content of the documents.

## Features

*   **RAG-based Document Analysis:** Utilizes a RAG architecture to provide accurate and context-aware answers from your documents.
*   **Collection-based Knowledge Management:** Organize your documents into collections to create distinct knowledge bases.
*   **Support for Multiple Document Formats:** Supports various document formats, including PDF, EPUB, and MOBI.
*   **Asynchronous Job Processing:** Uses a worker-based architecture to process document indexing and analysis jobs asynchronously.
*   **RESTful API:** Provides a simple and easy-to-use RESTful API for interacting with the system.
*   **CLI:** Includes a command-line interface for easy interaction with the API.

## Architecture

FileIntel is built on a microservices-oriented architecture, orchestrated with Docker Compose. The main components are:

*   **API Service:** A FastAPI application that serves as the primary entry point for all user interactions.
*   **Worker Service:** A background process that handles all heavy, time-consuming tasks asynchronously.
*   **PostgreSQL Database:** The primary data store, using the `pgvector` extension to store and query vector embeddings.
*   **Redis:** Functions as the message broker and backend for the job queue.

For a more detailed overview of the architecture, please refer to the `architecture.md` file.

## Getting Started

To get started with FileIntel, you will need to have Docker and Docker Compose installed on your system.

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/FileIntel.git
    cd FileIntel
    ```

2.  **Create a `.env` file:**

    Create a `.env` file in the root of the project and add the following environment variables:

    ```
    POSTGRES_USER=user
    POSTGRES_PASSWORD=password
    POSTGRES_DB=fileintel
    ```

3.  **Update the `docker-compose.yml` file:**

    Update the `docker-compose.yml` file to use the environment variables from the `.env` file:

    ```yaml
    services:
      postgres:
        image: "pgvector/pgvector:pg13"
        environment:
          POSTGRES_USER: ${POSTGRES_USER}
          POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
          POSTGRES_DB: ${POSTGRES_DB}
        # ...
      api:
        # ...
        environment:
          # ...
          - DB_USER=${POSTGRES_USER}
          - DB_PASSWORD=${POSTGRES_PASSWORD}
          - DB_HOST=postgres
          - DB_PORT=5432
          - DB_NAME=${POSTGRES_DB}
        # ...
      worker:
        # ...
        environment:
          # ...
          - DB_USER=${POSTGRES_USER}
          - DB_PASSWORD=${POSTGRES_PASSWORD}
          - DB_HOST=postgres
          - DB_PORT=5432
          - DB_NAME=${POSTGRES_DB}
        # ...
    ```

4.  **Build and run the application:**

    ```bash
    docker-compose up --build -d
    ```

    This will build the Docker images and start all the services in the background.

## Usage

Once the application is running, you can use the API to create collections, upload documents, and run queries. For detailed instructions on how to use the API, please refer to the `API_USAGE.md` file.

## Configuration

The application can be configured using the `config/default.yaml` file. This file contains various options for configuring the LLM, document processing, OCR, output, API, storage, and batch processing.

