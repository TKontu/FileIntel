import os
from typing import Any, Dict, List, Optional

import requests
from rich.console import Console
from rich.table import Table

API_BASE_URL = os.getenv("FILEINTEL_API_BASE_URL", "http://localhost:8000/api/v1")
CONSOLE = Console()


class FileIntelAPI:
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url

    def _request_raw(
        self, method: str, endpoint: str, **kwargs: Any
    ) -> requests.Response:
        try:
            response = requests.request(method, f"{self.base_url}/{endpoint}", **kwargs)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            CONSOLE.print(f"[bold red]API request failed:[/bold red] {e}")
            raise

    def _request(self, method: str, endpoint: str, **kwargs: Any) -> Any:
        response = self._request_raw(method, endpoint, **kwargs)
        return response.json()

    def create_collection(self, name: str) -> Dict[str, Any]:
        return self._request("POST", "collections", data={"name": name})

    def list_collections(self) -> List[Dict[str, Any]]:
        return self._request("GET", "collections")

    def get_collection(self, identifier: str) -> Dict[str, Any]:
        return self._request("GET", f"collections/{identifier}")

    def delete_collection(self, identifier: str) -> Dict[str, Any]:
        return self._request("DELETE", f"collections/{identifier}")

    def upload_document(
        self, collection_identifier: str, file_path: str
    ) -> Dict[str, Any]:
        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f)}
            return self._request(
                "POST", f"collections/{collection_identifier}/documents", files=files
            )

    def list_documents(self, collection_identifier: str) -> List[Dict[str, Any]]:
        return self._request("GET", f"collections/{collection_identifier}/documents")

    def delete_document(
        self, collection_identifier: str, document_identifier: str
    ) -> Dict[str, Any]:
        return self._request(
            "DELETE",
            f"collections/{collection_identifier}/documents/{document_identifier}",
        )

    def query_collection(
        self, collection_identifier: str, question: str
    ) -> Dict[str, Any]:
        return self._request(
            "POST",
            f"collections/{collection_identifier}/query",
            json={"question": question},
        )

    def analyze_collection(
        self, collection_identifier: str, task_name: str
    ) -> Dict[str, Any]:
        return self._request(
            "POST",
            f"collections/{collection_identifier}/analyze",
            json={"task_name": task_name},
        )

    def query_document(
        self, collection_identifier: str, document_identifier: str, question: str
    ) -> Dict[str, Any]:
        return self._request(
            "POST",
            f"collections/{collection_identifier}/documents/{document_identifier}/query",
            json={"question": question},
        )

    def analyze_document(
        self, collection_identifier: str, document_identifier: str, task_name: str
    ) -> Dict[str, Any]:
        return self._request(
            "POST",
            f"collections/{collection_identifier}/documents/{document_identifier}/analyze",
            json={"task_name": task_name},
        )

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        return self._request("GET", f"jobs/{job_id}/status")

    def get_job_result(self, job_id: str, markdown: bool = False) -> Any:
        import json

        endpoint = f"jobs/{job_id}/result"
        if markdown:
            # The --md flag should return the innermost document text
            response_data = self._request("GET", endpoint)
            try:
                # 1. Get the 'content' string, which is JSON-as-a-string
                content_str = response_data.get("result", {}).get("content", "{}")
                # 2. Clean up the string if it's wrapped in markdown code fences
                if content_str.startswith("```json"):
                    content_str = content_str[7:-4].strip()
                # 3. Parse the inner JSON
                inner_data = json.loads(content_str)
                # 4. Return the 'document' value, cleaning the markers
                doc_text = inner_data.get("document", "")
                if doc_text.startswith("Start of document\n---\n"):
                    doc_text = doc_text[22:]
                if doc_text.endswith("\n--- \nEnd of document"):
                    doc_text = doc_text[:-21]
                return doc_text.strip()
            except (json.JSONDecodeError, AttributeError):
                # Fallback if the structure is not as expected
                return "Could not parse the nested JSON to extract the document."

        # Default behavior: return the full JSON result
        return self._request("GET", endpoint)


def print_table(title: str, data: List[Dict[str, Any]]):
    if not data:
        CONSOLE.print(f"[yellow]No {title.lower()} found.[/yellow]")
        return

    table = Table(title=title)
    headers = data[0].keys()
    for header in headers:
        table.add_column(header.replace("_", " ").title())

    for item in data:
        table.add_row(*(str(item.get(header, "")) for header in headers))

    CONSOLE.print(table)


def print_json(data: Any):
    import json

    CONSOLE.print_json(json.dumps(data))
