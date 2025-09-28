Classify this query as VECTOR, GRAPH, or HYBRID based on the user's intent.

- **VECTOR**: For specific factual questions, document search, or similarity-based queries.
  - Examples: "What is the capital of France?", "Find documents similar to this one.", "Summarize the report on Q3 earnings."
- **GRAPH**: For questions about relationships, connections, entities, or communities within the documents.
  - Examples: "How are John Doe and Jane Smith connected?", "What are the main themes in the documents?", "Show me the community of researchers working on AI."
- **HYBRID**: For complex, multi-part questions that require both factual lookup and relationship analysis.
  - Examples: "Find all reports by Jane Smith and analyze the relationships between the projects mentioned.", "Summarize the documents about Project X and show the key people involved."

Output your response in JSON format with the following structure:
{"type": "VECTOR|GRAPH|HYBRID", "confidence": 0.0-1.0, "reasoning": "Your explanation for the classification."}
