Provide the extracted metadata as a valid JSON object with the following structure. Use `null` for any field that cannot be clearly identified from the document text:

```json
{
    "title": "string or null",
    "authors": ["array of author names"] or null,
    "publication_date": "YYYY-MM-DD or YYYY" or null,
    "publisher": "string or null",
    "doi": "string or null",
    "source_url": "string or null",
    "language": "string or null",
    "document_type": "string or null",
    "keywords": ["array of keywords"] or null,
    "abstract": "string or null",
    "harvard_citation": "string or null"
}
```

**Important**:
- Return ONLY the JSON object, no additional text or explanations
- Ensure the JSON is properly formatted and valid
- Be conservative - use null rather than guessing
- For authors, extract full names as they appear in the document
- For dates, prefer full dates but use year-only if that's all that's available
- The harvard_citation should follow proper Harvard referencing style based on available metadata
