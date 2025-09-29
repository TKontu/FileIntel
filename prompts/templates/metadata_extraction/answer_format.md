**Response Format Requirements:**

Return your response as a JSON object with these fields (only include fields where you found information):

```json
{
  "title": "Document title as it appears in the text",
  "authors": ["Author 1", "Author 2", "Author 3"],
  "publication_date": "YYYY-MM-DD or YYYY or Month YYYY format",
  "publisher": "Publisher name or organization",
  "doi": "DOI identifier if found",
  "source_url": "URL if mentioned in the document",
  "language": "Document language (e.g., English, Spanish)",
  "document_type": "Type (e.g., research paper, report, manual, thesis)",
  "keywords": ["keyword1", "keyword2", "keyword3"],
  "abstract": "Abstract or summary text if present",
  "harvard_citation": "Properly formatted Harvard-style citation"
}
```

**Important Rules:**
- Only include fields where you found actual information in the text
- Do not include empty strings, null values, or empty arrays
- For authors, extract full names as they appear
- For dates, use the most specific format possible (prefer YYYY-MM-DD)
- For keywords, extract 3-8 relevant terms from the document
- Keep abstracts under 500 characters
- Generate a Harvard citation only if you have sufficient information (author, title, year)
