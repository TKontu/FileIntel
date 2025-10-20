**Response Format Requirements:**

Return your response as a JSON object with these fields (only include fields where you found information):

```json
{
  "title": "Document title as it appears in the text",
  "authors": ["Author 1 Full Name", "Author 2 Full Name", "Author 3 Full Name"],
  "author_surnames": ["Surname1", "Surname2", "Surname3"],
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
- For author_surnames, extract ONLY the surname/family name for each author (e.g., "John Smith" -> "Smith", "María García López" -> "García López")
- The author_surnames array must match the authors array in length and order
- For dates, use the most specific format possible (prefer YYYY-MM-DD)
- For keywords, extract 3-8 relevant terms from the document
- Keep abstracts under 500 characters
- Generate a Harvard citation only if you have sufficient information (author, title, year)
