# Answer Format: JSON

Please provide your answer as a single, valid JSON object. The specific structure of the JSON object can be whatever you deem most appropriate for the user's question, but the entire output **must** be a single JSON object.

### Example of Expected Format

```json
{
  "summary": "This is a brief summary of the findings from the document.",
  "key_points": [
    {
      "point": "First key point",
      "details": "Elaboration on the first key point.",
      "page_reference": 2
    },
    {
      "point": "Second key point",
      "details": "Elaboration on the second key point.",
      "page_reference": 5
    }
  ],
  "confidence_score": 0.95
}
```