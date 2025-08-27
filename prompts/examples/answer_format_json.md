# Answer Format: JSON

Please provide your answer as a single, valid JSON object. The specific structure of the JSON object can be whatever you deem most appropriate for the user's question, but the entire output **must** be a single JSON object.

### Example of Expected Format

```json
{
  "document": "document name",
  "key_points": [
    {
      "category": "categorization of the item",
      "support_need": "categorization of support need severity related to the item, very high, high, minor, no need, not relevant",
      "point": "Question",
      "details": "First answer / first item",
      "page_reference": 2
    },
    {
      "category": "categorization of the item",
      "support_need": "categorization of support need severity related to the item, very high, high, minor, no need, not relevant",
      "point": "Question",
      "details": "second answer / second item",
      "page_reference": 2
    }
  ],
  "confidence_score": 0.95
}
```
