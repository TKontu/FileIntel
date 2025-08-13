# Answer Format: Table

Please provide your answer in a structured table format.

Your response **must** be a JSON object containing two keys:
1.  `"headers"`: A list of strings representing the column headers of the table.
2.  `"rows"`: A list of lists, where each inner list represents a single row of data. The order of items in each row must correspond to the order of the headers.

### Example of Expected Format

```json
{
  "headers": ["Employee ID", "Full Name", "Department", "Start Date"],
  "rows": [
    ["E1023", "Alice Johnson", "Engineering", "2022-08-15"],
    ["E1024", "Robert Williams", "Marketing", "2021-03-10"],
    ["E1025", "Emily Brown", "Engineering", "2023-01-20"]
  ]
}
```
