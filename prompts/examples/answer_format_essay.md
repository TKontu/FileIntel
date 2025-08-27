# Answer Format: Essay

Please provide your answer in the format of a structured essay.

Your response **must** be a JSON object containing two keys:
1.  `"title"`: A string for the main title of the essay.
2.  `"sections"`: A list of JSON objects, where each object represents a section of the essay and contains two keys:
    - `"heading"`: A string for the section's heading.
    - `"content"`: A string for the body text of that section.

### Example of Expected Format

```json
{
  "title": "In-Depth Analysis of the Financial Report",
  "sections": [
    {
      "heading": "Executive Summary",
      "content": "This report provides a detailed analysis of the company's financial performance over the last fiscal quarter, highlighting key areas of growth and potential risks."
    },
    {
      "heading": "Revenue Growth",
      "content": "The company saw a 15% increase in revenue, primarily driven by the successful launch of the new product line in the Asia-Pacific market."
    },
    {
      "heading": "Identified Risks and Mitigation",
      "content": "A potential risk identified is the increasing cost of raw materials. The proposed mitigation strategy involves diversifying suppliers and exploring alternative materials."
    }
  ]
}
```
