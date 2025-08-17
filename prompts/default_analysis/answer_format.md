# Answer Format: JSON

You must produce a **single valid JSON object** with this structure:

### Example of Expected Format

```json # Do not copy this line with backticks
{
  "document": "document name",
  "key_points": [
    {
      "classification": "<classification level from list>",
      "category": "<category from list>",
      "item": "<full-sentence behavior/experience description>",
      "confidence_score": "<value between 0...100 (low...high), which is an estimate of confidence how surely described item actually exists in the provided input prompt content>",
    }
  ]
}
# Do not copy these backticks
```

## Positive Examples

```json
{
  "classification": "Level 2: Requires Substantial Support",
  "category": "High Probability Trait",
  "item": "Struggles with verbal communication and often relies on alternative methods, requiring tailored support to express needs.",
  "confidence_score": 68
},
{
  "classification": "Level 1: Requires Support",
  "category": "Possible Trait of Functional ASD",
  "item": "Finds it challenging to transition between activities and benefits from preparation and visual supports to feel secure.",
  "confidence_score": 21
}
```

## Do not produce following kind of style:

```json
{
  "classification": "Level 2: Requires Substantial Support",
  "category": "Possible Trait of Functional ASD",
  "item": "Approximately 50-85% of autistic people experience alexithymia.",
  "page_reference": 2
}
```

## Item Description Rules

- **Each 'item' must be a complete, descriptive sentence** describing an **observable behavior, action, or internal thought/experience**.
- **It must be phrased so a person could relate to it** (“struggles to maintain eye contact when speaking,” “needs a quiet space to recover after social interaction,” etc.).
- **Do not** output plain statistics, prevalence data, or vague noun phrases.
- **Do not** simply restate the category or classification.
- **Avoid fragments** such as “meltdowns” — instead, write “Experiences meltdowns that involve shouting, throwing objects, or using hurtful language.”
- Keep the **page_reference** as the page where this information is found in the document.

```

```
