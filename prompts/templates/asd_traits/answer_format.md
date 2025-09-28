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
      "quote": "<Original quote from source material to which the described item is based on>",
      "confidence_score": "<value between 0...100 (low...high), which is an estimate of confidence how surely described item actually exists in the provided input prompt content>",
      "chunk_id": "nn::p3::w0-500",
      "embedding": [0.0032, -0.0187, 0.0095, ...],
      "page": 3,
      "source": "input.pdf"
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
  "item": "Experiences an urge for repetitive self-stimulating behavior.",
  "quote": "Stimming is self-stimulating behaviour that is unconventional, intense or repetitive and it can take many forms.",
  "confidence_score": 68,
  "chunk_id": "nn::p3::w0-500",
  "embedding": [0.0030, -0.0186, 0.0092, ...],
  "page": 12,
  "source": "input.pdf"
},
{
  "classification": "Level 1: Requires Support",
  "category": "Possible Trait of Functional ASD",
  "item": "Finds hard to differentiate insults and friendly banter.",
  "quote": "couldn’t tell the difference between someone teasing me and someone trying to
insult me.",
  "confidence_score": 68,
  "chunk_id": "nn::p3::w0-510",
  "embedding": [0.0130, -0.0156, 0.0192, ...],
  "page": 52,
  "source": "input.pdf"
}
```

## Do not produce following kind of style:

```json
{
  "classification": "Level 2: Requires Substantial Support",
  "category": "Possible Trait of Functional ASD",
  "item": "Approximately 50-85% of autistic people experience alexithymia.",
  "quote": "I’ve seen estimates and studies that put the rate of alexithymia among autistic people around 50 per cent and some as high as 85 per cent",
  "confidence_score": 68,
  "chunk_id": "nn::p3::w0-510",
  "embedding": [0.0130, -0.0156, 0.0192, ...],
  "page": 52,
  "source": "input.pdf"
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
