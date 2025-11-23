# bugs
- citation tracing does not properly handle citations with a entity name that is not surname
- Answers often refer to collections, prompts to be improved
  - src/graphrag/prompts/*  index AND query, replace mentions of community with e.g. topic.
- Lists in the embeddings are often fetched, and those segments cannot serve as citation sources.
- It is possible to submit an empty query: "fileintel query collection --type local --format list thesis_sources "" "
- Get rid of small chunks totally