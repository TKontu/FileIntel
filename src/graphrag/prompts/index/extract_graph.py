GRAPH_EXTRACTION_PROMPT = """
Extract named entities and their relationships from the provided text.

=== ENTITY TYPES ===
- ORGANIZATION: Companies, institutions, products, brands (e.g., "Microsoft", "Stage-Gate", "Harvard")
- PERSON: Named individuals (e.g., "Robert Cooper", "Dr. Smith")
- GEO: Locations, cities, countries, facilities (e.g., "Paris", "Aalborg University")
- EVENT: Specific named occurrences (e.g., "Product Launch 2024", "OOPSLA Conference")

=== EXTRACTION RULES ===
1. Extract ONLY proper nouns (specific names), NOT common nouns
2. Do NOT extract generic terms: "process", "information", "tasks", "results", "development"
3. Every entity must have one of the types above
4. Extract relationships between entities that are clearly connected

=== OUTPUT FORMAT ===
("entity"{tuple_delimiter}<ENTITY_NAME>{tuple_delimiter}<TYPE>{tuple_delimiter}<brief_description>)
("relationship"{tuple_delimiter}<SOURCE_ENTITY>{tuple_delimiter}<TARGET_ENTITY>{tuple_delimiter}<relationship_description>{tuple_delimiter}<strength_0_to_10>)

Use {record_delimiter} between entries.
End with {completion_delimiter}

=== EXAMPLE ===
Text: "Microsoft CEO Satya Nadella announced a partnership with OpenAI at the Seattle conference."

Output:
("entity"{tuple_delimiter}MICROSOFT{tuple_delimiter}ORGANIZATION{tuple_delimiter}Technology company)
{record_delimiter}
("entity"{tuple_delimiter}SATYA_NADELLA{tuple_delimiter}PERSON{tuple_delimiter}CEO of Microsoft)
{record_delimiter}
("entity"{tuple_delimiter}OPENAI{tuple_delimiter}ORGANIZATION{tuple_delimiter}AI research organization)
{record_delimiter}
("entity"{tuple_delimiter}SEATTLE{tuple_delimiter}GEO{tuple_delimiter}City where conference was held)
{record_delimiter}
("relationship"{tuple_delimiter}SATYA_NADELLA{tuple_delimiter}MICROSOFT{tuple_delimiter}Satya Nadella is the CEO of Microsoft{tuple_delimiter}9)
{record_delimiter}
("relationship"{tuple_delimiter}MICROSOFT{tuple_delimiter}OPENAI{tuple_delimiter}Microsoft partnered with OpenAI{tuple_delimiter}8)
{completion_delimiter}

=== TEXT TO ANALYZE ===
{input_text}

=== YOUR EXTRACTION ===
"""

CONTINUE_PROMPT = "MANY entities and relationships were missed in the last extraction. Remember to ONLY emit entities that match any of the previously extracted types. Add them below using the same format:\n"
LOOP_PROMPT = "It appears some entities and relationships may have still been missed. Answer Y if there are still entities or relationships that need to be added, or N if there are none. Please answer with a single letter Y or N.\n"
