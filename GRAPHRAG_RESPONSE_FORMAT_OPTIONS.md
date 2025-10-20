# GraphRAG Response Format Improvement Options

## Current Problem

The CLI displays raw Python dictionary output:
```python
{'response': "## Understanding Agile...", 'context': {'reports': DataFrame}}
```

This is unreadable and exposes internal structure.

## Root Cause

**File:** `src/fileintel/cli/graphrag.py:76`

```python
answer = response_data.get("answer", "No answer provided")  # ❌ Wrong key
```

The GraphRAG service returns:
```python
{
    "response": "markdown text...",  # ← The actual answer
    "context": {
        "reports": DataFrame,
        "entities": [...],
        "relationships": [...]
    }
}
```

But the CLI is looking for `"answer"` instead of `"response"`.

## Improvement Options

### Option 1: Quick Fix - Display Response Text Only (Recommended)

**Change:** Update line 76 to use correct key

**File:** `src/fileintel/cli/graphrag.py`

```python
# Before:
answer = response_data.get("answer", "No answer provided")

# After:
answer = response_data.get("response", "No answer provided")
```

**Result:**
```
GraphRAG Query: what is agile
Collection: test

Answer:
## Understanding Agile Methodologies

Agile methodologies are increasingly central to modern product development...
[Full markdown rendered cleanly]
```

**Pros:**
- ✅ Simple 1-line fix
- ✅ Clean output
- ✅ Renders markdown properly

**Cons:**
- ❌ Doesn't show which reports/entities were used
- ❌ No source attribution

---

### Option 2: Enhanced Display with Sources (Best UX)

**Change:** Parse context and show sources

```python
# Extract response
response_text = response_data.get("response", "No answer provided")

# Parse context for sources
context = response_data.get("context", {})
reports = context.get("reports")
entities = context.get("entities", [])

# Display response
cli_handler.console.print(f"[bold green]Answer:[/bold green]")
# Render markdown
from rich.markdown import Markdown
cli_handler.console.print(Markdown(response_text))

# Show sources
if reports is not None and not reports.empty:
    cli_handler.console.print(f"\n[bold blue]Sources - Community Reports ({len(reports)}):[/bold blue]")
    for _, report in reports.head(5).iterrows():
        title = report.get('title', 'Unknown')
        rank = report.get('rank', 0)
        cli_handler.console.print(f"  • {title} (rank: {rank})")

if entities:
    cli_handler.console.print(f"\n[bold blue]Key Entities ({len(entities)}):[/bold blue]")
    for entity in entities[:10]:
        name = entity.get('title', entity.get('name', 'Unknown'))
        entity_type = entity.get('type', 'Unknown')
        cli_handler.console.print(f"  • {name} ({entity_type})")
```

**Result:**
```
GraphRAG Query: what is agile
Collection: test

Answer:

  Understanding Agile Methodologies

  Agile methodologies are increasingly central to modern product
  development, impacting companies across various sectors [Data:
  Reports (5)]...

  Core Principles and Frameworks

  At its core, Agile emphasizes flexibility, collaboration...

Sources - Community Reports (27):
  • Agile Methodologies and New Product Development (rank: 7.0)
  • Scrum Framework and Implementation (rank: 7.0)
  • Manufacturing Adoption of Agile-Stage-Gate (rank: 7.5)
  • Lean-Agile Integration (rank: 7.0)
  • Business Model Innovation (rank: 7.0)

Key Entities (10):
  • AGILE (Methodology)
  • SCRUM (Framework)
  • NEW PRODUCT DEVELOPMENT (Process)
  • STAGE-GATE (Methodology)
  • Robert G. Cooper (Person)
```

**Pros:**
- ✅ Clean markdown rendering
- ✅ Shows source attribution
- ✅ Reveals which entities/reports contributed
- ✅ Professional UX

**Cons:**
- Requires more code changes

---

### Option 3: Full Detailed Output (For Research/Debugging)

**Change:** Show complete context with expandable sections

```python
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

# Display main response
response_text = response_data.get("response", "No answer provided")
cli_handler.console.print(Panel(Markdown(response_text), title="Answer", border_style="green"))

# Context details in expandable format
context = response_data.get("context", {})

# Reports table
if 'reports' in context and context['reports'] is not None:
    reports_df = context['reports']

    table = Table(title="Community Reports Used", show_header=True)
    table.add_column("ID", style="cyan")
    table.add_column("Title", style="white")
    table.add_column("Rank", justify="right", style="green")

    for _, row in reports_df.head(10).iterrows():
        table.add_row(
            str(row.get('id', '')),
            str(row.get('title', 'Unknown'))[:50],
            f"{row.get('rank', 0):.1f}"
        )

    cli_handler.console.print(table)

# Entities
if 'entities' in context:
    entities = context['entities']
    cli_handler.console.print(f"\n[bold]Entities Found:[/bold] {len(entities)}")
    for entity in entities[:15]:
        cli_handler.console.print(f"  • {entity.get('title', 'Unknown')} ({entity.get('type', 'Unknown')})")

# Relationships
if 'relationships' in context:
    relationships = context['relationships']
    cli_handler.console.print(f"\n[bold]Relationships:[/bold] {len(relationships)}")
    for rel in relationships[:10]:
        source = rel.get('source', '?')
        target = rel.get('target', '?')
        rel_type = rel.get('type', '?')
        cli_handler.console.print(f"  {source} --[{rel_type}]--> {target}")
```

**Result:**
```
╭─ Answer ─────────────────────────────────────────────────────╮
│                                                               │
│  # Understanding Agile Methodologies                          │
│                                                               │
│  Agile methodologies are increasingly central...              │
│                                                               │
╰───────────────────────────────────────────────────────────────╯

        Community Reports Used
┏━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┓
┃ ID  ┃ Title                                   ┃  Rank ┃
┡━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━┩
│ 10  │ Agile Methodologies and NPD             │   7.0 │
│ 4   │ Scrum Framework Implementation          │   7.0 │
│ 3   │ Manufacturing Agile Adoption            │   7.5 │
└─────┴─────────────────────────────────────────┴───────┘

Entities Found: 156
  • AGILE (Methodology)
  • SCRUM (Framework)
  • NEW PRODUCT DEVELOPMENT (Process)
  ...

Relationships: 423
  AGILE --[INFLUENCES]--> NEW PRODUCT DEVELOPMENT
  SCRUM --[PART_OF]--> AGILE
  ...
```

**Pros:**
- ✅ Complete transparency
- ✅ Great for debugging
- ✅ Shows knowledge graph structure

**Cons:**
- ❌ Verbose for quick queries
- ❌ Overwhelming for end users

---

### Option 4: Configurable Output Format

**Change:** Add `--format` flag

```python
@app.command("query")
def query_with_graphrag(
    collection_identifier: str = typer.Argument(...),
    question: str = typer.Argument(...),
    format: str = typer.Option(
        "simple",
        help="Output format: simple, detailed, json, markdown"
    ),
):
    # ... query logic ...

    if format == "simple":
        # Option 1: Just the response
        print_simple(response_data)
    elif format == "detailed":
        # Option 2: Response + sources
        print_detailed(response_data)
    elif format == "full":
        # Option 3: Everything
        print_full(response_data)
    elif format == "json":
        # Raw JSON output
        import json
        print(json.dumps(response_data, indent=2, default=str))
    elif format == "markdown":
        # Save to markdown file
        save_markdown(response_data, question)
```

**Usage:**
```bash
# Simple output (default)
fileintel graphrag query test "what is agile"

# Detailed with sources
fileintel graphrag query test "what is agile" --format detailed

# Full debug output
fileintel graphrag query test "what is agile" --format full

# Raw JSON for processing
fileintel graphrag query test "what is agile" --format json > result.json
```

**Pros:**
- ✅ Flexible for different use cases
- ✅ Power users get details when needed
- ✅ Simple users get clean output

**Cons:**
- Requires more implementation work

---

## Recommended Implementation

**Start with Option 2 (Enhanced Display)** as the default, with these improvements:

1. **Fix the key name** - `response` not `answer`
2. **Render markdown** - Use Rich's Markdown renderer
3. **Show top sources** - Display community reports used
4. **Extract citations** - Parse `[Data: Reports (X)]` references
5. **Format cleanly** - Use Rich panels and formatting

**Then optionally add Option 4** for power users who want JSON output or full details.

## Implementation Priority

### Immediate (Option 1 - Quick Fix)
```python
# Line 76 in graphrag.py
answer = response_data.get("response", "No answer provided")  # Fix key name
```

This alone will make the output readable.

### Short-term (Option 2 - Enhanced Display)
- Add markdown rendering
- Parse and display sources
- Show entity/report attribution

### Long-term (Option 4 - Configurable)
- Add `--format` flag
- Support multiple output modes
- Add export to markdown/JSON

## Example Implementation (Option 2)

```python
# src/fileintel/cli/graphrag.py

from rich.markdown import Markdown
from rich.panel import Panel

@app.command("query")
def query_with_graphrag(
    collection_identifier: str,
    question: str,
):
    """Query using GraphRAG."""

    # ... API call ...
    response_data = result.get("data", result)

    # Extract response and context
    response_text = response_data.get("response", "No answer provided")
    context = response_data.get("context", {})

    # Display query info
    cli_handler.console.print(f"[bold blue]GraphRAG Query:[/bold blue] {question}")
    cli_handler.console.print(f"[bold blue]Collection:[/bold blue] {collection_identifier}\n")

    # Render markdown answer in a panel
    cli_handler.console.print(
        Panel(
            Markdown(response_text),
            title="Answer",
            border_style="green",
            padding=(1, 2)
        )
    )

    # Show sources
    reports = context.get("reports")
    if reports is not None and len(reports) > 0:
        cli_handler.console.print(f"\n[bold blue]Community Reports Used ({len(reports)}):[/bold blue]")

        # Display top 5 reports
        for i, (_, report) in enumerate(reports.head(5).iterrows(), 1):
            title = report.get('title', 'Unknown Community')
            rank = report.get('rank', 0)
            cli_handler.console.print(f"  {i}. {title} [dim](relevance: {rank})[/dim]")

        if len(reports) > 5:
            cli_handler.console.print(f"  [dim]... and {len(reports) - 5} more[/dim]")
```

This gives you a **clean, professional output** that's easy to read while still showing source attribution.
