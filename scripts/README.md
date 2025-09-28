# Prompt Preview Script (`preview_prompt.py`)

This script is a command-line utility designed for debugging and previewing the final prompts that are sent to a Large Language Model (LLM). It constructs a complete prompt by combining a document's content, a specific question, and various template files.

This allows developers to see the exact text the LLM will receive, which is useful for ensuring the prompt is well-formed and contains the correct context.

## Usage

The script is run from the command line using Python. You must provide a document and a question.

```bash
python scripts/preview_prompt.py --doc <path_to_document> --question "<your_question>"
```

### Options

The script accepts the following command-line options:

- `--doc`: (Required) The file path to the document you want to use as context for the prompt.
- `--question`: (Required) The question you want to ask about the document.
- `--instruction`: (Optional) The name of the instruction template to use. This should be the filename (without the `.md` extension) located in the `prompts/templates` directory. Defaults to `instruction`.
- `--format`: (Optional) The name of the answer format template to use. This should be the filename (without the `.md` extension) located in the `prompts/templates` directory. Defaults to `answer_format`.
- `--max-length`: (Optional) An integer to specify the maximum character length for the composed prompt. If the prompt exceeds this length, it will be truncated.

## Examples

### Basic Preview

This example uses the default instruction and answer format templates.

```bash
python scripts/preview_prompt.py --doc C:\path\to\your\document.txt --question "What is the main idea of this document?"
```

### Specifying Custom Templates

This example shows how to use different templates for the instruction and answer format.

```bash
python scripts/preview_prompt.py \
  --doc C:\path\to\your\document.txt \
  --question "Summarize the key points." \
  --instruction "summary_instruction" \
  --format "bullet_points"
```

### Setting a Maximum Length

This example demonstrates how to limit the total length of the generated prompt.

```bash
python scripts/preview_prompt.py \
  --doc C:\path\to\a\very_long_document.txt \
  --question "What are the initial findings?" \
  --max-length 2048
```
