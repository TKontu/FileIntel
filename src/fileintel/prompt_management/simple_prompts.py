"""
Simplified prompt management for FileIntel.

Replaces complex template engine classes with basic template functions.
"""

import os
import glob
from pathlib import Path
from typing import Dict, Any, Optional
from jinja2 import Template, TemplateError

# Token estimation constants
CHARS_PER_TOKEN_ESTIMATE = 4  # Average characters per token heuristic
TRUNCATION_MESSAGE = "\n\n[Content truncated due to length]"


def load_prompt_template(template_path: str) -> str:
    """
    Load prompt template from file.

    Args:
        template_path: Path to template file

    Returns:
        Template content

    Raises:
        IOError: If file cannot be read
    """
    try:
        with open(template_path, "r", encoding="utf-8") as f:
            return f.read()
    except IOError as e:
        raise IOError(f"Failed to load template {template_path}: {e}")


def load_prompt_components(prompt_dir: str) -> Dict[str, str]:
    """
    Load all markdown files from prompt directory.

    Args:
        prompt_dir: Directory containing prompt files

    Returns:
        Dictionary mapping filenames to content
    """
    components = {}

    if not os.path.exists(prompt_dir):
        return components

    for md_file in glob.glob(os.path.join(prompt_dir, "*.md")):
        filename = Path(md_file).stem
        try:
            components[filename] = load_prompt_template(md_file)
        except IOError:
            # Skip files that can't be read
            continue

    return components


def validate_template_context(context: Dict[str, Any]) -> None:
    """
    Validate template context for common issues.

    Args:
        context: Template context to validate

    Raises:
        ValueError: If context contains invalid values
    """
    if not isinstance(context, dict):
        raise ValueError(f"Context must be a dictionary, got {type(context)}")

    for key, value in context.items():
        if not isinstance(key, str):
            raise ValueError(
                f"Context keys must be strings, got {type(key)} for key: {key}"
            )

        # Check for problematic values that Jinja2 can't handle
        if callable(value) and not hasattr(value, "__name__"):
            raise ValueError(
                f"Context value for '{key}' is an unnamed callable, which may cause rendering issues"
            )


def render_template(template_content: str, context: Dict[str, Any]) -> str:
    """
    Render Jinja2 template with context.

    Args:
        template_content: Template string
        context: Template variables

    Returns:
        Rendered template

    Raises:
        TemplateError: If template rendering fails
        ValueError: If context is malformed
    """
    # Validate context before rendering
    validate_template_context(context)

    try:
        template = Template(template_content)
        return template.render(**context)
    except TemplateError as e:
        raise TemplateError(f"Template rendering failed: {e}")
    except (TypeError, AttributeError) as e:
        raise TemplateError(f"Template rendering failed due to context error: {e}")


def validate_template(template_content: str) -> bool:
    """
    Validate that template content is valid Jinja2.

    Args:
        template_content: Template string

    Returns:
        True if valid, False otherwise
    """
    if not template_content or not isinstance(template_content, str):
        return False

    try:
        Template(template_content)
        return True
    except (TemplateError, TypeError, ValueError):
        return False


def estimate_token_count(text: str) -> int:
    """
    Rough estimate of token count for text.

    Args:
        text: Text to estimate

    Returns:
        Estimated token count
    """
    return len(text) // CHARS_PER_TOKEN_ESTIMATE


def truncate_text_to_tokens(text: str, max_tokens: int) -> str:
    """
    Truncate text to approximate token limit.

    Args:
        text: Text to truncate
        max_tokens: Maximum token count

    Returns:
        Truncated text
    """
    if not text:
        return text

    estimated_tokens = estimate_token_count(text)

    if estimated_tokens <= max_tokens:
        return text

    # Calculate approximate character limit
    char_limit = max_tokens * CHARS_PER_TOKEN_ESTIMATE

    if len(text) <= char_limit:
        return text

    # Truncate and add indicator
    truncated = text[: char_limit - len(TRUNCATION_MESSAGE)]

    # Try to break at word boundary
    last_space = truncated.rfind(" ")
    if last_space > char_limit * 0.8:  # Only break at word if close to limit
        truncated = truncated[:last_space]

    return truncated + TRUNCATION_MESSAGE


def compose_prompt(
    template_content: str, context: Dict[str, Any], max_tokens: Optional[int] = None
) -> str:
    """
    Compose prompt by rendering template and optionally truncating.

    Args:
        template_content: Jinja2 template
        context: Template variables
        max_tokens: Optional token limit

    Returns:
        Composed prompt

    Raises:
        TemplateError: If template rendering fails
    """
    rendered = render_template(template_content, context)

    if max_tokens:
        rendered = truncate_text_to_tokens(rendered, max_tokens)

    return rendered


def build_context(**kwargs) -> Dict[str, Any]:
    """
    Build template context from keyword arguments.

    Args:
        **kwargs: Context variables

    Returns:
        Context dictionary
    """
    return {k: v for k, v in kwargs.items() if v is not None}


def load_and_render_prompt(
    template_path: str, context: Dict[str, Any], max_tokens: Optional[int] = None
) -> str:
    """
    Load template from file and render with context.

    Args:
        template_path: Path to template file
        context: Template variables
        max_tokens: Optional token limit

    Returns:
        Rendered prompt

    Raises:
        IOError: If file cannot be read
        TemplateError: If template rendering fails
    """
    template_content = load_prompt_template(template_path)
    return compose_prompt(template_content, context, max_tokens)


def create_simple_prompt(
    instruction: str,
    context: str = "",
    question: str = "",
    format_instructions: str = "",
) -> str:
    """
    Create simple prompt without templates.

    Args:
        instruction: Main instruction
        context: Additional context
        question: Question to answer
        format_instructions: Output format instructions

    Returns:
        Composed prompt
    """
    parts = []

    if instruction:
        parts.append(instruction)

    if context:
        parts.append(f"Context:\n{context}")

    if question:
        parts.append(f"Question: {question}")

    if format_instructions:
        parts.append(f"Format: {format_instructions}")

    return "\n\n".join(parts)


# Common prompt templates as constants
ANALYSIS_TEMPLATE = """
Analyze the following document content and provide insights.

Context:
{{ context }}

Focus Areas:
{% if focus_areas %}
{% for area in focus_areas %}
- {{ area }}
{% endfor %}
{% endif %}

Question: {{ question }}

Please provide your analysis in {{ format | default('markdown') }} format.
"""

QUESTION_TEMPLATE = """
Answer the following question based on the provided context.

Context:
{{ context }}

Question: {{ question }}

{% if format_instructions %}
{{ format_instructions }}
{% endif %}
"""

# Template registry
PROMPT_TEMPLATES = {
    "analysis": ANALYSIS_TEMPLATE,
    "question": QUESTION_TEMPLATE,
}
