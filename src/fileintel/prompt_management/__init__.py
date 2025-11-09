"""
Simplified prompt management for FileIntel.

This module provides simple functions for template loading and rendering.
"""

# Import all functions from the simplified module
from .simple_prompts import (
    load_prompt_template,
    load_prompt_components,
    render_template,
    validate_template,
    validate_template_context,
    compose_prompt,
    load_and_render_prompt,
    create_simple_prompt,
    build_context,
    ANALYSIS_TEMPLATE,
    QUESTION_TEMPLATE,
    PROMPT_TEMPLATES,
    CHARS_PER_TOKEN_ESTIMATE,
    TRUNCATION_MESSAGE,
)

# Import answer format manager
from .format_manager import AnswerFormatManager

# Export modern simple functions only
__all__ = [
    "load_prompt_template",
    "load_prompt_components",
    "render_template",
    "validate_template",
    "validate_template_context",
    "compose_prompt",
    "load_and_render_prompt",
    "create_simple_prompt",
    "build_context",
    "ANALYSIS_TEMPLATE",
    "QUESTION_TEMPLATE",
    "PROMPT_TEMPLATES",
    "CHARS_PER_TOKEN_ESTIMATE",
    "TRUNCATION_MESSAGE",
    "AnswerFormatManager",
]
