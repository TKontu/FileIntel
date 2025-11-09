"""
Answer format management for RAG systems.

Provides loading and validation of answer format templates.
"""

import os
import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class AnswerFormatManager:
    """Manages answer format templates for RAG queries."""

    def __init__(self, formats_dir: Path | str):
        """
        Initialize the format manager.

        Args:
            formats_dir: Directory containing format template files
        """
        self.formats_dir = Path(formats_dir)
        self._cache: dict[str, str] = {}  # Cache loaded templates
        self._available_formats = self._scan_available_formats()

        logger.info(
            f"AnswerFormatManager initialized with {len(self._available_formats)} formats"
        )

    def _scan_available_formats(self) -> List[str]:
        """
        Scan formats directory for available format templates.

        Returns:
            List of format names (without .md extension)
        """
        if not self.formats_dir.exists():
            logger.warning(f"Formats directory not found: {self.formats_dir}")
            return []

        formats = []
        for file_path in self.formats_dir.glob("answer_format_*.md"):
            # Extract format name: answer_format_single_paragraph.md -> single_paragraph
            format_name = file_path.stem.replace("answer_format_", "")
            formats.append(format_name)

        logger.debug(f"Found {len(formats)} format templates: {formats}")
        return formats

    def get_format_template(self, format_name: str) -> str:
        """
        Get template content for specified format.

        Args:
            format_name: Name of the format (e.g., "single_paragraph")

        Returns:
            Template content as string

        Raises:
            ValueError: If format does not exist
        """
        if format_name == "default":
            return ""  # No specific format constraint

        # Check cache first
        if format_name in self._cache:
            logger.debug(f"Returning cached format template: {format_name}")
            return self._cache[format_name]

        # Validate format exists
        if format_name not in self._available_formats:
            raise ValueError(
                f"Unknown answer format: '{format_name}'. "
                f"Available formats: {', '.join(self._available_formats)}"
            )

        # Load template file
        template_path = self.formats_dir / f"answer_format_{format_name}.md"

        try:
            with open(template_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Cache the template
            self._cache[format_name] = content

            logger.debug(f"Loaded format template: {format_name} ({len(content)} chars)")
            return content

        except IOError as e:
            raise IOError(f"Failed to load format template {format_name}: {e}")

    def list_available_formats(self) -> List[str]:
        """
        Return all available format names.

        Returns:
            List of format names including 'default'
        """
        return ["default"] + self._available_formats

    def validate_format(self, format_name: str) -> bool:
        """
        Check if format exists.

        Args:
            format_name: Name of the format to validate

        Returns:
            True if format exists, False otherwise
        """
        if format_name == "default":
            return True

        return format_name in self._available_formats

    def reload_formats(self) -> None:
        """
        Reload available formats and clear cache.

        Useful for development or when formats are added/modified.
        """
        self._cache.clear()
        self._available_formats = self._scan_available_formats()
        logger.info(f"Reloaded {len(self._available_formats)} format templates")
