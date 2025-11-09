"""Unit tests for AnswerFormatManager."""

import pytest
import tempfile
import os
from pathlib import Path
from typing import List


class TestAnswerFormatManager:
    """Test cases for AnswerFormatManager."""

    @pytest.fixture
    def temp_formats_dir(self):
        """Create a temporary directory with test format templates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            formats_dir = Path(tmpdir) / "formats"
            formats_dir.mkdir()

            # Create test format files
            (formats_dir / "answer_format_test_format.md").write_text(
                "# Test Format\n\nProvide answer as a test."
            )
            (formats_dir / "answer_format_another_format.md").write_text(
                "# Another Format\n\nProvide answer differently."
            )

            yield formats_dir

    @pytest.fixture
    def manager(self, temp_formats_dir):
        """Create AnswerFormatManager instance with test directory."""
        from fileintel.prompt_management import AnswerFormatManager
        return AnswerFormatManager(temp_formats_dir)

    def test_initialization(self, temp_formats_dir):
        """Test AnswerFormatManager initialization."""
        from fileintel.prompt_management import AnswerFormatManager

        manager = AnswerFormatManager(temp_formats_dir)
        assert manager is not None
        assert manager.formats_dir == Path(temp_formats_dir)

    def test_initialization_with_string_path(self, temp_formats_dir):
        """Test AnswerFormatManager accepts string path."""
        from fileintel.prompt_management import AnswerFormatManager

        manager = AnswerFormatManager(str(temp_formats_dir))
        assert manager.formats_dir == Path(temp_formats_dir)

    def test_scan_available_formats(self, manager):
        """Test scanning for available format templates."""
        formats = manager.list_available_formats()

        # Should include default plus scanned formats
        assert "default" in formats
        assert "test_format" in formats
        assert "another_format" in formats
        assert len(formats) == 3

    def test_get_default_format(self, manager):
        """Test getting default format returns empty string."""
        template = manager.get_format_template("default")
        assert template == ""

    def test_get_valid_format(self, manager):
        """Test getting valid format template."""
        template = manager.get_format_template("test_format")
        assert template == "# Test Format\n\nProvide answer as a test."

    def test_get_another_valid_format(self, manager):
        """Test getting another valid format template."""
        template = manager.get_format_template("another_format")
        assert template == "# Another Format\n\nProvide answer differently."

    def test_get_invalid_format_raises_error(self, manager):
        """Test getting invalid format raises ValueError."""
        from fileintel.prompt_management import AnswerFormatManager

        with pytest.raises(ValueError, match="Unknown answer format"):
            manager.get_format_template("nonexistent_format")

    def test_invalid_format_error_message(self, manager):
        """Test error message includes available formats."""
        from fileintel.prompt_management import AnswerFormatManager

        try:
            manager.get_format_template("invalid")
        except ValueError as e:
            error_msg = str(e)
            assert "invalid" in error_msg
            assert "test_format" in error_msg
            assert "another_format" in error_msg

    def test_caching_works(self, manager):
        """Test that templates are cached after first load."""
        # First call loads from file
        template1 = manager.get_format_template("test_format")

        # Second call should use cache
        template2 = manager.get_format_template("test_format")

        assert template1 == template2
        assert "test_format" in manager._cache

    def test_validate_format_default(self, manager):
        """Test validating default format."""
        assert manager.validate_format("default") is True

    def test_validate_format_valid(self, manager):
        """Test validating existing format."""
        assert manager.validate_format("test_format") is True

    def test_validate_format_invalid(self, manager):
        """Test validating non-existent format."""
        assert manager.validate_format("nonexistent") is False

    def test_reload_formats(self, manager, temp_formats_dir):
        """Test reloading formats clears cache and rescans."""
        # Load a format to populate cache
        manager.get_format_template("test_format")
        assert len(manager._cache) > 0

        # Add a new format file
        (temp_formats_dir / "answer_format_new_format.md").write_text(
            "# New Format\n\nNew content."
        )

        # Reload formats
        manager.reload_formats()

        # Cache should be cleared
        assert len(manager._cache) == 0

        # New format should be available
        assert "new_format" in manager.list_available_formats()

    def test_reload_formats_preserves_functionality(self, manager):
        """Test that reload doesn't break existing functionality."""
        # Get a format before reload
        template_before = manager.get_format_template("test_format")

        # Reload
        manager.reload_formats()

        # Get the same format after reload
        template_after = manager.get_format_template("test_format")

        assert template_before == template_after

    def test_empty_directory_handling(self):
        """Test handling of empty formats directory."""
        from fileintel.prompt_management import AnswerFormatManager

        with tempfile.TemporaryDirectory() as tmpdir:
            empty_dir = Path(tmpdir) / "empty"
            empty_dir.mkdir()

            manager = AnswerFormatManager(empty_dir)
            formats = manager.list_available_formats()

            # Should only have default
            assert formats == ["default"]

    def test_nonexistent_directory_handling(self):
        """Test handling of non-existent directory."""
        from fileintel.prompt_management import AnswerFormatManager

        with tempfile.TemporaryDirectory() as tmpdir:
            nonexistent_dir = Path(tmpdir) / "nonexistent"

            manager = AnswerFormatManager(nonexistent_dir)
            formats = manager.list_available_formats()

            # Should only have default
            assert formats == ["default"]

    def test_file_read_error_handling(self, manager, temp_formats_dir):
        """Test handling of file read errors."""
        # Create a format file with restricted permissions
        restricted_file = temp_formats_dir / "answer_format_restricted.md"
        restricted_file.write_text("content")

        # Make file unreadable
        os.chmod(restricted_file, 0o000)

        # Reload to pick up new file
        manager.reload_formats()

        # Try to get the restricted format
        with pytest.raises(IOError):
            manager.get_format_template("restricted")

        # Cleanup: restore permissions
        os.chmod(restricted_file, 0o644)

    def test_format_filename_parsing(self, temp_formats_dir):
        """Test correct parsing of format names from filenames."""
        from fileintel.prompt_management import AnswerFormatManager

        # Create formats with various naming patterns
        (temp_formats_dir / "answer_format_single_word.md").write_text("content")
        (temp_formats_dir / "answer_format_multiple_words.md").write_text("content")
        (temp_formats_dir / "answer_format_with_underscore_parts.md").write_text("content")

        manager = AnswerFormatManager(temp_formats_dir)
        formats = manager.list_available_formats()

        assert "single_word" in formats
        assert "multiple_words" in formats
        assert "with_underscore_parts" in formats

    def test_non_markdown_files_ignored(self, temp_formats_dir):
        """Test that non-.md files are ignored."""
        from fileintel.prompt_management import AnswerFormatManager

        # Create non-markdown files
        (temp_formats_dir / "answer_format_test.txt").write_text("content")
        (temp_formats_dir / "answer_format_test.json").write_text("{}")
        (temp_formats_dir / "not_answer_format.md").write_text("content")

        manager = AnswerFormatManager(temp_formats_dir)
        formats = manager.list_available_formats()

        # Should not include these files
        assert "test" not in [f for f in formats if f != "default"]

    def test_unicode_content_handling(self, temp_formats_dir):
        """Test handling of Unicode content in format files."""
        from fileintel.prompt_management import AnswerFormatManager

        unicode_content = "# Format\n\nProvide answer with émojis 🎉 and ñoñ-ASCII ℂ𝕙𝕒𝕣𝕤"
        (temp_formats_dir / "answer_format_unicode.md").write_text(
            unicode_content, encoding="utf-8"
        )

        manager = AnswerFormatManager(temp_formats_dir)
        template = manager.get_format_template("unicode")

        assert template == unicode_content

    def test_large_template_handling(self, temp_formats_dir):
        """Test handling of large template files."""
        from fileintel.prompt_management import AnswerFormatManager

        # Create a large template (10KB)
        large_content = "# Large Format\n\n" + ("Content " * 1000)
        (temp_formats_dir / "answer_format_large.md").write_text(large_content)

        manager = AnswerFormatManager(temp_formats_dir)
        template = manager.get_format_template("large")

        assert len(template) > 5000
        assert template == large_content


class TestAnswerFormatManagerIntegration:
    """Integration tests for AnswerFormatManager with real format files."""

    def test_real_format_files_exist(self):
        """Test that real format files exist in prompts/examples/."""
        formats_dir = Path(__file__).parent.parent.parent / "prompts" / "examples"

        # Check directory exists
        assert formats_dir.exists(), f"Formats directory not found: {formats_dir}"

        # Check for expected format files
        expected_formats = [
            "answer_format_single_paragraph.md",
            "answer_format_table.md",
            "answer_format_list.md",
            "answer_format_json.md",
            "answer_format_essay.md",
            "answer_format_markdown.md",
        ]

        for format_file in expected_formats:
            file_path = formats_dir / format_file
            assert file_path.exists(), f"Format file not found: {format_file}"

    def test_load_real_single_paragraph_format(self):
        """Test loading real single_paragraph format."""
        from fileintel.prompt_management import AnswerFormatManager

        formats_dir = Path(__file__).parent.parent.parent / "prompts" / "examples"

        if not formats_dir.exists():
            pytest.skip("Prompts directory not found")

        manager = AnswerFormatManager(formats_dir)
        template = manager.get_format_template("single_paragraph")

        # Should contain expected content
        assert "single" in template.lower() or "paragraph" in template.lower()
        assert len(template) > 0

    def test_all_real_formats_loadable(self):
        """Test that all real format files are loadable."""
        from fileintel.prompt_management import AnswerFormatManager

        formats_dir = Path(__file__).parent.parent.parent / "prompts" / "examples"

        if not formats_dir.exists():
            pytest.skip("Prompts directory not found")

        manager = AnswerFormatManager(formats_dir)
        formats = manager.list_available_formats()

        # Try to load each format (except default)
        for format_name in formats:
            if format_name != "default":
                template = manager.get_format_template(format_name)
                assert len(template) > 0, f"Format {format_name} is empty"
