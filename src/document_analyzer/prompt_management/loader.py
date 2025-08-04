from pathlib import Path
import mistune
from ..core.exceptions import ConfigException

class PromptLoader:
    def __init__(self, prompts_dir: Path):
        self.prompts_dir = prompts_dir
        self.markdown_parser = mistune.create_markdown(renderer=None)

    def load_prompt(self, name: str, version: str = None) -> str:
        """
        Loads the content of a prompt from a markdown file, excluding the first heading.
        """
        if version:
            prompt_path = self.prompts_dir / version / f"{name}.md"
        else:
            prompt_path = self.prompts_dir / f"{name}.md"

        if not prompt_path.exists():
            raise ConfigException(f"Prompt file not found: {prompt_path}")

        with open(prompt_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Find the first line that is a heading
        first_heading_index = -1
        for i, line in enumerate(lines):
            if line.strip().startswith("#"):
                first_heading_index = i
                break

        if first_heading_index == -1:
            raise ConfigException(f"Prompt '{name}' has no heading. Please add at least one heading.")

        # Return the content *after* the first heading
        return "".join(lines[first_heading_index + 1:]).strip()
