from pathlib import Path
from ..core.exceptions import ConfigException

class PromptLoader:
    def __init__(self, prompts_dir: Path):
        self.prompts_dir = prompts_dir

    def load_prompt(self, name: str) -> str:
        """
        Loads a prompt from a markdown file.
        """
        prompt_path = self.prompts_dir / f"{name}.md"
        if not prompt_path.exists():
            raise ConfigException(f"Prompt file not found: {prompt_path}")

        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
