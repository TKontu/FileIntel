from pathlib import Path
from ..core.exceptions import ConfigException

class PromptLoader:
    def __init__(self, prompts_dir: Path):
        self.prompts_dir = prompts_dir

    def load_task_templates(self, task_name: str) -> dict:
        """
        Loads all prompt component templates from a specific task directory.
        """
        task_dir = self.prompts_dir / task_name
        if not task_dir.is_dir():
            raise ConfigException(f"Prompt task directory not found: {task_dir}")

        templates = {}
        for file_path in task_dir.glob("*.md"):
            template_name = file_path.stem  # e.g., "instruction" from "instruction.md"
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    templates[template_name] = f.read()
            except IOError as e:
                raise ConfigException(f"Error reading prompt file {file_path}: {e}")
        
        if not templates:
            raise ConfigException(f"No prompt templates found in directory: {task_dir}")

        return templates
