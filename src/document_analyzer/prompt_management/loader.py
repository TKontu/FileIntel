from pathlib import Path
from ..core.exceptions import ConfigException


class PromptLoader:
    def __init__(self, prompts_dir: Path):
        self.prompts_dir = prompts_dir

    def load_prompt_components(self, task_name: str) -> dict:
        """
        Loads all prompt component templates from a specific task directory.
        """
        task_dir = self.prompts_dir / task_name
        if not task_dir.is_dir():
            raise ConfigException(f"Prompt task directory not found: {task_dir}")

        components = {}
        for file_path in task_dir.glob("*.md"):
            component_name = file_path.stem
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    components[component_name] = f.read()
            except IOError as e:
                raise ConfigException(f"Error reading prompt file {file_path}: {e}")

        # embedding_reference is optional, so we don't check for its existence
        # like we might for other required components.

        if not components:
            raise ConfigException(
                f"No prompt components found in directory: {task_dir}"
            )

        return components
