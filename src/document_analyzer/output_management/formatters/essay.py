from .base import OutputFormatter

class EssayFormatter(OutputFormatter):
    def format(self, data: dict) -> str:
        """
        Formats the given data as a structured essay.
        Assumes the data contains 'title' and 'sections' keys.
        """
        title = data.get("title", "Untitled Essay")
        sections = data.get("sections", [])

        essay = f"# {title}\n\n"
        for section in sections:
            heading = section.get("heading", "Introduction")
            content = section.get("content", "")
            essay += f"## {heading}\n\n{content}\n\n"
        
        return essay.strip()

