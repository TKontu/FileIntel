from jinja2 import Environment, FileSystemLoader, BaseLoader
from pathlib import Path

class TemplateEngine:
    def __init__(self, templates_dir: Path = None):
        if templates_dir:
            self.env = Environment(loader=FileSystemLoader(templates_dir))
        else:
            self.env = Environment(loader=BaseLoader())

    def render(self, template_name: str, **kwargs) -> str:
        """
        Renders a template from a file with the given context.
        """
        template = self.env.get_template(template_name)
        return template.render(**kwargs)

    def render_string(self, template_string: str, **kwargs) -> str:
        """
        Renders a template from a string with the given context.
        """
        template = self.env.from_string(template_string)
        return template.render(**kwargs)
