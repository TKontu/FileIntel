from jinja2 import Environment, FileSystemLoader, BaseLoader
from jinja2.exceptions import TemplateSyntaxError
from pathlib import Path
from ..core.exceptions import ConfigException


class TemplateEngine:
    def __init__(self, templates_dir: Path = None):
        if templates_dir:
            self.env = Environment(loader=FileSystemLoader(templates_dir))
        else:
            self.env = Environment(loader=BaseLoader())

    def validate_template(self, template_string: str):
        """
        Validates the syntax of a Jinja2 template.
        Raises a ConfigException if the syntax is invalid.
        """
        try:
            self.env.parse(template_string)
        except TemplateSyntaxError as e:
            raise ConfigException(f"Invalid template syntax: {e}")

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
        self.validate_template(template_string)
        template = self.env.from_string(template_string)
        return template.render(**kwargs)
