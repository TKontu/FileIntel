import typer
from rich.console import Console

from .client import FileIntelAPI, print_json

app = typer.Typer()
api = FileIntelAPI()
console = Console()


@app.command("status")
def get_job_status(
    job_id: str = typer.Argument(..., help="The ID of the job to check.")
):
    """Gets the status of a job."""
    try:
        status = api.get_job_status(job_id)
        print_json(status)
    except Exception:
        raise typer.Exit(1)


@app.command("result")
def get_job_result(
    job_id: str = typer.Argument(
        ..., help="The ID of the job to retrieve the result for."
    ),
    md: bool = typer.Option(
        False, "--md", help="Render the result as Markdown in the console."
    ),
):
    """Gets the result of a job."""
    try:
        result = api.get_job_result(job_id, markdown=md)
        if md:
            from rich.markdown import Markdown

            console.print(Markdown(result), justify="left")
        else:
            print_json(result)
    except Exception:
        raise typer.Exit(1)
