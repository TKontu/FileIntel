import typer

from . import collections, documents, jobs, query, analyze

app = typer.Typer(
    name="fileintel",
    help="A CLI for interacting with the FileIntel API.",
    add_completion=False,
)

app.add_typer(collections.app, name="collections", help="Manage collections.")
app.add_typer(
    documents.app, name="documents", help="Manage documents within collections."
)
app.add_typer(jobs.app, name="jobs", help="Check the status and results of jobs.")
app.add_typer(query.app, name="query", help="Ask questions using RAG.")
app.add_typer(analyze.app, name="analyze", help="Perform template-driven analysis.")

if __name__ == "__main__":
    app()
