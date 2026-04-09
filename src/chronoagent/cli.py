"""Typer CLI for ChronoAgent.

Commands:
    serve           Start the FastAPI server with hot reload.
    run-experiment  Execute a signal-validation experiment from a YAML config.
    check-health    Ping the running service health endpoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import httpx
import typer
import uvicorn

app = typer.Typer(
    name="chronoagent",
    help="ChronoAgent — temporal health monitoring for LLM multi-agent systems.",
    add_completion=False,
)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Bind host."),
    port: int = typer.Option(8000, help="Bind port."),
    reload: bool = typer.Option(True, help="Enable hot reload (dev only)."),
    config: Optional[Path] = typer.Option(None, help="Path to YAML config overlay."),
) -> None:
    """Start the FastAPI server.

    Args:
        host: Interface address to bind.
        port: TCP port to listen on.
        reload: Whether to enable Uvicorn auto-reload.
        config: Optional YAML config overlay path.
    """
    if config is not None:
        import os

        os.environ["CHRONO_CONFIG_PATH"] = str(config)

    uvicorn.run(
        "chronoagent.main:app",
        host=host,
        port=port,
        reload=reload,
    )


@app.command("run-experiment")
def run_experiment(
    config: Path = typer.Option(..., help="Path to experiment YAML config."),
    output: Path = typer.Option(Path("results/"), help="Directory for result artefacts."),
) -> None:
    """Run a signal-validation experiment.

    Args:
        config: Path to the experiment YAML configuration file.
        output: Directory where result artefacts (plots, CSVs) are written.
    """
    typer.echo(f"Running experiment: {config}")
    typer.echo(f"Output directory:   {output}")
    typer.echo("Experiment runner not yet implemented (Phase 1).")
    raise typer.Exit(code=0)


@app.command("check-health")
def check_health(
    url: str = typer.Option("http://localhost:8000", help="Base URL of the running service."),
) -> None:
    """Ping the /health endpoint of a running ChronoAgent service.

    Args:
        url: Base URL of the ChronoAgent service.
    """
    try:
        response = httpx.get(f"{url}/health", timeout=5.0)
        response.raise_for_status()
        data = response.json()
        typer.echo(f"OK  status={data['status']}  version={data['version']}")
    except httpx.HTTPError as exc:
        typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(code=1)
