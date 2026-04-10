"""Typer CLI for ChronoAgent.

Commands:
    serve           Start the FastAPI server with hot reload.
    run-experiment  Execute a signal-validation experiment from a YAML config.
    check-health    Ping the running service health endpoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal, cast

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
    host: Annotated[str, typer.Option(help="Bind host.")] = "0.0.0.0",
    port: Annotated[int, typer.Option(help="Bind port.")] = 8000,
    reload: Annotated[bool, typer.Option(help="Enable hot reload (dev only).")] = True,
    config: Annotated[Path | None, typer.Option(help="Path to YAML config overlay.")] = None,
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
    config: Annotated[Path, typer.Option(help="Path to experiment YAML config.")],
    output: Annotated[
        Path, typer.Option(help="Directory for result artefacts.")
    ] = Path("results/"),
) -> None:
    """Run a signal-validation experiment.

    Reads the YAML config, executes the four-phase
    :class:`~chronoagent.experiments.runner.SignalValidationRunner`, then
    writes plots and a decision-matrix CSV via
    :class:`~chronoagent.experiments.analysis.SignalAnalyzer`.

    Args:
        config: Path to the experiment YAML configuration file.
        output: Directory where result artefacts (plots, CSVs) are written.
    """
    import yaml

    from chronoagent.experiments.analysis import AnalysisConfig, SignalAnalyzer
    from chronoagent.experiments.runner import SignalValidationRunner

    typer.echo(f"Running experiment: {config}")
    typer.echo(f"Output directory:   {output}")

    with open(config) as fh:
        cfg = yaml.safe_load(fh)

    runner_cfg: dict[str, Any] = cfg.get("runner", {})
    analysis_cfg: dict[str, Any] = cfg.get("analysis", {})

    attack_raw = str(runner_cfg.get("attack", "minja"))
    if attack_raw not in ("minja", "agentpoison"):
        typer.echo(f"ERROR: unknown attack type {attack_raw!r}", err=True)
        raise typer.Exit(code=1)
    attack_val = cast(Literal["minja", "agentpoison"], attack_raw)

    runner = SignalValidationRunner.create(
        attack=attack_val,
        n_steps=int(runner_cfg.get("n_steps", 25)),
        n_poison_docs=int(runner_cfg.get("n_poison_docs", 10)),
        n_calibration=int(runner_cfg.get("n_calibration", 10)),
        seed=int(runner_cfg.get("seed", 42)),
        pr_seed=int(runner_cfg.get("pr_seed", 0)),
    )

    typer.echo("Phase A — clean run …")
    typer.echo("Phase B — injecting attack …")
    typer.echo("Phase C — poisoned run …")
    typer.echo("Phase D — computing statistics …")
    result = runner.run()

    analysis_config = AnalysisConfig.from_yaml_section(analysis_cfg)
    analyzer = SignalAnalyzer(result=result, config=analysis_config)

    typer.echo(result.summary())
    typer.echo("")
    typer.echo(analyzer.decision_table())

    analyzer.run(output)
    typer.echo(f"\nArtefacts written to: {output}")


@app.command("check-health")
def check_health(
    url: Annotated[str, typer.Option(help="Base URL of the running service.")] = "http://localhost:8000",
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
        raise typer.Exit(code=1) from None
