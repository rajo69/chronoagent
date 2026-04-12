"""Typer CLI for ChronoAgent.

Commands:
    serve                Start the FastAPI server with hot reload.
    run-experiment       Subcommand group for experiment execution.
        phase1           Execute a signal-validation experiment (Phase 1 GO/NO-GO).
        phase10          Execute a single Phase 10 research experiment.
    compare-experiments  Aggregate multi-experiment plots and tables from a results dir.
    check-health         Ping the running service health endpoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal, cast

import httpx
import typer
import uvicorn

app = typer.Typer(
    name="chronoagent",
    help="ChronoAgent - temporal health monitoring for LLM multi-agent systems.",
    add_completion=False,
)

run_experiment_app = typer.Typer(
    name="run-experiment",
    help="Run a signal-validation (phase1) or Phase 10 research experiment.",
    add_completion=False,
    no_args_is_help=True,
)
app.add_typer(run_experiment_app, name="run-experiment")


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


@run_experiment_app.command("phase1")
def run_experiment_phase1(
    config: Annotated[Path, typer.Option(help="Path to Phase 1 signal-validation YAML.")],
    output: Annotated[Path, typer.Option(help="Directory for result artefacts.")] = Path(
        "results/"
    ),
) -> None:
    """Run a Phase 1 signal-validation experiment.

    Reads the Phase 1 YAML (see ``configs/experiments/signal_validation.yaml``),
    executes the four-phase
    :class:`~chronoagent.experiments.runner.SignalValidationRunner`, then
    writes plots and a decision-matrix CSV via
    :class:`~chronoagent.experiments.analysis.SignalAnalyzer`.

    Args:
        config: Path to the Phase 1 YAML configuration file.
        output: Directory where result artefacts (plots, CSVs) are written.
    """
    import yaml

    from chronoagent.experiments.analysis import AnalysisConfig, SignalAnalyzer
    from chronoagent.experiments.runner import SignalValidationRunner

    typer.echo(f"Running phase1 experiment: {config}")
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

    typer.echo("Phase A - clean run ...")
    typer.echo("Phase B - injecting attack ...")
    typer.echo("Phase C - poisoned run ...")
    typer.echo("Phase D - computing statistics ...")
    result = runner.run()

    analysis_config = AnalysisConfig.from_yaml_section(analysis_cfg)
    analyzer = SignalAnalyzer(result=result, config=analysis_config)

    typer.echo(result.summary())
    typer.echo("")
    typer.echo(analyzer.decision_table())

    analyzer.run(output)
    typer.echo(f"\nArtefacts written to: {output}")


@run_experiment_app.command("phase10")
def run_experiment_phase10(
    config: Annotated[Path, typer.Option(help="Path to Phase 10 experiment YAML.")],
    output: Annotated[Path, typer.Option(help="Directory for result artefacts.")] = Path(
        "results/"
    ),
    plots: Annotated[
        bool,
        typer.Option(
            "--plots/--no-plots",
            help="Render the per-experiment signal-drift figure after the run.",
        ),
    ] = True,
    tables: Annotated[
        bool,
        typer.Option(
            "--tables/--no-tables",
            help="Render the single-row main results LaTeX table after the run.",
        ),
    ] = True,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", help="Suppress progress output."),
    ] = False,
) -> None:
    """Run a single Phase 10 research experiment.

    Loads an :class:`~chronoagent.experiments.config_schema.ExperimentConfig`
    from ``config``, runs the experiment via
    :class:`~chronoagent.experiments.experiment_runner.ExperimentRunner`
    (with raw run collection enabled so figures can be rendered), persists
    ``runs.csv`` / ``aggregate.json`` / ``raw/`` under ``output/<name>/``,
    and optionally renders the per-experiment drift figure and headline
    results table.

    Comparison plots and tables that require multiple experiments are
    intentionally skipped here; use ``compare-experiments`` once every
    experiment has been run to generate those aggregate artefacts.

    Args:
        config: Path to a Phase 10 experiment YAML.
        output: Directory where result artefacts are written.
        plots: Render the single-experiment signal-drift figure.
        tables: Render the headline main results LaTeX table.
        quiet: Suppress progress output.
    """
    from pydantic import ValidationError

    from chronoagent.experiments.analysis.plots import plot_signal_drift
    from chronoagent.experiments.analysis.tables import make_main_results_table
    from chronoagent.experiments.config_schema import ExperimentConfig
    from chronoagent.experiments.experiment_runner import (
        ExperimentRunner,
        write_experiment_results,
    )

    def _say(message: str) -> None:
        if not quiet:
            typer.echo(message)

    _say(f"Running phase10 experiment: {config}")
    _say(f"Output directory:   {output}")

    try:
        cfg = ExperimentConfig.from_yaml(config)
    except FileNotFoundError as exc:
        typer.echo(f"ERROR: config file not found: {exc}", err=True)
        raise typer.Exit(code=1) from None
    except ValidationError as exc:
        typer.echo(f"ERROR: invalid experiment config: {exc}", err=True)
        raise typer.Exit(code=1) from None
    except ValueError as exc:
        typer.echo(f"ERROR: invalid experiment config: {exc}", err=True)
        raise typer.Exit(code=1) from None

    _say(f"Experiment name:    {cfg.name}")
    _say(f"num_runs={cfg.num_runs}  num_prs={cfg.num_prs}  seed={cfg.seed}")

    runner = ExperimentRunner(cfg, collect_raw=True)
    _say("Running ...")
    aggregate = runner.run()
    runs_path, json_path = write_experiment_results(
        aggregate,
        output,
        raw_runs=runner.raw_runs,
    )
    _say(f"Wrote runs:      {runs_path}")
    _say(f"Wrote aggregate: {json_path}")

    if plots:
        try:
            plot_paths = plot_signal_drift(output, cfg.name, run_index=0)
        except (FileNotFoundError, ValueError) as exc:
            typer.echo(f"WARNING: signal-drift plot skipped: {exc}", err=True)
        else:
            for path in plot_paths:
                _say(f"Wrote figure:    {path}")
        _say(
            "Note: comparison figures (health, AWT box, allocation, ROC, ablation) "
            "require multiple experiments; run `chronoagent compare-experiments` after "
            "all experiments have completed."
        )

    if tables:
        try:
            table_path = make_main_results_table(output, [cfg.name])
        except (FileNotFoundError, ValueError) as exc:
            typer.echo(f"WARNING: main results table skipped: {exc}", err=True)
        else:
            _say(f"Wrote table:     {table_path}")
        _say(
            "Note: ablation and signal-validation tables require multiple experiments; "
            "use `chronoagent compare-experiments` for the full paper table set."
        )

    _say(f"\nArtefacts written to: {output / cfg.name}")


@app.command("compare-experiments")
def compare_experiments(
    output: Annotated[
        Path,
        typer.Option(help="Results directory that already contains per-experiment subdirs."),
    ],
    experiments: Annotated[
        list[str],
        typer.Option(
            "--experiment",
            "-e",
            help="Experiment name to include (repeat the flag for each experiment).",
        ),
    ],
    full_system: Annotated[
        str | None,
        typer.Option(
            help=(
                "Experiment name to use as the full-system row in the ablation "
                "table (defaults to the first --experiment value)."
            )
        ),
    ] = None,
    ablation: Annotated[
        list[str] | None,
        typer.Option(
            "--ablation",
            "-a",
            help="Ablation experiment name (repeat the flag for each ablation).",
        ),
    ] = None,
    drift_experiment: Annotated[
        str | None,
        typer.Option(help="Which experiment's run 0 is used for the signal-drift figure."),
    ] = None,
    plots: Annotated[
        bool,
        typer.Option("--plots/--no-plots", help="Render the six comparison figures."),
    ] = True,
    tables: Annotated[
        bool,
        typer.Option("--tables/--no-tables", help="Render the main results + ablation tables."),
    ] = True,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", help="Suppress progress output."),
    ] = False,
) -> None:
    """Aggregate multi-experiment plots and tables from a results directory.

    Walks ``output/`` looking for the named experiment subdirectories (each
    one created by a previous ``run-experiment phase10`` invocation) and
    hands them to
    :func:`~chronoagent.experiments.analysis.plots.generate_all_plots` and
    :func:`~chronoagent.experiments.analysis.tables.generate_all_tables`.

    Separate from ``run-experiment phase10`` because comparison artefacts
    need multiple experiments to be meaningful; the typical research
    workflow is "run all experiments, then compare".

    Args:
        output: Results directory containing per-experiment subdirectories.
        experiments: Experiment names to include in every comparison
            figure and the main results table. Order is preserved.
        full_system: Experiment name to use as the full-system row in the
            ablation table; defaults to the first ``--experiment`` value.
        ablation: Experiment names of ablation rows in the ablation table.
            Skipped if empty.
        drift_experiment: Experiment whose run 0 drives the signal-drift
            figure; defaults to the first ``--experiment`` value.
        plots: Render the six comparison figures.
        tables: Render the main results and (optional) ablation tables.
        quiet: Suppress progress output.
    """
    from chronoagent.experiments.analysis.plots import generate_all_plots
    from chronoagent.experiments.analysis.tables import (
        make_ablation_table,
        make_latency_table,
        make_main_results_table,
    )

    def _say(message: str) -> None:
        if not quiet:
            typer.echo(message)

    if not experiments:
        typer.echo("ERROR: at least one --experiment value is required", err=True)
        raise typer.Exit(code=1)

    if not output.exists():
        typer.echo(f"ERROR: output directory does not exist: {output}", err=True)
        raise typer.Exit(code=1)

    full_system_name = full_system if full_system is not None else experiments[0]
    ablation_names = list(ablation) if ablation else []
    drift_target = drift_experiment if drift_experiment is not None else experiments[0]

    _say(f"Comparing experiments: {', '.join(experiments)}")
    _say(f"Results directory:     {output}")

    if plots:
        _say("Rendering comparison figures ...")
        try:
            plot_paths = generate_all_plots(
                output,
                experiments,
                drift_experiment=drift_target,
            )
        except (FileNotFoundError, ValueError) as exc:
            typer.echo(f"ERROR: plot generation failed: {exc}", err=True)
            raise typer.Exit(code=1) from None
        for path in plot_paths:
            _say(f"Wrote figure: {path}")

    if tables:
        _say("Rendering tables ...")
        try:
            main_path = make_main_results_table(output, experiments)
        except (FileNotFoundError, ValueError) as exc:
            typer.echo(f"ERROR: main results table failed: {exc}", err=True)
            raise typer.Exit(code=1) from None
        _say(f"Wrote table:  {main_path}")

        try:
            latency_path = make_latency_table(output, experiments)
        except (FileNotFoundError, ValueError) as exc:
            typer.echo(f"ERROR: latency table failed: {exc}", err=True)
            raise typer.Exit(code=1) from None
        _say(f"Wrote table:  {latency_path}")

        if ablation_names:
            try:
                ablation_path = make_ablation_table(
                    output,
                    full_system_name,
                    ablation_names,
                )
            except (FileNotFoundError, ValueError) as exc:
                typer.echo(f"ERROR: ablation table failed: {exc}", err=True)
                raise typer.Exit(code=1) from None
            _say(f"Wrote table:  {ablation_path}")
        else:
            _say("Skipping ablation table (no --ablation values supplied).")

    _say(f"\nComparison artefacts written under: {output}")


@app.command("check-health")
def check_health(
    url: Annotated[
        str, typer.Option(help="Base URL of the running service.")
    ] = "http://localhost:8000",
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
