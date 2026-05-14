"""DocKar CLI entrypoints."""

from pathlib import Path
from typing import Annotated

import typer

from dockar.config import ConfigError, DocKarConfig, load_config
from dockar.logging import configure_logging

app = typer.Typer(help="DocKar document extraction toolkit.")


@app.callback()
def main() -> None:
    """DocKar command-line interface."""


@app.command()
def config_check(
    config: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
) -> None:
    """Validate a DocKar YAML configuration file."""

    try:
        dockar_config = load_config(config)
    except ConfigError as exc:
        raise typer.BadParameter(str(exc), param_hint="config") from exc

    configure_logging(dockar_config.logging)
    typer.echo(dockar_config.model_dump_json(indent=2))


@app.command()
def run(
    docs: Annotated[Path, typer.Option(exists=True, help="Directory containing input documents.")],
    schema: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, help="Target schema JSON file."),
    ],
    labels: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, help="Golden labels JSON file."),
    ],
    config: Annotated[
        Path | None,
        typer.Option(exists=True, dir_okay=False, help="YAML config file."),
    ] = None,
    budget: Annotated[
        float | None,
        typer.Option(min=0.01, help="Maximum spend in USD."),
    ] = None,
    iterations: Annotated[
        int | None,
        typer.Option(min=1, help="Maximum optimization iterations."),
    ] = None,
) -> None:
    """Prepare an extraction run.

    The full optimization loop is intentionally behind interfaces; this command currently
    validates inputs and resolves effective configuration.
    """

    try:
        dockar_config = load_config(config) if config else DocKarConfig()
    except ConfigError as exc:
        raise typer.BadParameter(str(exc), param_hint="--config") from exc

    if budget is not None:
        dockar_config.loop.budget_usd = budget
    if iterations is not None:
        dockar_config.loop.max_iterations = iterations

    configure_logging(dockar_config.logging)
    typer.echo(dockar_config.model_dump_json(indent=2))
