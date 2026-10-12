from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer

from pitchlab.core.config import load_config
from pitchlab.m6_detect_track.pipeline import run_m6_pipeline

app = typer.Typer(
    help="PitchLab M6 Detection & Tracking CLI.",
    add_completion=False,
)


@app.command()
def run(
    video: Annotated[
        Path,
        typer.Option(
            "--video",
            "-v",
            help="Path to input video file.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    config_file: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="Path to YAML configuration file.",
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Path to output tracks Parquet file.",
            file_okay=True,
            dir_okay=False,
        ),
    ] = None,
    max_frames: Annotated[
        int | None,
        typer.Option(
            "--max-frames",
            help="Maximum frames to process.",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            help="Enable debug logging level.",
        ),
    ] = False,
) -> None:
    """Runs M6 Detection and Tracking pipeline from CLI."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    root_cfg = load_config(config_file)
    m6_cfg = root_cfg.m6

    out_file = output or Path("runs/detect_track/tracks.parquet")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    tracks = run_m6_pipeline(
        video_path=video,
        cfg=m6_cfg,
        output_path=out_file,
        max_frames=max_frames,
    )

    typer.echo(f"Successfully processed {len(tracks)} track records -> {out_file}")


if __name__ == "__main__":
    app()
