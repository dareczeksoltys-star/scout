"""Command-line interface (CLI) for PITCHLAB.

Follows command specifications from Section 9 of docs/PLAN.md:
- pitchlab shots <video>
- pitchlab calib <video> --backend <name>
- pitchlab footprint <video>
- pitchlab eval --backends <b1,b2>
- pitchlab track <video>
- pitchlab world <video>
- pitchlab events <match>
- pitchlab options <match>
- pitchlab learn --matches <m1,m2>
- pitchlab tag <match>
- pitchlab run-all <video>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from pitchlab.core.config import AppConfig
from pitchlab.core.pitch_template import PitchTemplate


def cmd_shots(args: argparse.Namespace, cfg: AppConfig) -> int:
    print(f"[PITCHLAB] Running M1 (shots) on: {args.video} [mode={cfg.env_mode}]")
    return 0


def cmd_calib(args: argparse.Namespace, cfg: AppConfig) -> int:
    from pitchlab.m2_calib.pipeline import run_calibration

    backend = args.backend or cfg.calibration.default_backend
    video_path = Path(args.video)
    stem = video_path.stem
    out_dir = Path("data/out") / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    out_parquet = out_dir / "calib.parquet"

    if out_parquet.exists() and not args.force:
        print(f"[PITCHLAB] Calibration output already exists at {out_parquet}. Use --force to recompute.")
        return 0

    print(f"[PITCHLAB] Calibrating {args.video} with backend '{backend}' [mode={cfg.env_mode}]...")
    results = run_calibration(
        video_path=args.video,
        output_parquet_path=out_parquet,
        backend_name=backend,
        config=cfg,
    )
    print(f"[PITCHLAB] Calibration complete. Processed {len(results)} frames -> {out_parquet}")
    return 0


def cmd_footprint(args: argparse.Namespace, cfg: AppConfig) -> int:
    from pitchlab.core.io import load_parquet
    from pitchlab.core.types import FrameCalib
    from pitchlab.m2_calib.pipeline import run_calibration
    from pitchlab.m4_footprint.video_overlay import render_control_video

    video_path = Path(args.video)
    stem = video_path.stem
    out_dir = Path("data/out") / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    out_parquet = out_dir / "calib.parquet"
    out_video = out_dir / "control_preview.mp4"
    out_json = out_dir / "calib_summary.json"

    if out_video.exists() and not args.force:
        print(f"[PITCHLAB] Control video already exists at {out_video}. Use --force to recompute.")
        return 0

    # If calibration parquet not found, run calibration first
    if not out_parquet.exists() or args.force:
        print(f"[PITCHLAB] Running calibration before rendering footprint...")
        calib_results = run_calibration(
            video_path=args.video,
            output_parquet_path=out_parquet,
            backend_name=cfg.calibration.default_backend,
            config=cfg,
        )
    else:
        print(f"[PITCHLAB] Loading calibration from {out_parquet}...")
        import json
        import numpy as np
        import pandas as pd
        df = load_parquet(out_parquet)
        calib_results = []
        for _, row in df.iterrows():
            h_vals = [row.get(f"h_{i // 3}_{i % 3}") for i in range(9)]
            H = np.array(h_vals, dtype=np.float64).reshape((3, 3)) if None not in h_vals else None
            fp = np.array(json.loads(row["footprint_m"])) if pd.notna(row.get("footprint_m")) and row["footprint_m"] else None
            calib_results.append(
                FrameCalib(
                    frame_idx=int(row["frame_idx"]),
                    t=float(row["t"]),
                    status=row["status"],
                    H_img2pitch=H,
                    conf=float(row["conf"]),
                    footprint_m=fp,
                    source=row["source"],
                )
            )

    print(f"[PITCHLAB] Rendering control video with 2D minimap -> {out_video}...")
    render_control_video(
        video_path=args.video,
        calib_results=calib_results,
        output_video_path=out_video,
        output_json_path=out_json,
        config=cfg,
    )
    print(f"[PITCHLAB] Done! Control preview exported to {out_video} and {out_json}")
    return 0


def cmd_eval(args: argparse.Namespace, cfg: AppConfig) -> int:
    from pitchlab.m5_eval.reporter import run_benchmark_and_report

    backends = [b.strip() for b in args.backends.split(",")] if args.backends else ["roboflow"]
    print(f"[PITCHLAB] Running M5 benchmark for backends: {backends}...")

    metrics = run_benchmark_and_report(
        backend_names=backends,
        report_md_path="docs/EVAL_REPORT.md",
        results_csv_path="data/out/eval_results.csv",
    )

    print("\n" + "=" * 65)
    print("PITCHLAB M5 EVALUATION SUMMARY")
    print("=" * 65)
    for m in metrics:
        gate_str = "PASS (>= 0.90)" if m.gate_passed else "FAIL (< 0.90)"
        print(f"Backend: {m.backend_name:<12} | Median IoU (main): {m.median_iou_main:.3f} | Gate: {gate_str}")
    print("=" * 65)
    print(f"Report saved to: docs/EVAL_REPORT.md")
    print(f"CSV results:     data/out/eval_results.csv\n")
    return 0


def cmd_serve(args: argparse.Namespace, cfg: AppConfig) -> int:
    import uvicorn
    host = args.host or "127.0.0.1"
    port = args.port or 8000
    print(f"\n" + "=" * 65)
    print(f"🚀 PITCHLAB STUDIO WEB APP URUCHOMIONA!")
    print(f"👉 Otwórz w przeglądarce: http://{host}:{port}")
    print("=" * 65 + "\n")
    uvicorn.run("pitchlab.app.server:app", host=host, port=port, log_level="info")
    return 0


def cmd_track(args: argparse.Namespace, cfg: AppConfig) -> int:
    print(f"[PITCHLAB] Running M6+M7 (track) on: {args.video}")
    return 0


def cmd_world(args: argparse.Namespace, cfg: AppConfig) -> int:
    print(f"[PITCHLAB] Running M8 (world coordinates) on: {args.video}")
    return 0


def cmd_events(args: argparse.Namespace, cfg: AppConfig) -> int:
    print(f"[PITCHLAB] Running M9 (events) for match: {args.match}")
    return 0


def cmd_options(args: argparse.Namespace, cfg: AppConfig) -> int:
    print(f"[PITCHLAB] Running M10 (pass options) for match: {args.match}")
    return 0


def cmd_learn(args: argparse.Namespace, cfg: AppConfig) -> int:
    matches = args.matches.split(",") if args.matches else []
    print(f"[PITCHLAB] Running M11 (learn) on matches: {matches}")
    return 0


def cmd_tag(args: argparse.Namespace, cfg: AppConfig) -> int:
    print(f"[PITCHLAB] Running M12 (tagging) for match: {args.match}")
    return 0


def cmd_run_all(args: argparse.Namespace, cfg: AppConfig) -> int:
    print(f"[PITCHLAB] Running full pipeline for: {args.video} [mode={cfg.env_mode}]")
    return 0


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pitchlab",
        description="PITCHLAB: Tactical football analysis & pitch calibration from TV broadcast highlights"
    )
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file")

    subparsers = parser.add_subparsers(dest="command", required=True, help="Sub-commands")

    # M1: shots
    p_shots = subparsers.add_parser("shots", help="M1: Scene cut detection and view classification")
    p_shots.add_argument("video", type=str, help="Path to input video (local or gs://)")
    p_shots.add_argument("--force", action="store_true", help="Force recomputation")

    # M2+M3: calib
    p_calib = subparsers.add_parser("calib", help="M2+M3: Camera calibration & temporal smoothing")
    p_calib.add_argument("video", type=str, help="Path to input video")
    p_calib.add_argument("--backend", type=str, default="roboflow", help="Calibration backend (roboflow, pnlcalib, snbaseline)")
    p_calib.add_argument("--force", action="store_true", help="Force recomputation")

    # M4: footprint
    p_fp = subparsers.add_parser("footprint", help="M4: Camera footprint and minimapa overlay")
    p_fp.add_argument("video", type=str, help="Path to input video")
    p_fp.add_argument("--force", action="store_true", help="Force recomputation")

    # M5: eval
    p_eval = subparsers.add_parser("eval", help="M5: Evaluate and compare calibration backends")
    p_eval.add_argument("--backends", type=str, default="roboflow,pnlcalib,snbaseline", help="Comma-separated backend names")

    # M6+M7: track
    p_track = subparsers.add_parser("track", help="M6+M7: Player/ball detection, ByteTrack and team clustering")
    p_track.add_argument("video", type=str, help="Path to input video")
    p_track.add_argument("--force", action="store_true", help="Force recomputation")

    # M8: world
    p_world = subparsers.add_parser("world", help="M8: Projection to meters and physical kinematics")
    p_world.add_argument("video", type=str, help="Path to input video")
    p_world.add_argument("--force", action="store_true", help="Force recomputation")

    # M9: events
    p_events = subparsers.add_parser("events", help="M9: Detect ball possession, passes, and turnovers")
    p_events.add_argument("match", type=str, help="Match identifier or video path")
    p_events.add_argument("--force", action="store_true", help="Force recomputation")

    # M10: options
    p_options = subparsers.add_parser("options", help="M10: Calculate pass options and pitch control")
    p_options.add_argument("match", type=str, help="Match identifier")
    p_options.add_argument("--force", action="store_true", help="Force recomputation")

    # M11: learn
    p_learn = subparsers.add_parser("learn", help="M11: Train pass preference scoring weights")
    p_learn.add_argument("--matches", type=str, required=True, help="Comma-separated match IDs")

    # M12: tag
    p_tag = subparsers.add_parser("tag", help="M12: Manual tagging and tag dataset export")
    p_tag.add_argument("match", type=str, help="Match identifier")

    # run-all
    p_all = subparsers.add_parser("run-all", help="Execute complete pipeline for a highlight video")
    p_all.add_argument("video", type=str, help="Path to input video")
    p_all.add_argument("--backend", type=str, default=None, help="Calibration backend")
    p_all.add_argument("--force", action="store_true", help="Force recomputation")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)

    cfg = AppConfig.load(args.config)

    commands = {
        "shots": cmd_shots,
        "calib": cmd_calib,
        "footprint": cmd_footprint,
        "eval": cmd_eval,
        "track": cmd_track,
        "world": cmd_world,
        "events": cmd_events,
        "options": cmd_options,
        "learn": cmd_learn,
        "tag": cmd_tag,
        "run-all": cmd_run_all,
    }

    handler = commands.get(args.command)
    if handler:
        return handler(args, cfg)
    return 1


if __name__ == "__main__":
    sys.exit(main())
