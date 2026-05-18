#!/usr/bin/env python3
"""Project runner for reproducing the phishing detection paper."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def run_check() -> int:
    from phishing_detection.check_kaggle_setup import main

    return int(main() or 0)


def run_tests() -> int:
    try:
        import pytest
    except ImportError as exc:
        raise SystemExit(
            "pytest is required. Install dependencies with: "
            "uv pip install -r requirements.txt"
        ) from exc

    return pytest.main([str(PROJECT_ROOT / "tests")])


def run_train() -> int:
    from phishing_detection.main import main

    main()
    return 0


def run_experiment(args: argparse.Namespace) -> int:
    from phishing_detection.path_config import PATHS

    log_file = Path(PATHS["LOGS_DIR"]) / "research_experiments.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )

    if args.experiment_name == "h1-meajor":
        from phishing_detection.experiments.h1_meajor import run_h1_meajor

        run_h1_meajor(
            sample_size=args.sample_size,
            split=args.split,
            models=args.models,
            force_download=args.force_download,
            random_state=args.random_state,
        )
        return 0
    if args.experiment_name == "h2-url-meta":
        from phishing_detection.experiments.h2_url_meta import run_h2_url_meta

        run_h2_url_meta(
            sample_size=args.sample_size,
            split=args.split,
            force_download=args.force_download,
            random_state=args.random_state,
        )
        return 0
    if args.experiment_name == "h3-fusion":
        from phishing_detection.experiments.h3_fusion import run_h3_fusion

        run_h3_fusion(
            sample_size=args.sample_size,
            split=args.split,
            force_download=args.force_download,
            random_state=args.random_state,
        )
        return 0
    if args.experiment_name == "h4-compact-encoders":
        from phishing_detection.experiments.h4_encoders import run_h4_compact_encoders

        run_h4_compact_encoders(
            sample_size=args.sample_size,
            models=args.models,
            epochs=args.epochs,
            max_length=args.max_length,
            batch_size=args.batch_size,
            force_download=args.force_download,
            random_state=args.random_state,
        )
        return 0
    if args.experiment_name == "h5-robustness":
        from phishing_detection.experiments.h5_robustness import run_h5_robustness

        run_h5_robustness(
            sample_size=args.sample_size,
            force_download=args.force_download,
            random_state=args.random_state,
        )
        return 0
    if args.experiment_name == "h6-base-rates":
        from phishing_detection.experiments.h6_base_rates import run_h6_base_rates

        run_h6_base_rates(
            sample_size=args.sample_size,
            split=args.split,
            force_download=args.force_download,
            random_state=args.random_state,
        )
        return 0
    if args.experiment_name == "paper-v1":
        from phishing_detection.experiments.paper_v1 import run_paper_v1

        run_paper_v1(
            profile=args.profile,
            force_download=args.force_download,
            random_state=args.random_state,
        )
        return 0

    raise SystemExit(f"Unknown experiment: {args.experiment_name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproduce the phishing detection paper.")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("train", help="Train all paper models.")
    subparsers.add_parser("check", help="Check Kaggle credentials and project setup.")
    subparsers.add_parser("test", help="Run the project test suite.")

    experiment_parser = subparsers.add_parser(
        "experiment",
        help="Run research experiments without changing the paper reproduction pipeline.",
    )
    experiment_subparsers = experiment_parser.add_subparsers(
        dest="experiment_name",
        required=True,
    )
    h1_parser = experiment_subparsers.add_parser(
        "h1-meajor",
        help="Run H1 on the MeAJOR multi-source phishing email corpus.",
    )
    h1_parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional stratified sample size for a fast smoke run.",
    )
    h1_parser.add_argument(
        "--split",
        choices=["random", "source", "leave-one-source-out"],
        default="random",
        help="Train/test split policy.",
    )
    h1_parser.add_argument(
        "--models",
        default="nb,logreg,svm",
        help="Comma-separated models: nb,logreg,svm.",
    )
    h1_parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download the MeAJOR CSV even when cached.",
    )
    h1_parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used by sampling, splitting, and models.",
    )

    for name, help_text in [
        ("h2-url-meta", "Run H2 URL/metadata feature ablation."),
        ("h3-fusion", "Run H3 fusion and threshold calibration."),
        ("h6-base-rates", "Run H6 realistic base-rate simulation."),
    ]:
        parser_i = experiment_subparsers.add_parser(name, help=help_text)
        add_common_experiment_args(parser_i, include_split=True)

    h5_parser = experiment_subparsers.add_parser(
        "h5-robustness",
        help="Run H5 controlled rewrite robustness stress test.",
    )
    add_common_experiment_args(h5_parser, include_split=False)

    h4_parser = experiment_subparsers.add_parser(
        "h4-compact-encoders",
        help="Run H4 compact transformer encoders.",
    )
    add_common_experiment_args(h4_parser, include_split=False)
    h4_parser.add_argument(
        "--models",
        default="distilbert,minilm",
        help="Comma-separated encoders: distilbert,minilm,bert.",
    )
    h4_parser.add_argument("--epochs", type=int, default=1)
    h4_parser.add_argument("--max-length", type=int, default=256)
    h4_parser.add_argument("--batch-size", type=int, default=8)

    paper_parser = experiment_subparsers.add_parser(
        "paper-v1",
        help="Run the implemented paper-v1 experiment suite.",
    )
    paper_parser.add_argument(
        "--profile",
        choices=["smoke", "full", "confirmatory"],
        default="smoke",
        help="Execution profile.",
    )
    paper_parser.add_argument("--force-download", action="store_true")
    paper_parser.add_argument("--random-state", type=int, default=42)

    return parser


def add_common_experiment_args(parser: argparse.ArgumentParser, include_split: bool) -> None:
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional stratified sample size for a fast smoke run.",
    )
    if include_split:
        parser.add_argument(
            "--split",
            choices=["random", "source", "leave-one-source-out"],
            default="random",
            help="Train/test split policy.",
        )
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    command = args.command or "train"
    if command == "check":
        return run_check()
    if command == "test":
        return run_tests()
    if command == "train":
        return run_train()
    if command == "experiment":
        return run_experiment(args)

    parser.error(f"Unknown command: {command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
