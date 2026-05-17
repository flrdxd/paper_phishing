#!/usr/bin/env python3
"""Project runner for reproducing the phishing detection paper."""

from __future__ import annotations

import argparse
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproduce the phishing detection paper.")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("train", help="Train all paper models.")
    subparsers.add_parser("check", help="Check Kaggle credentials and project setup.")
    subparsers.add_parser("test", help="Run the project test suite.")

    return parser


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

    parser.error(f"Unknown command: {command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
