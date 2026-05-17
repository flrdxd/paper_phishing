"""Command-line wrapper for the data quality test suite."""

from pathlib import Path
import sys


def main():
    try:
        import pytest
    except ImportError as exc:
        raise SystemExit("pytest is required. Install project requirements first.") from exc

    project_root = Path(__file__).resolve().parents[2]
    return pytest.main([str(project_root / "tests")])


if __name__ == "__main__":
    sys.exit(main())
