"""Command line entry point.

Kept intentionally thin: the real work lives in ``scripts/run_experiment.py`` and
``scripts/build_readme_table.py`` so that experiments are invoked the same way
whether or not the package is installed.
"""

import argparse
import sys
from collections.abc import Sequence

from tsfm_peft import __version__


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="tsfm-peft",
        description="Parameter-efficient fine-tuning for time series foundation models.",
    )
    parser.add_argument("--version", action="version", version=f"tsfm-peft {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI. Returns a process exit code."""
    parser = build_parser()
    parser.parse_args(argv)
    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
