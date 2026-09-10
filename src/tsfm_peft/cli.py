"""Command line entry point.

Subcommands are thin: they parse arguments and hand off to the library, so that running an
experiment from the CLI, from ``scripts/run_experiment.py`` and from a test all go through
exactly the same code path. Anything that can differ between those three is a way for the
published numbers to stop matching what a reader reproduces.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from tsfm_peft import __version__


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="tsfm-peft",
        description="Parameter-efficient fine-tuning for time series foundation models.",
    )
    parser.add_argument("--version", action="version", version=f"tsfm-peft {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    run = subparsers.add_parser("run", help="run one or more experiment configs")
    run.add_argument("configs", nargs="+", type=Path, help="experiment YAML files")
    run.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="where to write metrics artifacts (default: $TSFM_PEFT_RESULTS or ./results)",
    )
    run.add_argument(
        "--no-write", action="store_true", help="evaluate but do not write an artifact"
    )
    run.add_argument(
        "--force-download", action="store_true", help="re-download datasets even if cached"
    )
    run.add_argument(
        "--check",
        action="store_true",
        help="validate the configs and exit without loading data or weights",
    )

    subparsers.add_parser("list", help="list registered datasets and models")
    return parser


def _run(args: argparse.Namespace) -> int:
    """Run the ``run`` subcommand."""
    from tsfm_peft.config import load_experiment
    from tsfm_peft.experiment import run_experiment, summarise

    configs = [load_experiment(path) for path in args.configs]
    if args.check:
        for path, config in zip(args.configs, configs, strict=True):
            print(f"ok  {path}  ({config.name}: {config.data.dataset} / {config.model.name})")
        return 0

    for config in configs:
        outcome = run_experiment(
            config,
            results_dir=args.results_dir,
            write=not args.no_write,
            force_download=args.force_download,
        )
        print(summarise(outcome))
        print()
    return 0


def _list() -> int:
    """Run the ``list`` subcommand."""
    from tsfm_peft.data.registry import DATASETS
    from tsfm_peft.models.registry import MODELS

    print("datasets")
    for spec in DATASETS.values():
        marker = "  (generated fixture)" if spec.is_generated else ""
        print(f"  {spec.name:<12} {spec.freq:<3} m={spec.seasonality:<3} {spec.license}{marker}")
    print("\nmodels")
    for spec in MODELS.values():
        notes = [f"needs --extra {spec.extra}"] if spec.extra else []
        if spec.finetunable:
            notes.append("LoRA/DoRA")
        if spec.is_fixture:
            notes.append("smoke fixture, not a benchmark")
        suffix = f"  [{'; '.join(notes)}]" if notes else ""
        print(f"  {spec.name:<16} {spec.description}{suffix}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return _run(args)
    if args.command == "list":
        return _list()
    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
