#!/usr/bin/env python3
"""Render the README results table from the metrics artifacts on disk.

A shim over ``tsfm-peft table`` so the table is generated the same way whether or not the
package is installed as a console script:

    python scripts/build_readme_table.py --configs configs/experiments --write

The table is never hand-edited. Every number in it comes from an artifact written by a run,
so a number with no artifact behind it cannot appear.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tsfm_peft.cli import main

if __name__ == "__main__":
    sys.exit(main(["table", *sys.argv[1:]]))
