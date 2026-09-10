#!/usr/bin/env python3
"""Run one or more experiment configs and write their metrics artifacts.

A shim over ``tsfm-peft run`` so experiments are invoked the same way whether or not the
package is installed as a console script:

    python scripts/run_experiment.py configs/experiments/etth1-timesfm-zeroshot.yaml
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tsfm_peft.cli import main

if __name__ == "__main__":
    sys.exit(main(["run", *sys.argv[1:]]))
