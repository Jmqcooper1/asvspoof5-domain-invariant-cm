#!/usr/bin/env python3
"""Run layer-wise domain probes to analyze domain leakage.

This is a wrapper for probe_domain.py for backwards compatibility.
See probe_domain.py for full documentation.

Usage:
    python scripts/run_probes.py --checkpoint runs/my_run/checkpoints/best.pt --split dev

For ERM vs DANN comparison:
    python scripts/probe_domain.py --erm-checkpoint ... --dann-checkpoint ...
"""

import subprocess
import sys
from pathlib import Path


def main():
    # Get the directory containing this script
    script_dir = Path(__file__).parent

    # Build command to run probe_domain.py with same arguments
    cmd = [sys.executable, str(script_dir / "probe_domain.py")] + sys.argv[1:]

    # Execute
    result = subprocess.run(cmd)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
