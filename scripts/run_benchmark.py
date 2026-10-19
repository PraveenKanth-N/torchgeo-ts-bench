"""CLI execution script to run TorchGeo-TS-Bench benchmarks."""

import os
import sys

# Ensure src is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_dir = os.path.join(project_root, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from torchgeo_ts_bench.experiment_runner import main

if __name__ == "__main__":
    main()
