#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mlphonon.config import load_config, write_yaml
from mlphonon.phono3py_backend import run_phono3py_endpoint


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a cached-IFC phono3py endpoint at an explicit q mesh."
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--material", required=True)
    parser.add_argument("--mesh", nargs=3, type=int, required=True)
    parser.add_argument("--output-label", required=True)
    parser.add_argument("--runtime-config", type=Path, required=True)
    args = parser.parse_args()
    if any(value <= 0 for value in args.mesh):
        raise ValueError("qmesh values must be positive")

    config = copy.deepcopy(load_config(args.config))
    config["transport"]["qmesh_default"] = args.mesh
    settings = config["transport"].setdefault("phono3py", {})
    settings["output_label"] = args.output_label
    settings["plot"] = False
    write_yaml(config, args.runtime_config)
    run_phono3py_endpoint(
        args.runtime_config,
        model_label_arg=args.model,
        material=args.material,
    )


if __name__ == "__main__":
    main()
