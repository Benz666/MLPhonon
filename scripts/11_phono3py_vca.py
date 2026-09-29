#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mlphonon.phono3py_vca import run_phono3py_vca


def _triple(values: list[str]) -> list[int]:
    parsed = [int(value) for value in values]
    if len(parsed) != 3 or any(value <= 0 for value in parsed):
        raise argparse.ArgumentTypeError("three positive integers are required")
    return parsed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run phono3py RTA for one binary-alloy VCA composition.")
    parser.add_argument("--vca-workdir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--endpoint-aln-source", type=Path, required=True)
    parser.add_argument("--endpoint-gan-source", type=Path, required=True)
    parser.add_argument("--endpoint-aln-phono3py", type=Path, required=True)
    parser.add_argument("--endpoint-gan-phono3py", type=Path, required=True)
    parser.add_argument("--x-al", type=float, required=True)
    parser.add_argument("--fc2-supercell", nargs=3, default=[5, 5, 3])
    parser.add_argument("--fc3-supercell", nargs=3, default=[5, 5, 3])
    parser.add_argument("--mesh", nargs=3, default=[16, 16, 10])
    parser.add_argument("--temperature", type=float, action="append", default=None)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    run_phono3py_vca(
        vca_workdir=args.vca_workdir,
        output_root=args.output_root,
        endpoint_aln_source=args.endpoint_aln_source,
        endpoint_gan_source=args.endpoint_gan_source,
        endpoint_aln_phono3py=args.endpoint_aln_phono3py,
        endpoint_gan_phono3py=args.endpoint_gan_phono3py,
        x_al=args.x_al,
        fc2_supercell=_triple(args.fc2_supercell),
        fc3_supercell=_triple(args.fc3_supercell),
        mesh=_triple(args.mesh),
        temperatures=args.temperature or [300.0],
        preflight_only=args.preflight_only,
    )
