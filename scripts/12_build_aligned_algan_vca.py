#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlphonon.algan_vca_alignment import build_aligned_algan_vca_scan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build orientation-aligned AlGaN VCA IFCs.")
    parser.add_argument("--endpoint-aln", type=Path, required=True)
    parser.add_argument("--endpoint-gan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fraction", type=float, action="append", required=True)
    parser.add_argument("--fc2-supercell", nargs=3, type=int, default=[5, 5, 3])
    parser.add_argument("--fc3-supercell", nargs=3, type=int, default=[5, 5, 3])
    args = parser.parse_args()
    build_aligned_algan_vca_scan(
        endpoint_aln=args.endpoint_aln,
        endpoint_gan=args.endpoint_gan,
        output_root=args.output_root,
        fractions=args.fraction,
        fc2_supercell=args.fc2_supercell,
        fc3_supercell=args.fc3_supercell,
    )
