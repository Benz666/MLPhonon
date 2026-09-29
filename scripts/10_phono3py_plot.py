#!/usr/bin/env python3
"""Generate standard plots from an existing MLPhonon phono3py result."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mlphonon.phono3py_plot import postprocess_phono3py_result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--material", required=True)
    parser.add_argument("--dft-transport", type=Path, default=None)
    args = parser.parse_args()
    postprocess_phono3py_result(
        args.result,
        source=args.source,
        material=args.material,
        dft_transport=args.dft_transport,
    )
    print(f"[DONE] phono3py plots written to {args.result}")


if __name__ == "__main__":
    main()
