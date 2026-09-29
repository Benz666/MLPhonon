#!/usr/bin/env python
from pathlib import Path
from step_common import parse_step_args
from mlphonon.runtime import run_endpoint_stage

if __name__ == "__main__":
    args = parse_step_args("Compute FC2/FC3 IFCs.")
    run_endpoint_stage(Path(args.config), model_label_arg=args.model, material=args.material, step="fc2_fc3")
