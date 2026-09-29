#!/usr/bin/env python
from pathlib import Path
from step_common import parse_step_args
from mlphonon.config import load_config
from mlphonon.phono3py_backend import run_phono3py_endpoint

if __name__ == "__main__":
    args = parse_step_args("Run endpoint RTA with the configured transport backend.")
    config_path = Path(args.config)
    backend = load_config(config_path).get("transport", {}).get("backend", "phono3py")
    if backend != "phono3py":
        raise SystemExit("MLPhonon endpoint transport requires backend=phono3py")
    run_phono3py_endpoint(
        config_path, model_label_arg=args.model, material=args.material
    )
