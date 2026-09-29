#!/usr/bin/env python
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from step_common import parse_step_args
from mlphonon.config import load_config
from mlphonon.runtime import find_model
from mlphonon.config import model_label, slug


def run_label(cfg, model):
    fc = cfg.get("force_constants", {})
    tr = cfg.get("transport", {})
    return slug(
        f"{model_label(model)}_fc2_{''.join(map(str, fc.get('fc2_supercell', [1,1,1])))}"
        f"_fc3_{''.join(map(str, fc.get('fc3_supercell', [1,1,1])))}"
        f"_k{''.join(map(str, tr.get('qmesh_default', [1,1,1])))}"
    )


def material_cfg(cfg, formula):
    system = cfg["system"]
    if system["calculation_type"] == "single":
        return system["material"]
    for item in system["endpoints"].values():
        if item["formula"] == formula:
            return item
    raise KeyError(formula)


def prepare_vasp_nac(cfg, model, formula):
    nac = cfg.get("nac", {})
    project = Path(cfg["project"]["root"])
    model_kind = "dpa" if model.get("type") in {"dpa", "dp", "deepmd"} else "mace"
    label = run_label(cfg, model)
    result_dir = project / "results" / formula / model_kind / label
    mat = material_cfg(cfg, formula)
    relaxed = result_dir / "POSCAR.mlip_relaxed"
    source_poscar = relaxed if relaxed.exists() else Path(mat["primitive"])
    nac_dir = result_dir / "nac_vasp"
    nac_dir.mkdir(parents=True, exist_ok=True)
    if not source_poscar.exists():
        raise FileNotFoundError(source_poscar)
    shutil.copy2(source_poscar, nac_dir / "POSCAR")

    ref_dir = nac.get("reference_input_dir") or mat.get("nac_reference")
    if ref_dir:
        ref = Path(ref_dir)
        for name in ("INCAR", "KPOINTS", "POTCAR"):
            src = ref / name
            if src.exists():
                shutil.copy2(src, nac_dir / name)
            else:
                print(f"[WARN] missing reference NAC input {src}")
    else:
        (nac_dir / "INCAR").write_text(
            "SYSTEM = NAC\nENCUT = 520\nEDIFF = 1E-8\nIBRION = 8\nLEPSILON = .TRUE.\nLREAL = .FALSE.\n"
        )
        (nac_dir / "KPOINTS").write_text("Gamma\n0\nGamma\n4 4 4\n0 0 0\n")
        print("[WARN] POTCAR was not generated; set nac.reference_input_dir for production VASP NAC.")

    (nac_dir / "metadata.txt").write_text(
        f"formula = {formula}\nmodel = {model_label(model)}\nrun_label = {label}\nsource_poscar = {source_poscar}\n"
    )
    return nac_dir, result_dir

if __name__ == "__main__":
    args = parse_step_args("Prepare or import NAC/BORN information.")
    cfg = load_config(args.config)
    mode = cfg.get("nac", {}).get("nac_mode", "vasp_relaxed")
    project = Path(cfg["project"]["root"])
    out = project / "workflow_manifests"
    out.mkdir(parents=True, exist_ok=True)
    model = find_model(cfg, args.model)
    if mode in {"dft_reference", "none"}:
        note = out / f"02_prepare_nac_{args.model}_{args.material}.txt"
        note.write_text(f"nac_mode={mode}\nNo VASP NAC input needed for this mode.\n")
        print(note)
    elif mode == "vasp_relaxed":
        nac_dir, result_dir = prepare_vasp_nac(cfg, model, args.material)
        print(f"[DONE] prepared {nac_dir}")
        if cfg.get("nac", {}).get("run_vasp", False):
            cmd = cfg.get("nac", {}).get("vasp_command", "mpirun -n ${NSLOTS:-32} vasp_std")
            subprocess.run(cmd, cwd=nac_dir, shell=True, check=True)
            converter = Path(__file__).resolve().parent / "vasprun_to_borninfo.py"
            subprocess.run(["python", str(converter), "vasprun.xml", "-o", "BORNINFO"], cwd=nac_dir, check=True)
            shutil.copy2(nac_dir / "BORNINFO", result_dir / "BORNINFO.mlip_relaxed")
            print(f"[DONE] wrote {result_dir / 'BORNINFO.mlip_relaxed'}")
    else:
        raise SystemExit(f"Unknown nac_mode: {mode}")
