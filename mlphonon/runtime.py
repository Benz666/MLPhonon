from __future__ import annotations

import runpy
import sys
import types
from pathlib import Path
from typing import Any

from .config import WORKFLOW_ROOT, find_auto_kappa_files, load_config, model_label, slug


def _first_existing(paths: list[Path], fallback: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return fallback


def _material_map(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    system = config["system"]
    if system["calculation_type"] == "single":
        materials = [system["material"]]
    else:
        materials = [system["endpoints"]["A"], system["endpoints"]["B"]]
    out: dict[str, dict[str, Any]] = {}
    fc = config.get("force_constants", {})
    tr = config.get("transport", {})
    banddos = config.get("banddos", {})
    for mat in materials:
        formula = mat["formula"]
        dft_root = Path(mat.get("dft_auto_kappa_root", "."))
        found = find_auto_kappa_files(dft_root)
        bands = _first_existing(found["bands"], dft_root / "harm" / "bandos" / f"{formula}.bands")
        dos = _first_existing(found["dos"], dft_root / "harm" / "bandos" / f"{formula}.dos")
        born = _first_existing(found["borninfo"], dft_root / "harm" / "bandos" / "BORNINFO")
        dft_transport = dft_root
        if found["tau_300K"]:
            dft_transport = found["tau_300K"][-1].parent
        elif found["kappa"]:
            dft_transport = found["kappa"][-1].parent
        out[formula] = {
            "symbols": mat.get("symbols", []),
            "poscar": Path(mat["primitive"]),
            "nac_reference": dft_root / "nac",
            "borninfo": born,
            "dft_bands": bands,
            "dft_dos": dos,
            "dft_transport": dft_transport,
            "dos_mesh": mat.get("dos_mesh", tr.get("qmesh_default", [16, 16, 16])),
            "kaldo_kmesh": tr.get("qmesh_default", [16, 16, 16]),
            "fc2_supercell": fc.get("fc2_supercell", [5, 5, 3]),
            "fc3_supercell": fc.get("fc3_supercell", [5, 5, 3]),
            "dos_tail_mass_fraction": banddos.get("tail_mass_fraction", 1e-6),
            "frequency_margin_fraction": banddos.get("frequency_margin_fraction", 0.02),
            "frequency_minimum_margin_cm": banddos.get("minimum_margin_cm", 5.0),
            "projected_dos_plot": banddos.get("projected_dos_plot", False),
        }
    return out


def install_project_config(config: dict[str, Any], model: dict[str, Any] | None = None) -> None:
    module = types.ModuleType("project_config")
    project_root = Path(config["project"]["root"])
    module.PROJECT_ROOT = project_root
    module.MACE_BASE = Path(config.get("paths", {}).get("mace_base", project_root))
    module.ALGAN_DFT_BASE = Path(".")
    if model:
        model_path = Path(model["path"])
    else:
        model_path = Path(config["models"]["list"][0]["path"])
    module.MACE_MODEL = model_path
    module.DPA_MODEL = model_path
    module.MATERIALS = _material_map(config)
    module.BAND_PATH = config.get(
        "banddos",
        {},
    ).get(
        "band_path",
        [
            ([0, 0, 0], [0.5, 0, 0], 50),
            ([0.5, 0, 0], [1 / 3, 1 / 3, 0], 38),
            ([1 / 3, 1 / 3, 0], [0, 0, 0], 48),
            ([0, 0, 0], [0, 0, 0.5], 50),
            ([0, 0, 0.5], [0.5, 0, 0.5], 50),
            ([0.5, 0, 0.5], [1 / 3, 1 / 3, 0.5], 38),
            ([1 / 3, 1 / 3, 0.5], [0, 0, 0.5], 48),
            ([0.5, 0, 0.5], [0.5, 0, 0], 50),
            ([1 / 3, 1 / 3, 0.5], [1 / 3, 1 / 3, 0], 50),
        ],
    )
    module.DFT_TICKS = None
    module.DFT_TICK_LABELS = None
    module.SEGMENT_X_RANGES = None
    module.THZ_TO_CM = 33.3564095
    module.CM_TO_THZ = 1.0 / module.THZ_TO_CM
    module.KB = 1.38064852e-23
    module.HBAR = 1.05457182e-34
    module.ISOTOPE_G_FACTORS = config.get(
        "isotopes",
        {
            "Al": 0.0,
            "Ga": 0.00019712731536263687,
            "N": 3.6339105302149555e-05,
            "In": 0.0001242,
            "As": 0.0,
        },
    )
    sys.modules["project_config"] = module


def find_model(config: dict[str, Any], label: str) -> dict[str, Any]:
    for model in config["models"]["list"]:
        if model_label(model) == label or slug(model["name"]) == label:
            return model
    raise KeyError(f"Unknown model label: {label}")


def run_endpoint_stage(config_path: Path, *, model_label_arg: str, material: str, step: str) -> None:
    config = load_config(config_path)
    model = find_model(config, model_label_arg)
    model_type = model.get("type", "mace")
    model_kind = "dpa" if model_type in {"dpa", "dp", "deepmd"} else "mace"
    if step == "relaxation":
        device = config.get("relaxation", {}).get("device", config.get("transport", {}).get("device", "cpu"))
    elif step in {"fc2_fc3", "banddos"}:
        device = config.get("force_constants", {}).get("device", config.get("transport", {}).get("device", "cpu"))
    else:
        device = config.get("transport", {}).get("device", "cpu")
    if device == "gpu":
        device_arg = "cuda"
    elif device == "auto":
        device_arg = "auto"
    else:
        device_arg = "cpu"

    fc = config.get("force_constants", {})
    tr = config.get("transport", {})
    temps = [str(x) for x in tr.get("temperatures", {}).get("values", [300])]
    qmesh = [str(x) for x in tr.get("qmesh_default", [16, 16, 16])]
    fc2 = [str(x) for x in fc.get("fc2_supercell", [5, 5, 3])]
    fc3 = [str(x) for x in fc.get("fc3_supercell", [5, 5, 3])]
    run_label = slug(
        f"{model_label(model)}_fc2_{''.join(fc2)}_fc3_{''.join(fc3)}_k{''.join(qmesh)}"
    )
    install_project_config(config, model)
    nac_mode = config.get("nac", {}).get("nac_mode", "vasp_relaxed")
    if nac_mode == "vasp_relaxed":
        relaxed_born = (
            Path(config["project"]["root"])
            / "results"
            / material
            / model_kind
            / run_label
            / "BORNINFO.mlip_relaxed"
        )
        if relaxed_born.exists():
            sys.modules["project_config"].MATERIALS[material]["borninfo"] = relaxed_born
        elif step in {"banddos", "endpoint_rta", "plot"}:
            raise FileNotFoundError(
                f"nac_mode=vasp_relaxed requires self-consistent BORNINFO before {step}: "
                f"{relaxed_born}"
            )
    args = [
        "mlphonon.endpoint",
        material,
        "--model-kind",
        model_kind,
        "--model",
        str(model["path"]),
        "--device",
        device_arg,
        "--run-label",
        run_label,
        "--fc2-supercell",
        *fc2,
        "--fc3-supercell",
        *fc3,
        "--kpts",
        *qmesh,
        "--temps",
        *temps,
        "--method",
        tr.get("solver", "rta"),
        "--kaldo-unfolding",
        tr.get("unfolding", "auto"),
    ]
    if model.get("head") not in (None, "", "null"):
        args.extend(["--head", str(model["head"])])
    if config.get("relaxation", {}).get("relax_mode", "mlip") == "dft":
        args.append("--skip-relax")

    if nac_mode == "none":
        args.extend(["--borninfo", str(Path("__missing_BORNINFO__"))])

    if tr.get("max_frequency_from_banddos", False):
        args.append("--kaldo-max-frequency-from-banddos")
    if config.get("banddos", {}).get("projected_dos_plot", False):
        args.append("--projected-dos-plot")

    if step == "relaxation":
        args.extend(["--skip-banddos", "--skip-transport"])
    elif step == "fc2_fc3":
        args.extend(["--skip-banddos", "--fc-only"])
    elif step == "banddos":
        args.extend(["--skip-relax", "--skip-fd", "--skip-transport", "--banddos-from-kaldo"])
    elif step == "endpoint_rta":
        args.extend(["--skip-relax", "--skip-fd", "--skip-banddos"])
        finite = config.get("transport", {}).get("finite_size", {})
        if finite.get("enabled"):
            args.extend(["--finite-length-angstrom", *[str(x) for x in finite["length_angstrom"]]])
            args.extend(["--finite-length-method", finite.get("method", "ms")])
    elif step == "plot":
        args.append("--plot-only")
    else:
        raise ValueError(step)

    old_argv = sys.argv
    try:
        sys.argv = args
        runpy.run_path(str(WORKFLOW_ROOT / "mlphonon" / "endpoint.py"), run_name="__main__")
    finally:
        sys.argv = old_argv
