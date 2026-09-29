from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


WORKFLOW_ROOT = Path(__file__).resolve().parents[1]


class ConfigError(RuntimeError):
    """Raised when a workflow config is incomplete or inconsistent."""


def _expand_environment(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    return value


def load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    cfg_path = Path(path)
    text = cfg_path.read_text()
    if cfg_path.suffix.lower() == ".json":
        return _expand_environment(json.loads(text))
    try:
        import yaml
    except ImportError as exc:
        raise ConfigError(
            "YAML config support requires PyYAML in this environment. "
            "Install PyYAML or use a .json config."
        ) from exc
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ConfigError(f"Config root must be a mapping: {cfg_path}")
    return _expand_environment(data)


def write_yaml(data: dict[str, Any], path: str | os.PathLike[str]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml
    except ImportError:
        out.write_text(json.dumps(data, indent=2) + "\n")
        return
    out.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=False))


def as_path(value: str | None) -> Path | None:
    if value in (None, "", "null"):
        return None
    return Path(str(value)).expanduser()


def require_mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"Missing or invalid mapping: {key}")
    return value


def require_list(data: dict[str, Any], key: str) -> list[Any]:
    value = data.get(key)
    if not isinstance(value, list):
        raise ConfigError(f"Missing or invalid list: {key}")
    return value


def slug(value: str) -> str:
    out = []
    for char in value.strip():
        if char.isalnum():
            out.append(char)
        elif char in ("-", "_", "."):
            out.append("_" if char == "." else char)
        else:
            out.append("_")
    return "".join(out).strip("_").lower()


def mesh_tag(mesh: list[int] | tuple[int, int, int]) -> str:
    return "x".join(str(int(x)) for x in mesh)


def model_label(model: dict[str, Any]) -> str:
    name = str(model["name"])
    head = model.get("head")
    if head in (None, "", "null"):
        return slug(name)
    if slug(str(head)) in slug(name):
        return slug(name)
    return slug(f"{name}_{head}")


def endpoint_materials(config: dict[str, Any]) -> list[dict[str, Any]]:
    system = require_mapping(config, "system")
    ctype = system.get("calculation_type")
    if ctype == "single":
        material = require_mapping(system, "material")
        return [material]
    if ctype == "binary_alloy":
        endpoints = require_mapping(system, "endpoints")
        mats = []
        for key in ("A", "B"):
            mat = endpoints.get(key)
            if not isinstance(mat, dict):
                raise ConfigError(f"Missing system.endpoints.{key}")
            mats.append(mat)
        return mats
    raise ConfigError("system.calculation_type must be single or binary_alloy")


def vca_fractions(config: dict[str, Any]) -> list[float]:
    system = require_mapping(config, "system")
    if system.get("calculation_type") != "binary_alloy":
        return []
    alloy = require_mapping(system, "alloy")
    values = alloy.get("fractions", [])
    if not isinstance(values, list):
        raise ConfigError("system.alloy.fractions must be a list")
    return [float(x) for x in values]


def workflow_steps(config: dict[str, Any]) -> list[str]:
    steps = ["relaxation", "nac", "fc2_fc3"]
    if config.get("banddos", {}).get("enabled", True):
        steps.append("banddos")
    if config.get("transport", {}).get("enabled", True):
        steps.append("endpoint_rta")
    if config.get("system", {}).get("calculation_type") == "binary_alloy":
        if config.get("vca", {}).get("enabled", True):
            steps.append("vca_rta")
        if config.get("qmesh_extrapolation", {}).get("enabled", False):
            steps.append("qmesh_extrapolation")
    steps.append("collect_plot")
    return steps


def find_auto_kappa_files(root: Path) -> dict[str, list[Path]]:
    found: dict[str, list[Path]] = {
        "bands": [],
        "dos": [],
        "borninfo": [],
        "tau_300K": [],
        "kappa": [],
    }
    if not root.exists():
        return found
    found["bands"] = sorted(root.glob("harm/bandos/*.bands"))
    found["dos"] = sorted(root.glob("harm/bandos/*.dos"))
    found["borninfo"] = sorted(root.glob("harm/bandos/BORNINFO")) + sorted(root.glob("**/BORNINFO"))
    found["tau_300K"] = sorted(root.glob("cube/**/tau_300K.csv"))
    found["kappa"] = sorted(root.glob("cube/**/kappa.dat"))
    return found


def validate_reference_csv(path: Path) -> tuple[int, list[str]]:
    if not path.exists():
        raise ConfigError(f"Reference CSV missing: {path}")
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        try:
            first = next(reader)
        except StopIteration as exc:
            raise ConfigError(f"Reference CSV is empty: {path}") from exc
        norm = [h.strip().lower() for h in first]
        has_named_columns = "x" in norm or "x_al" in norm
        has_kappa_column = any(k in norm for k in ("kappa", "kappa_w_mk", "k", "thermal_conductivity"))
        if not (has_named_columns and has_kappa_column):
            try:
                float(first[0])
                float(first[1])
            except (IndexError, TypeError, ValueError) as exc:
                raise ConfigError(
                    f"Reference CSV must contain x and kappa-like columns, or be two numeric columns: {path}"
                ) from exc
            rows = 1 + sum(1 for _ in reader)
            return rows, ["x", "kappa"]
        if "x" not in norm and "x_al" not in norm:
            raise ConfigError(
                f"Reference CSV must contain x and kappa-like columns: {path}"
            )
        rows = sum(1 for _ in reader)
    return rows, first


@dataclass
class ValidationReport:
    errors: list[str]
    warnings: list[str]
    info: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def add_info(self, msg: str) -> None:
        self.info.append(msg)


def validate_config(config: dict[str, Any], *, check_paths: bool = True) -> ValidationReport:
    report = ValidationReport(errors=[], warnings=[], info=[])

    try:
        project = require_mapping(config, "project")
        models = require_mapping(config, "models")
        system = require_mapping(config, "system")
    except ConfigError as exc:
        report.add_error(str(exc))
        return report

    root = as_path(project.get("root"))
    if root is None:
        report.add_error("project.root is required")
    elif check_paths and not root.exists():
        report.add_warning(f"project.root does not exist yet: {root}")

    mode = models.get("mode")
    if mode not in ("single", "multi"):
        report.add_error("models.mode must be single or multi")
    model_list = models.get("list")
    if not isinstance(model_list, list) or not model_list:
        report.add_error("models.list must contain at least one model")
    else:
        if mode == "single" and len(model_list) != 1:
            report.add_warning("models.mode is single but more than one model is listed")
        for idx, model in enumerate(model_list):
            if not isinstance(model, dict):
                report.add_error(f"models.list[{idx}] must be a mapping")
                continue
            for key in ("name", "type", "path"):
                if key not in model:
                    report.add_error(f"models.list[{idx}] missing {key}")
            path = as_path(model.get("path"))
            if path and check_paths and not path.exists():
                report.add_warning(f"model path does not exist: {path}")

    ctype = system.get("calculation_type")
    if ctype not in ("single", "binary_alloy"):
        report.add_error("system.calculation_type must be single or binary_alloy")

    try:
        mats = endpoint_materials(config)
        for mat in mats:
            formula = mat.get("formula", "<unknown>")
            prim = as_path(mat.get("primitive"))
            if prim and check_paths and not prim.exists():
                report.add_warning(f"{formula} primitive does not exist: {prim}")
            dft_root = as_path(mat.get("dft_auto_kappa_root"))
            if dft_root:
                if check_paths and not dft_root.exists():
                    report.add_warning(f"{formula} DFT auto-kappa root missing: {dft_root}")
                else:
                    found = find_auto_kappa_files(dft_root)
                    report.add_info(
                        f"{formula} DFT files: bands={len(found['bands'])}, dos={len(found['dos'])}, "
                        f"borninfo={len(found['borninfo'])}, tau_300K={len(found['tau_300K'])}, "
                        f"kappa={len(found['kappa'])}"
                    )
            else:
                report.add_warning(f"{formula} has no dft_auto_kappa_root")
    except ConfigError as exc:
        report.add_error(str(exc))

    relax_mode = config.get("relaxation", {}).get("relax_mode", "mlip")
    if relax_mode not in ("mlip", "dft"):
        report.add_error("relaxation.relax_mode must be mlip or dft")
    nac_mode = config.get("nac", {}).get("nac_mode", "vasp_relaxed")
    if nac_mode not in ("vasp_relaxed", "dft_reference", "none"):
        report.add_error("nac.nac_mode must be vasp_relaxed, dft_reference, or none")
    if relax_mode == "mlip" and nac_mode == "dft_reference":
        report.add_warning("MLIP-relaxed cells with dft_reference NAC require careful cell/basis matching")

    for key in ("fc2_supercell", "fc3_supercell"):
        value = config.get("force_constants", {}).get(key)
        if not (isinstance(value, list) and len(value) == 3 and all(int(x) > 0 for x in value)):
            report.add_error(f"force_constants.{key} must be three positive integers")

    qmesh = config.get("transport", {}).get("qmesh_default", [1, 1, 1])
    if not (isinstance(qmesh, list) and len(qmesh) == 3 and all(int(x) > 0 for x in qmesh)):
        report.add_error("transport.qmesh_default must be three positive integers")
    device = config.get("transport", {}).get("device", "cpu")
    if device != "cpu":
        report.add_error("phono3py transport requires transport.device=cpu")
    backend = config.get("transport", {}).get("backend", "phono3py")
    if backend != "phono3py":
        report.add_error("MLPhonon release transport backend must be phono3py")
    if backend == "phono3py":
        phono = config.get("transport", {}).get("phono3py", {})
        nac_setting = phono.get("nac", "auto")
        if nac_setting not in (True, False, "auto", "on", "off", "none"):
            report.add_error(
                "transport.phono3py.nac must be auto, on, off, none, true, or false"
            )
        if phono.get("nac_method", "wang") not in ("wang", "gonze"):
            report.add_error("transport.phono3py.nac_method must be wang or gonze")
        sigma = phono.get("sigma_thz")
        if sigma not in (None, "null", "tetrahedron"):
            if not isinstance(sigma, (int, float)) or float(sigma) <= 0:
                report.add_error(
                    "transport.phono3py.sigma_thz must be null/tetrahedron or positive"
                )
        mass_variances = phono.get("mass_variances")
        if mass_variances is not None:
            if not isinstance(mass_variances, (dict, list)):
                report.add_error(
                    "transport.phono3py.mass_variances must be a species mapping or list"
                )
            elif isinstance(mass_variances, dict) and any(
                not isinstance(value, (int, float)) or float(value) < 0
                for value in mass_variances.values()
            ):
                report.add_error(
                    "transport.phono3py.mass_variances values must be non-negative numbers"
                )
        source_result = phono.get("source_result")
        if check_paths and isinstance(source_result, str) and not Path(source_result).exists():
            report.add_warning(
                f"transport.phono3py.source_result does not exist: {source_result}"
            )
        dft_transport = phono.get("dft_transport")
        if (
            check_paths
            and isinstance(dft_transport, str)
            and not Path(dft_transport).exists()
        ):
            report.add_warning(
                f"transport.phono3py.dft_transport does not exist: {dft_transport}"
            )
        if not isinstance(phono.get("plot", True), bool):
            report.add_error("transport.phono3py.plot must be boolean")

    banddos = config.get("banddos", {})
    tail_mass_fraction = banddos.get("tail_mass_fraction", 1e-6)
    if not isinstance(tail_mass_fraction, (int, float)) or not 0 <= float(tail_mass_fraction) < 0.5:
        report.add_error("banddos.tail_mass_fraction must satisfy 0 <= value < 0.5")
    margin_fraction = banddos.get("frequency_margin_fraction", 0.02)
    if not isinstance(margin_fraction, (int, float)) or float(margin_fraction) < 0:
        report.add_error("banddos.frequency_margin_fraction must be non-negative")
    minimum_margin = banddos.get("minimum_margin_cm", 5.0)
    if not isinstance(minimum_margin, (int, float)) or float(minimum_margin) < 0:
        report.add_error("banddos.minimum_margin_cm must be non-negative")
    projected_dos_plot = banddos.get("projected_dos_plot", False)
    if not isinstance(projected_dos_plot, bool):
        report.add_error("banddos.projected_dos_plot must be boolean")

    if ctype == "binary_alloy":
        try:
            fractions = vca_fractions(config)
            if not fractions:
                report.add_error("binary_alloy requires system.alloy.fractions")
            for x in fractions:
                if x <= 0.0 or x >= 1.0:
                    report.add_warning(f"VCA fraction is endpoint-like; expected 0<x<1: {x}")
        except ConfigError as exc:
            report.add_error(str(exc))
        if config.get("qmesh_extrapolation", {}).get("enabled", False):
            meshes = config.get("qmesh_extrapolation", {}).get("meshes", [])
            if len(meshes) < int(config.get("qmesh_extrapolation", {}).get("min_mesh_count", 3)):
                report.add_error("qmesh_extrapolation requires at least min_mesh_count meshes")

    refs = config.get("references", {})
    if refs.get("enabled", False):
        for item in refs.get("curves", []):
            path = Path(item["path"] if isinstance(item, dict) else item)
            try:
                rows, header = validate_reference_csv(path)
                report.add_info(f"reference CSV OK: {path} rows={rows} header={header}")
            except ConfigError as exc:
                report.add_error(str(exc))
        for derived in refs.get("derived", []):
            if not isinstance(derived, dict):
                report.add_error("references.derived entries must be mappings")
                continue
            if derived.get("operation") != "average":
                report.add_error("Only references.derived operation=average is supported in v1")
            for file_name in derived.get("files", []):
                try:
                    rows, header = validate_reference_csv(Path(file_name))
                    report.add_info(f"derived reference CSV OK: {file_name} rows={rows} header={header}")
                except ConfigError as exc:
                    report.add_error(str(exc))

    cleanup = config.get("cleanup", {})
    if cleanup.get("mode", "standard") != "standard":
        report.add_warning("Only cleanup.mode=standard is implemented in v1")

    return report


def format_report(report: ValidationReport) -> str:
    lines: list[str] = []
    if report.info:
        lines.append("INFO")
        lines.extend(f"  - {msg}" for msg in report.info)
    if report.warnings:
        lines.append("WARNINGS")
        lines.extend(f"  - {msg}" for msg in report.warnings)
    if report.errors:
        lines.append("ERRORS")
        lines.extend(f"  - {msg}" for msg in report.errors)
    lines.append("STATUS: OK" if report.ok else "STATUS: FAILED")
    return "\n".join(lines)
