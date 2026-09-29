from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np

from .config import endpoint_materials, find_auto_kappa_files, load_config, mesh_tag, model_label, slug
from .runtime import find_model


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _select_by_material(value: Any, material: str) -> Any:
    if isinstance(value, dict):
        return value.get(material)
    return value


def _as_phonopy_atoms(atoms: Any) -> Any:
    from phonopy.structure.atoms import PhonopyAtoms

    return PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        cell=atoms.cell.array,
        scaled_positions=atoms.get_scaled_positions(),
        masses=atoms.get_masses(),
    )


def _match_atoms(source: Any, target: Any, tolerance: float = 1e-5) -> np.ndarray:
    """Return a one-to-one source-index to target-index periodic mapping."""
    target_frac = np.asarray(target.scaled_positions, dtype=float) % 1.0
    source_frac = np.asarray(source.get_scaled_positions(wrap=True), dtype=float) % 1.0
    target_symbols = np.asarray(target.symbols)
    source_symbols = np.asarray(source.get_chemical_symbols())
    if len(source_frac) != len(target_frac):
        raise ValueError(
            f"Atom-map size mismatch: cached={len(source_frac)}, phono3py={len(target_frac)}"
        )
    mapping = np.full(len(source_frac), -1, dtype=int)
    used: set[int] = set()
    for index, (fractional, symbol) in enumerate(
        zip(source_frac, source_symbols, strict=True)
    ):
        candidates = np.where(target_symbols == symbol)[0]
        if not candidates.size:
            raise ValueError(f"No target atoms have species {symbol}")
        delta = target_frac[candidates] - fractional
        delta -= np.rint(delta)
        distances = np.linalg.norm(delta @ np.asarray(target.cell), axis=1)
        for local_index in np.argsort(distances):
            candidate = int(candidates[local_index])
            if candidate in used:
                continue
            if distances[local_index] > tolerance:
                raise ValueError(
                    f"Atom mapping failed at cached atom {index}: nearest unused "
                    f"distance={distances[local_index]:.6e} A"
                )
            mapping[index] = candidate
            used.add(candidate)
            break
    if np.any(mapping < 0) or len(used) != len(source_frac):
        raise ValueError("Cached-to-phono3py atom mapping is not one-to-one")
    return mapping


def _primitive_mapping(replicated: Any, primitive: Any, n_primitive: int) -> np.ndarray:
    origin = replicated[:n_primitive].copy()
    origin.set_cell(np.asarray(primitive.cell), scale_atoms=False)
    origin.set_pbc(True)
    return _match_atoms(origin, primitive)


def _load_compact_ifcs(
    source: Path,
    ph3: Any,
    fc2_supercell: list[int],
    fc3_supercell: list[int],
    *,
    ifc_dir_name: str = "kaldo_ifc",
    structure_name: str = "POSCAR.mlip_relaxed",
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    from ase.io import read as ase_read
    from scipy import sparse

    ifc_dir = source / ifc_dir_name
    second_path = ifc_dir / "second.npy"
    third_path = ifc_dir / "third.npz"
    fc2_atoms_path = ifc_dir / "replicated_atoms.xyz"
    fc3_atoms_path = ifc_dir / "replicated_atoms_third.xyz"
    if not fc3_atoms_path.exists():
        fc3_atoms_path = fc2_atoms_path
    structure_path = source / structure_name
    required = [second_path, third_path, fc2_atoms_path, structure_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing cached IFC inputs: " + ", ".join(missing))

    unitcell = ase_read(structure_path, format="vasp")
    n_primitive = len(unitcell)
    fc2_atoms = ase_read(fc2_atoms_path, format="extxyz")
    fc3_atoms = ase_read(fc3_atoms_path, format="extxyz")
    expected_fc2_atoms = n_primitive * int(np.prod(fc2_supercell))
    expected_fc3_atoms = n_primitive * int(np.prod(fc3_supercell))
    if len(fc2_atoms) != expected_fc2_atoms:
        raise ValueError(
            f"FC2 replicated atom count is {len(fc2_atoms)}, expected {expected_fc2_atoms}"
        )
    if len(fc3_atoms) != expected_fc3_atoms:
        raise ValueError(
            f"FC3 replicated atom count is {len(fc3_atoms)}, expected {expected_fc3_atoms}"
        )

    phonon_supercell = ph3.phonon_supercell
    fc2_super_map = _match_atoms(fc2_atoms, phonon_supercell)
    fc3_super_map = _match_atoms(fc3_atoms, ph3.supercell)
    fc2_primitive_map = _primitive_mapping(fc2_atoms, ph3.primitive, n_primitive)
    fc3_primitive_map = _primitive_mapping(fc3_atoms, ph3.primitive, n_primitive)
    phono_to_fc2_super = np.argsort(fc2_super_map)
    phono_to_fc3_super = np.argsort(fc3_super_map)
    phono_to_fc2_primitive = np.argsort(fc2_primitive_map)
    phono_to_fc3_primitive = np.argsort(fc3_primitive_map)

    second_raw = np.load(second_path)
    expected_second_size = n_primitive * 3 * len(fc2_atoms) * 3
    if second_raw.size != expected_second_size:
        raise ValueError(
            f"FC2 size is {second_raw.size}, expected {expected_second_size}"
        )
    second = second_raw.reshape(n_primitive, 3, len(fc2_atoms), 3)
    second = second.transpose(0, 2, 1, 3)
    second = second[phono_to_fc2_primitive][:, phono_to_fc2_super]

    third_sparse = sparse.load_npz(third_path)
    expected_third_size = n_primitive * 3 * len(fc3_atoms) * 3 * len(fc3_atoms) * 3
    if int(np.prod(third_sparse.shape)) != expected_third_size:
        raise ValueError(
            f"FC3 sparse shape {third_sparse.shape} has {int(np.prod(third_sparse.shape))} "
            f"elements, expected {expected_third_size}"
        )
    third = third_sparse.toarray().reshape(
        n_primitive, 3, len(fc3_atoms), 3, len(fc3_atoms), 3
    )
    third = third.transpose(0, 2, 4, 1, 3, 5)
    third = third[phono_to_fc3_primitive]
    third = third[:, phono_to_fc3_super]
    third = third[:, :, phono_to_fc3_super]

    report = {
        "fc2_supercell_matrix": fc2_supercell,
        "fc3_supercell_matrix": fc3_supercell,
        "fc2_replicated_atoms": str(fc2_atoms_path.resolve()),
        "fc3_replicated_atoms": str(fc3_atoms_path.resolve()),
        "structure": str(structure_path.resolve()),
        "cached_to_phono3py_fc2_supercell": fc2_super_map.tolist(),
        "cached_to_phono3py_fc3_supercell": fc3_super_map.tolist(),
        "cached_to_phono3py_fc2_primitive": fc2_primitive_map.tolist(),
        "cached_to_phono3py_fc3_primitive": fc3_primitive_map.tolist(),
        "fc2_shape": list(second.shape),
        "fc3_shape": list(third.shape),
        "checksums": {
            "second.npy": _sha256(second_path),
            "third.npz": _sha256(third_path),
            "replicated_atoms_fc2": _sha256(fc2_atoms_path),
            "replicated_atoms_fc3": _sha256(fc3_atoms_path),
        },
    }
    return np.ascontiguousarray(second), np.ascontiguousarray(third), report


def _configure_nac(ph3: Any, borninfo: Path | None, method: str) -> dict[str, Any] | None:
    if borninfo is None:
        return None
    from phonopy.physical_units import get_calculator_physical_units

    data = np.loadtxt(borninfo)
    n_atoms = len(ph3.primitive)
    expected_shape = (3 + 3 * n_atoms, 3)
    if data.shape != expected_shape:
        raise ValueError(
            f"Unexpected BORNINFO shape {data.shape}; expected {expected_shape}"
        )
    dielectric = data[:3]
    born = data[3:].reshape(n_atoms, 3, 3)
    factor = get_calculator_physical_units("vasp").nac_factor
    ph3.nac_params = {
        "born": born,
        "dielectric": dielectric,
        "factor": factor,
        "method": method,
    }
    return {
        "borninfo": str(borninfo.resolve()),
        "borninfo_sha256": _sha256(borninfo),
        "method": method,
        "factor": float(factor),
        "dielectric": dielectric.tolist(),
        "born_charge_sum_max_abs": float(np.max(np.abs(born.sum(axis=0)))),
    }


def _resolve_mass_variances(ph3: Any, settings: dict[str, Any]) -> np.ndarray | None:
    if not bool(settings.get("isotope", False)):
        return None
    configured = settings.get("mass_variances")
    symbols = list(ph3.primitive.symbols)
    if configured is None:
        from phono3py.other.isotope import get_mass_variances

        return np.asarray(get_mass_variances(primitive=ph3.primitive), dtype=float)
    if isinstance(configured, dict):
        missing = sorted(set(symbols) - set(configured))
        if missing:
            raise ValueError(
                "transport.phono3py.mass_variances is missing species: "
                + ", ".join(missing)
            )
        return np.asarray([configured[symbol] for symbol in symbols], dtype=float)
    values = np.asarray(configured, dtype=float)
    if values.shape != (len(symbols),):
        raise ValueError(
            "transport.phono3py.mass_variances must be a species mapping or "
            f"a list of length {len(symbols)}"
        )
    if np.any(values < 0):
        raise ValueError("transport.phono3py.mass_variances cannot be negative")
    return values


def _voigt_to_tensor(values: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [values[0], values[5], values[4]],
            [values[5], values[1], values[3]],
            [values[4], values[3], values[2]],
        ]
    )


def _tensor_summary(voigt: np.ndarray, cell: np.ndarray) -> dict[str, Any]:
    tensor = _voigt_to_tensor(np.asarray(voigt, dtype=float))
    axis3 = np.asarray(cell[2], dtype=float)
    axis3 /= np.linalg.norm(axis3)
    along_axis3 = float(axis3 @ tensor @ axis3)
    perpendicular = float((np.trace(tensor) - along_axis3) / 2.0)
    return {
        "voigt_order": ["xx", "yy", "zz", "yz", "xz", "xy"],
        "cartesian_tensor_W_mK": tensor.tolist(),
        "principal_values_W_mK": np.linalg.eigvalsh(tensor).tolist(),
        "trace_average_W_mK": float(np.trace(tensor) / 3.0),
        "cell_axis_3_cartesian": axis3.tolist(),
        "kappa_along_cell_axis_3_W_mK": along_axis3,
        "kappa_perpendicular_to_cell_axis_3_W_mK": perpendicular,
    }


def _read_reference_kappa(path: Path, temperature: float) -> np.ndarray:
    rows: list[list[float]] = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        values = [float(value) for value in stripped.replace(",", " ").split()]
        if len(values) >= 10:
            rows.append(values)
    if not rows:
        raise ValueError(f"No 3x3 conductivity row found in {path}")
    row = min(rows, key=lambda values: abs(values[0] - temperature))
    tensor = np.asarray(row[1:10], dtype=float).reshape(3, 3)
    return np.array(
        [tensor[0, 0], tensor[1, 1], tensor[2, 2], tensor[1, 2], tensor[0, 2], tensor[0, 1]]
    )


def _reference_comparison(
    current_voigt: np.ndarray,
    reference_path: Path,
    temperature: float,
    cell: np.ndarray,
) -> dict[str, Any]:
    reference_voigt = _read_reference_kappa(reference_path, temperature)
    current = _tensor_summary(current_voigt, cell)
    reference = _tensor_summary(reference_voigt, cell)
    keys = [
        "trace_average_W_mK",
        "kappa_along_cell_axis_3_W_mK",
        "kappa_perpendicular_to_cell_axis_3_W_mK",
    ]
    differences = {
        key: float((current[key] - reference[key]) / reference[key]) for key in keys
    }
    return {
        "reference_kappa": str(reference_path.resolve()),
        "reference_kappa_sha256": _sha256(reference_path),
        "current": current,
        "reference": reference,
        "relative_differences": differences,
        "maximum_absolute_relative_difference": float(
            max(abs(value) for value in differences.values())
        ),
    }


def _write_kappa(path: Path, temperatures: np.ndarray, kappa: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Temperature [K], kxx kxy kxz kyx kyy kyz kzx kzy kzz [W/mK]\n")
        for temperature, voigt in zip(temperatures, kappa[0], strict=True):
            tensor = _voigt_to_tensor(voigt)
            handle.write(
                f"{temperature:10.2f}  "
                + "  ".join(f"{value:14.6f}" for value in tensor.ravel())
                + "\n"
            )


def _write_mode_tables(output: Path, conductivity: Any) -> dict[str, Any]:
    gamma = np.asarray(conductivity.gamma)
    frequencies = np.asarray(conductivity.frequencies)
    velocities = np.asarray(conductivity.group_velocities)
    weights = np.asarray(conductivity.grid_weights, dtype=int)
    mode_kappa = np.asarray(conductivity.mode_kappa)
    sampling_points = int(conductivity.number_of_sampling_grid_points)
    reconstruction_errors: dict[str, float] = {}
    for temp_index, temperature in enumerate(np.asarray(conductivity.temperatures)):
        gamma_at_temp = gamma[0, temp_index]
        lifetime_ps = np.zeros_like(gamma_at_temp)
        valid = np.isfinite(gamma_at_temp) & (gamma_at_temp > 0) & (frequencies > 1e-4)
        lifetime_ps[valid] = 1.0 / (4.0 * np.pi * gamma_at_temp[valid])
        speed_m_s = np.linalg.norm(velocities * 100.0, axis=-1)
        mfp_nm = speed_m_s * lifetime_ps * 1e-3
        mode_star = mode_kappa[0, temp_index] / sampling_points
        reconstructed = mode_star.sum(axis=(0, 1))
        target = np.asarray(conductivity.kappa)[0, temp_index]
        error = float(np.max(np.abs(reconstructed - target)))
        reconstruction_errors[f"{float(temperature):g}K"] = error
        if error > 1e-5:
            raise RuntimeError(
                f"Mode-kappa reconstruction error at {temperature:g} K is {error:.3e} W/mK"
            )
        per_q = mode_star / weights[:, None, None]
        tag = f"{float(temperature):g}K"
        with (output / f"tau_{tag}.csv").open("w", encoding="utf-8") as handle:
            handle.write("# Created by the MLPhonon phono3py backend\n")
            handle.write("# Lifetime = 1 / (4 pi gamma); velocity magnitude is reported\n")
            handle.write(
                "# iq,band,frequency[cm^-1],lifetime[ps],|velocity|[m/s],MFP[nm],"
                "multiplicity,kxx,kyy,kzz,kyz,kxz,kxy\n"
            )
            for iq in range(frequencies.shape[0]):
                for band in range(frequencies.shape[1]):
                    values = [
                        iq + 1,
                        band + 1,
                        frequencies[iq, band] * 33.3564095198152,
                        lifetime_ps[iq, band],
                        speed_m_s[iq, band],
                        mfp_nm[iq, band],
                        int(weights[iq]),
                        *per_q[iq, band],
                    ]
                    handle.write(
                        f"{values[0]},{values[1]},"
                        + ",".join(f"{value:.8e}" for value in values[2:6])
                        + f",{values[6]},"
                        + ",".join(f"{value:.8e}" for value in values[7:])
                        + "\n"
                    )
    return {
        "mode_kappa_reconstruction_max_abs_error_W_mK": reconstruction_errors,
        "number_of_sampling_grid_points": sampling_points,
        "number_of_irreducible_grid_points": int(len(weights)),
    }


def _resolve_source(
    config: dict[str, Any], material: str, model: dict[str, Any]
) -> tuple[Path, str]:
    fc = config["force_constants"]
    mesh = config["transport"]["qmesh_default"]
    source_label = slug(
        f"{model_label(model)}_fc2_{''.join(str(x) for x in fc['fc2_supercell'])}_"
        f"fc3_{''.join(str(x) for x in fc['fc3_supercell'])}_k{''.join(str(x) for x in mesh)}"
    )
    phono = config.get("transport", {}).get("phono3py", {})
    override = _select_by_material(phono.get("source_result"), material)
    if override:
        return Path(str(override)).expanduser(), source_label
    model_kind = "dpa" if model.get("type") in {"dpa", "dp", "deepmd"} else "mace"
    output_root = config.get("project", {}).get("output_root", "results")
    source = Path(config["project"]["root"]) / output_root / material / model_kind / source_label
    return source, source_label


def _resolve_borninfo(
    config: dict[str, Any], material: str, source: Path
) -> Path | None:
    phono = config.get("transport", {}).get("phono3py", {})
    nac_setting = phono.get("nac", "auto")
    if nac_setting in (False, "off", "none"):
        return None
    explicit = _select_by_material(phono.get("borninfo"), material)
    if explicit:
        return Path(str(explicit)).expanduser()
    if config.get("nac", {}).get("nac_mode", "vasp_relaxed") == "none":
        if nac_setting in (True, "on"):
            raise ValueError("phono3py NAC was requested but nac.nac_mode is none")
        return None
    relaxed = source / "BORNINFO.mlip_relaxed"
    if relaxed.exists():
        return relaxed
    mat = next(
        item for item in endpoint_materials(config) if item["formula"] == material
    )
    dft_root = Path(mat.get("dft_auto_kappa_root", "."))
    found = find_auto_kappa_files(dft_root)["borninfo"]
    if found:
        return found[0]
    if nac_setting in (True, "on"):
        raise FileNotFoundError(f"No BORNINFO found for required phono3py NAC: {material}")
    return None


def _resolve_dft_transport(
    config: dict[str, Any], material: str, mesh: list[int]
) -> Path | None:
    settings = config.get("transport", {}).get("phono3py", {})
    explicit = _select_by_material(settings.get("dft_transport"), material)
    if explicit:
        path = Path(str(explicit)).expanduser()
        return path if path.exists() else None
    mat = next(item for item in endpoint_materials(config) if item["formula"] == material)
    dft_root = Path(mat.get("dft_auto_kappa_root", "."))
    candidates = sorted(
        {path.parent for path in find_auto_kappa_files(dft_root)["tau_300K"]}
    )
    mesh_text = "x".join(str(value) for value in mesh)
    matched = [path for path in candidates if mesh_text in path.name]
    if matched:
        return sorted(matched)[0]
    return candidates[0] if len(candidates) == 1 else None


def run_phono3py_endpoint(
    config_path: Path, *, model_label_arg: str, material: str
) -> Path:
    from ase.io import read as ase_read
    import phono3py
    import phonopy
    from phono3py import Phono3py

    config = load_config(config_path)
    model = find_model(config, model_label_arg)
    target = [item for item in endpoint_materials(config) if item["formula"] == material]
    if not target:
        raise KeyError(f"Unknown endpoint material: {material}")
    transport = config.get("transport", {})
    if transport.get("solver", "rta") != "rta":
        raise ValueError("The phono3py backend currently supports endpoint RTA only")
    settings = transport.get("phono3py", {})
    fc2_supercell = [int(x) for x in config["force_constants"]["fc2_supercell"]]
    fc3_supercell = [int(x) for x in config["force_constants"]["fc3_supercell"]]
    mesh = [int(x) for x in transport.get("qmesh_default", [16, 16, 16])]
    temperatures = np.asarray(
        transport.get("temperatures", {}).get("values", [300]), dtype=float
    )
    sigma_value = settings.get("sigma_thz")
    sigma = None if sigma_value in (None, "null", "tetrahedron") else float(sigma_value)
    source, source_label = _resolve_source(config, material, model)
    if not source.exists():
        raise FileNotFoundError(f"Cached endpoint result does not exist: {source}")
    borninfo = _resolve_borninfo(config, material, source)
    if borninfo is not None and not borninfo.exists():
        raise FileNotFoundError(f"BORNINFO does not exist: {borninfo}")

    integration_tag = "tetra" if sigma is None else f"sigma{sigma:g}thz"
    nac_tag = "nonac" if borninfo is None else f"nac_{settings.get('nac_method', 'wang')}"
    output_label = _select_by_material(settings.get("output_label"), material)
    if not output_label:
        output_label = slug(
            f"{model_label(model)}_fc2_{mesh_tag(fc2_supercell)}_fc3_{mesh_tag(fc3_supercell)}_"
            f"q{mesh_tag(mesh)}_{integration_tag}_{nac_tag}"
        )
    output_root = config.get("project", {}).get("output_root", "results")
    output = Path(config["project"]["root"]) / output_root / material / "phono3py" / output_label
    output.mkdir(parents=True, exist_ok=True)

    atoms = ase_read(source / "POSCAR.mlip_relaxed", format="vasp")
    use_symmetry = bool(settings.get("use_symmetry", True))
    ph3 = Phono3py(
        _as_phonopy_atoms(atoms),
        supercell_matrix=np.diag(fc3_supercell),
        phonon_supercell_matrix=np.diag(fc2_supercell),
        primitive_matrix=np.eye(3),
        is_symmetry=use_symmetry,
        is_mesh_symmetry=use_symmetry,
        make_r0_average=bool(settings.get("r0_average", True)),
        log_level=int(settings.get("log_level", 0)),
    )
    fc2, fc3, mapping = _load_compact_ifcs(
        source, ph3, fc2_supercell, fc3_supercell
    )
    from phono3py.phonon3.fc3 import (
        cutoff_fc3_by_zero,
        get_drift_fc3,
        set_permutation_symmetry_compact_fc3,
        set_translational_invariance_compact_fc3,
    )

    drift_before = get_drift_fc3(fc3, ph3.primitive)[:3]
    nonzero_blocks_before = int(
        np.count_nonzero(np.any(np.abs(fc3) > 0, axis=(3, 4, 5)))
    )
    cutoff_value = settings.get("cutoff_fc3_distance_A")
    cutoff_distance = None if cutoff_value in (None, "none", "null") else float(cutoff_value)
    if cutoff_distance is not None:
        if cutoff_distance <= 0:
            raise ValueError("cutoff_fc3_distance_A must be positive")
        cutoff_fc3_by_zero(
            fc3,
            ph3.supercell,
            cutoff_distance,
            p2s_map=ph3.primitive.p2s_map,
        )
    drift_after_cutoff = get_drift_fc3(fc3, ph3.primitive)[:3]
    nonzero_blocks_after_cutoff = int(
        np.count_nonzero(np.any(np.abs(fc3) > 0, axis=(3, 4, 5)))
    )
    symmetrize_iterations = int(settings.get("symmetrize_fc3_iterations", 0))
    if symmetrize_iterations < 0:
        raise ValueError("symmetrize_fc3_iterations cannot be negative")
    for _ in range(symmetrize_iterations):
        set_translational_invariance_compact_fc3(fc3, ph3.primitive)
        set_permutation_symmetry_compact_fc3(fc3, ph3.primitive)
    drift_after = get_drift_fc3(fc3, ph3.primitive)[:3]
    mapping["fc3_symmetrization"] = {
        "iterations": symmetrize_iterations,
        "maximum_drift_before": [float(value) for value in drift_before],
        "cutoff_distance_A": cutoff_distance,
        "nonzero_triplet_blocks_before_cutoff": nonzero_blocks_before,
        "nonzero_triplet_blocks_after_cutoff": nonzero_blocks_after_cutoff,
        "maximum_drift_after_cutoff": [float(value) for value in drift_after_cutoff],
        "maximum_drift_after": [float(value) for value in drift_after],
    }
    ph3.fc2 = fc2
    ph3.fc3 = fc3
    ph3.mesh_numbers = mesh
    nac = _configure_nac(ph3, borninfo, settings.get("nac_method", "wang"))
    mass_variances = _resolve_mass_variances(ph3, settings)
    ph3.init_phph_interaction()
    ph3.run_phonon_solver()
    frequencies_bz, _, _ = ph3.get_phonon_data()
    regular_frequencies = frequencies_bz[ph3.grid.grg2bzg]
    gamma_frequencies = frequencies_bz[ph3.grid.gp_Gamma]
    harmonic = {
        "n_frequencies": int(regular_frequencies.size),
        "minimum_THz": float(np.min(regular_frequencies)),
        "maximum_THz": float(np.max(regular_frequencies)),
        "imaginary_below_minus_0p01_THz": int(
            np.count_nonzero(regular_frequencies < -0.01)
        ),
        "gamma_acoustic_THz": np.sort(gamma_frequencies)[:3].tolist(),
    }
    allowed_imaginary = int(settings.get("maximum_imaginary_modes", 0))
    if harmonic["imaginary_below_minus_0p01_THz"] > allowed_imaginary:
        raise RuntimeError(
            "Harmonic preflight failed: "
            f"{harmonic['imaginary_below_minus_0p01_THz']} frequencies are below -0.01 THz"
        )

    provenance: dict[str, Any] = {
        "workflow": "MLPhonon",
        "backend": "phono3py",
        "config": str(config_path.resolve()),
        "config_sha256": _sha256(config_path),
        "project": config["project"].get("name"),
        "material": material,
        "model": model,
        "source_result": str(source.resolve()),
        "source_label": source_label,
        "output": str(output.resolve()),
        "mesh": mesh,
        "temperatures_K": temperatures.tolist(),
        "integration": "tetrahedron" if sigma is None else "gaussian",
        "sigma_THz": sigma,
        "isotope_scattering": bool(settings.get("isotope", False)),
        "mass_variances": (
            None if mass_variances is None else mass_variances.tolist()
        ),
        "symmetry_enabled": use_symmetry,
        "r0_average_enabled": bool(settings.get("r0_average", True)),
        "nac": nac,
        "mapping": mapping,
        "harmonic_preflight": harmonic,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "phonopy": phonopy.__version__,
            "phono3py": phono3py.__version__,
        },
    }
    (output / "transport_backend.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    config_copy = output / f"workflow_config_input{config_path.suffix.lower()}"
    config_copy.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")

    ph3.sigmas = [sigma]
    ph3.run_thermal_conductivity(
        is_LBTE=False,
        temperatures=temperatures.tolist(),
        is_isotope=bool(settings.get("isotope", False)),
        mass_variances=mass_variances,
        write_gamma=False,
        log_level=int(settings.get("log_level", 0)),
    )
    conductivity = ph3.thermal_conductivity
    if conductivity is None or conductivity.kappa is None:
        raise RuntimeError("phono3py did not return RTA conductivity")
    kappa = np.asarray(conductivity.kappa)
    mode_report = _write_mode_tables(output, conductivity)
    _write_kappa(output / "kappa.dat", np.asarray(conductivity.temperatures), kappa)
    np.savez_compressed(
        output / "phono3py_rta.npz",
        kappa=kappa,
        gamma=np.asarray(conductivity.gamma),
        temperatures=np.asarray(conductivity.temperatures),
        grid_points=np.asarray(conductivity.grid_points),
        grid_weights=np.asarray(conductivity.grid_weights),
        frequencies=np.asarray(conductivity.frequencies),
        group_velocities=np.asarray(conductivity.group_velocities),
        mode_kappa=np.asarray(conductivity.mode_kappa),
    )
    result = dict(provenance)
    result["kappa_W_mK"] = kappa.tolist()
    result["tensor_summaries"] = [
        _tensor_summary(kappa[0, index], np.asarray(atoms.cell))
        for index in range(len(temperatures))
    ]
    result["mode_data"] = mode_report

    reference_value = _select_by_material(settings.get("reference_kappa"), material)
    if reference_value:
        reference_path = Path(str(reference_value)).expanduser()
        comparison = _reference_comparison(
            kappa[0, 0], reference_path, float(temperatures[0]), np.asarray(atoms.cell)
        )
        tolerance = float(settings.get("reference_tolerance_fraction", 0.25))
        comparison["tolerance_fraction"] = tolerance
        comparison["passed"] = (
            comparison["maximum_absolute_relative_difference"] <= tolerance
        )
        result["reference_comparison"] = comparison
    (output / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    if settings.get("plot", True):
        try:
            from .phono3py_plot import postprocess_phono3py_result

            result["postprocessing"] = postprocess_phono3py_result(
                output,
                source=source,
                material=material,
                dft_transport=_resolve_dft_transport(config, material, mesh),
            )
        except Exception as error:
            result["postprocessing_error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
            print(
                f"[WARNING] phono3py plotting failed: "
                f"{type(error).__name__}: {error}"
            )
        (output / "result.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(result["tensor_summaries"], indent=2))
    if result.get("reference_comparison"):
        print(json.dumps(result["reference_comparison"], indent=2))
        if settings.get("require_reference_match", False) and not result["reference_comparison"]["passed"]:
            raise RuntimeError(
                "phono3py/reference conductivity comparison exceeded the configured tolerance"
            )
    print(f"[DONE] MLPhonon phono3py endpoint result: {output}")
    return output
