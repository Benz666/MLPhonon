from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from .phono3py_backend import (
    _as_phonopy_atoms,
    _configure_nac,
    _load_compact_ifcs,
    _tensor_summary,
    _write_kappa,
    _write_mode_tables,
)


NATURAL_N_MASS_VARIANCE = 1.8376660528527028e-05
AL_MASS_AMU = 26.9815385
GA_MASS_AMU = 69.723


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _isotope_inclusive_cation_g2(x_al: float) -> tuple[float, dict[str, Any]]:
    isotope_sets = {
        "Al": {26.9815385: 1.0},
        "Ga": {69.0: 0.601, 71.0: 0.399},
    }
    fractions = {"Al": x_al, "Ga": 1.0 - x_al}
    isotope_averages = {
        element: sum(mass * abundance for mass, abundance in isotopes.items())
        for element, isotopes in isotope_sets.items()
    }
    average = sum(fractions[element] * isotope_averages[element] for element in fractions)
    g2 = sum(
        fractions[element] * abundance * (1.0 - mass / average) ** 2
        for element, isotopes in isotope_sets.items()
        for mass, abundance in isotopes.items()
    )
    return float(g2), {
        "formula": "sum_species,isotope c_species p_isotope (1-M_isotope/M_avg)^2",
        "fractions": fractions,
        "isotope_sets": isotope_sets,
        "isotope_average_masses_amu": isotope_averages,
        "average_isotope_mass_amu": average,
    }


def _read_borninfo(path: Path, n_atoms: int) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path)
    expected = (3 + 3 * n_atoms, 3)
    if data.shape != expected:
        raise ValueError(f"Unexpected BORNINFO shape {data.shape} in {path}; expected {expected}")
    return data[:3], data[3:].reshape(n_atoms, 3, 3)


def _write_interpolated_borninfo(
    output: Path, endpoint_aln: Path, endpoint_gan: Path, x_al: float, n_atoms: int
) -> Path:
    aln_eps, aln_born = _read_borninfo(endpoint_aln, n_atoms)
    gan_eps, gan_born = _read_borninfo(endpoint_gan, n_atoms)
    data = np.vstack(
        [
            x_al * aln_eps + (1.0 - x_al) * gan_eps,
            (x_al * aln_born + (1.0 - x_al) * gan_born).reshape(-1, 3),
        ]
    )
    path = output / "BORNINFO.vca_interpolated"
    np.savetxt(path, data, fmt="%.16e")
    return path


def _max_sparse_interpolation_error(
    vca_path: Path, aln_path: Path, gan_path: Path, x_al: float
) -> float:
    vca = sparse.load_npz(vca_path)
    expected = x_al * sparse.load_npz(aln_path) + (1.0 - x_al) * sparse.load_npz(gan_path)
    delta = vca - expected
    return 0.0 if delta.nnz == 0 else float(np.max(np.abs(delta.data)))


def _validate_interpolated_ifcs(
    vca_ifc: Path, aln_ifc: Path, gan_ifc: Path, x_al: float
) -> dict[str, Any]:
    aligned_report = vca_ifc / "alignment_validation.json"
    if aligned_report.exists():
        report = json.loads(aligned_report.read_text(encoding="utf-8"))
        checksums = report.get("checksums", {})
        for filename in ("second.npy", "third.npz", "POSCAR.vca_primitive"):
            expected = checksums.get(filename)
            current = _sha256(vca_ifc / filename)
            if expected != current:
                raise ValueError(
                    f"Aligned VCA checksum mismatch for {filename}: {current} != {expected}"
                )
        if not np.isclose(float(report.get("x_al", -1.0)), x_al, atol=1e-12):
            raise ValueError("Aligned VCA composition does not match requested x_Al")
        return {
            "mode": "orientation_aligned",
            "report": str(aligned_report.resolve()),
            "method": report.get("method"),
            "formula": report.get("formula"),
            "fc2_roundtrip_max_abs_error": report.get("fc2_roundtrip_max_abs_error"),
            "fc3_roundtrip_max_abs_error": report.get("fc3_roundtrip_max_abs_error"),
        }
    vca_fc2 = np.load(vca_ifc / "second.npy")
    expected_fc2 = x_al * np.load(aln_ifc / "second.npy") + (1.0 - x_al) * np.load(
        gan_ifc / "second.npy"
    )
    fc2_error = float(np.max(np.abs(vca_fc2 - expected_fc2)))
    fc3_error = _max_sparse_interpolation_error(
        vca_ifc / "third.npz",
        aln_ifc / "third.npz",
        gan_ifc / "third.npz",
        x_al,
    )
    tolerance = 1e-11
    if fc2_error > tolerance or fc3_error > tolerance:
        raise ValueError(
            f"VCA IFC interpolation validation failed: fc2={fc2_error:.3e}, "
            f"fc3={fc3_error:.3e}"
        )
    return {
        "formula": "FC_VCA = x_Al FC_AlN + (1-x_Al) FC_GaN",
        "maximum_absolute_fc2_error": fc2_error,
        "maximum_absolute_fc3_error": fc3_error,
        "tolerance": tolerance,
    }


def run_phono3py_vca(
    *,
    vca_workdir: Path,
    output_root: Path,
    endpoint_aln_source: Path,
    endpoint_gan_source: Path,
    endpoint_aln_phono3py: Path,
    endpoint_gan_phono3py: Path,
    x_al: float,
    fc2_supercell: list[int],
    fc3_supercell: list[int],
    mesh: list[int],
    temperatures: list[float],
    nac_method: str = "wang",
    use_symmetry: bool = True,
    r0_average: bool = True,
    log_level: int = 0,
    preflight_only: bool = False,
) -> Path:
    from ase.io import read as ase_read
    import phono3py
    import phonopy
    from phono3py import Phono3py

    if not 0.0 < x_al < 1.0:
        raise ValueError("VCA composition x_Al must be strictly between zero and one")
    vca_ifc = vca_workdir / "ifc_vca_numpy"
    structure = vca_ifc / "POSCAR.vca_primitive"
    required = [
        structure,
        vca_ifc / "second.npy",
        vca_ifc / "third.npz",
        vca_ifc / "replicated_atoms.xyz",
        vca_ifc / "replicated_atoms_third.xyz",
        endpoint_aln_source / "kaldo_ifc" / "second.npy",
        endpoint_aln_source / "kaldo_ifc" / "third.npz",
        endpoint_gan_source / "kaldo_ifc" / "second.npy",
        endpoint_gan_source / "kaldo_ifc" / "third.npz",
        endpoint_aln_source / "BORNINFO.mlip_relaxed",
        endpoint_gan_source / "BORNINFO.mlip_relaxed",
        endpoint_aln_phono3py / "transport_backend.json",
        endpoint_gan_phono3py / "transport_backend.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing phono3py VCA input(s): " + ", ".join(missing))

    output = output_root / vca_workdir.name
    output.mkdir(parents=True, exist_ok=True)
    atoms = ase_read(structure, format="vasp")
    vca_mass = x_al * AL_MASS_AMU + (1.0 - x_al) * GA_MASS_AMU
    masses = atoms.get_masses()
    for index, symbol in enumerate(atoms.get_chemical_symbols()):
        if symbol != "N":
            masses[index] = vca_mass
    atoms.set_masses(masses)

    ph3 = Phono3py(
        _as_phonopy_atoms(atoms),
        supercell_matrix=np.diag(fc3_supercell),
        phonon_supercell_matrix=np.diag(fc2_supercell),
        primitive_matrix=np.eye(3),
        is_symmetry=use_symmetry,
        is_mesh_symmetry=use_symmetry,
        make_r0_average=r0_average,
        log_level=log_level,
    )
    interpolation = _validate_interpolated_ifcs(
        vca_ifc,
        endpoint_aln_source / "kaldo_ifc",
        endpoint_gan_source / "kaldo_ifc",
        x_al,
    )
    fc2, fc3, mapping = _load_compact_ifcs(
        vca_workdir,
        ph3,
        fc2_supercell,
        fc3_supercell,
        ifc_dir_name="ifc_vca_numpy",
        structure_name="ifc_vca_numpy/POSCAR.vca_primitive",
    )
    ph3.fc2 = fc2
    ph3.fc3 = fc3
    ph3.mesh_numbers = mesh
    aligned_borninfo = vca_ifc / "BORNINFO.vca_aligned"
    if aligned_borninfo.exists():
        borninfo = output / "BORNINFO.vca_aligned"
        borninfo.write_bytes(aligned_borninfo.read_bytes())
    else:
        borninfo = _write_interpolated_borninfo(
            output,
            endpoint_aln_source / "BORNINFO.mlip_relaxed",
            endpoint_gan_source / "BORNINFO.mlip_relaxed",
            x_al,
            len(atoms),
        )
    nac = _configure_nac(ph3, borninfo, nac_method)
    cation_g2, cation_meta = _isotope_inclusive_cation_g2(x_al)
    mass_variances = np.asarray(
        [NATURAL_N_MASS_VARIANCE if symbol == "N" else cation_g2 for symbol in ph3.primitive.symbols],
        dtype=float,
    )

    ph3.init_phph_interaction()
    ph3.run_phonon_solver()
    frequencies_bz, _, _ = ph3.get_phonon_data()
    frequencies = frequencies_bz[ph3.grid.grg2bzg]
    gamma_frequencies = frequencies_bz[ph3.grid.gp_Gamma]
    harmonic = {
        "minimum_THz": float(np.min(frequencies)),
        "maximum_THz": float(np.max(frequencies)),
        "imaginary_below_minus_0p01_THz": int(np.count_nonzero(frequencies < -0.01)),
        "gamma_acoustic_THz": np.sort(gamma_frequencies)[:3].tolist(),
    }
    if harmonic["imaginary_below_minus_0p01_THz"]:
        raise RuntimeError(f"VCA harmonic preflight failed: {harmonic}")

    provenance: dict[str, Any] = {
        "workflow": "MLPhonon",
        "backend": "phono3py",
        "calculation": "AlGaN VCA RTA",
        "x_al": x_al,
        "vca_workdir": str(vca_workdir.resolve()),
        "output": str(output.resolve()),
        "endpoint_phono3py_results": {
            "AlN": str(endpoint_aln_phono3py.resolve()),
            "GaN": str(endpoint_gan_phono3py.resolve()),
        },
        "endpoint_ifc_sources": {
            "AlN": str(endpoint_aln_source.resolve()),
            "GaN": str(endpoint_gan_source.resolve()),
        },
        "endpoint_backend_sha256": {
            "AlN": _sha256(endpoint_aln_phono3py / "transport_backend.json"),
            "GaN": _sha256(endpoint_gan_phono3py / "transport_backend.json"),
        },
        "mesh": mesh,
        "temperatures_K": temperatures,
        "fc2_supercell": fc2_supercell,
        "fc3_supercell": fc3_supercell,
        "integration": "tetrahedron",
        "vca_cation_mass_amu": vca_mass,
        "mass_variances": mass_variances.tolist(),
        "mass_variance_metadata": {
            "cation": cation_meta,
            "natural_N": NATURAL_N_MASS_VARIANCE,
        },
        "nac": nac,
        "ifc_interpolation_validation": interpolation,
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
    if preflight_only:
        print(json.dumps(provenance["harmonic_preflight"], indent=2))
        print(f"[DONE] phono3py VCA preflight: {output}")
        return output

    ph3.sigmas = [None]
    ph3.run_thermal_conductivity(
        is_LBTE=False,
        temperatures=temperatures,
        is_isotope=True,
        mass_variances=mass_variances,
        write_gamma=False,
        log_level=log_level,
    )
    conductivity = ph3.thermal_conductivity
    if conductivity is None or conductivity.kappa is None:
        raise RuntimeError("phono3py did not return VCA RTA conductivity")
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
    (output / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["tensor_summaries"], indent=2))
    print(f"[DONE] phono3py VCA RTA result: {output}")
    return output
