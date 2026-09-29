from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from .phono3py_backend import _as_phonopy_atoms, _load_compact_ifcs, _match_atoms


AL_MASS_AMU = 26.9815385
GA_MASS_AMU = 69.723
WURTZITE_GAN_TO_ALN_FRACTIONAL = np.array(
    [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]], dtype=float
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ph3(atoms: Any, fc2_supercell: list[int], fc3_supercell: list[int]) -> Any:
    from phono3py import Phono3py

    return Phono3py(
        _as_phonopy_atoms(atoms),
        supercell_matrix=np.diag(fc3_supercell),
        phonon_supercell_matrix=np.diag(fc2_supercell),
        primitive_matrix=np.eye(3),
        is_symmetry=True,
        is_mesh_symmetry=True,
        make_r0_average=True,
        log_level=0,
    )


def _phonopy_to_ase(atoms: Any, *, positions: np.ndarray | None = None) -> Any:
    from ase import Atoms

    return Atoms(
        symbols=list(atoms.symbols),
        positions=np.asarray(atoms.positions) if positions is None else positions,
        cell=np.asarray(atoms.cell),
        pbc=True,
    )


def _rotation_from_fractional(cell: np.ndarray, fractional: np.ndarray) -> np.ndarray:
    # Row-vector convention: r_new = r_old @ Q and f_new = f_old @ fractional.
    q = np.linalg.inv(cell) @ fractional @ cell
    if not np.allclose(q.T @ q, np.eye(3), atol=1e-10) or not np.isclose(
        np.linalg.det(q), 1.0, atol=1e-10
    ):
        raise ValueError(f"The endpoint alignment is not a proper Cartesian rotation: {q}")
    return q


def _transformed_index_map(
    source: Any,
    target: Any,
    q_row: np.ndarray,
    translation_cart: np.ndarray,
) -> np.ndarray:
    rotated = _phonopy_to_ase(
        source,
        positions=np.asarray(source.positions, dtype=float) @ q_row + translation_cart,
    )
    rotated.set_cell(np.asarray(target.cell), scale_atoms=False)
    return _match_atoms(rotated, target, tolerance=1e-5)


def _rotate_fc2(fc2: np.ndarray, q_row: np.ndarray) -> np.ndarray:
    r = q_row.T
    return np.einsum("ai,psij,bj->psab", r, fc2, r, optimize=True)


def _rotate_fc3(fc3: np.ndarray, q_row: np.ndarray) -> np.ndarray:
    r = q_row.T
    return np.einsum("ai,pstijk,bj,ck->pstabc", r, fc3, r, r, optimize=True)


def _align_gan_compact_ifcs(
    fc2: np.ndarray,
    fc3: np.ndarray,
    source_ph3: Any,
    target_ph3: Any,
    q_row: np.ndarray,
    translation_cart: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    primitive_map = _transformed_index_map(
        source_ph3.primitive, target_ph3.primitive, q_row, translation_cart
    )
    fc2_super_map = _transformed_index_map(
        source_ph3.phonon_supercell,
        target_ph3.phonon_supercell,
        q_row,
        translation_cart,
    )
    fc3_super_map = _transformed_index_map(
        source_ph3.supercell, target_ph3.supercell, q_row, translation_cart
    )
    target_to_source_primitive = np.argsort(primitive_map)
    target_to_source_fc2 = np.argsort(fc2_super_map)
    target_to_source_fc3 = np.argsort(fc3_super_map)
    reordered_fc2 = fc2[target_to_source_primitive][:, target_to_source_fc2]
    reordered_fc3 = fc3[target_to_source_primitive]
    reordered_fc3 = reordered_fc3[:, target_to_source_fc3]
    reordered_fc3 = reordered_fc3[:, :, target_to_source_fc3]
    return (
        np.ascontiguousarray(_rotate_fc2(reordered_fc2, q_row)),
        np.ascontiguousarray(_rotate_fc3(reordered_fc3, q_row)),
        {
            "primitive_source_to_target": primitive_map.tolist(),
            "fc2_supercell_source_to_target": fc2_super_map.tolist(),
            "fc3_supercell_source_to_target": fc3_super_map.tolist(),
            "translation_cart_A": translation_cart.tolist(),
        },
    )


def _interpolate_fractional(
    aln_fractional: np.ndarray, gan_aligned_fractional: np.ndarray, x_al: float
) -> np.ndarray:
    delta = gan_aligned_fractional - aln_fractional
    delta -= np.rint(delta)
    return (aln_fractional + (1.0 - x_al) * delta) % 1.0


def _rotate_second_rank(tensors: np.ndarray, q_row: np.ndarray) -> np.ndarray:
    r = q_row.T
    return np.einsum("ai,...ij,bj->...ab", r, tensors, r, optimize=True)


def _read_borninfo(path: Path, n_atoms: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.loadtxt(path)
    expected = (3 + 3 * n_atoms, 3)
    if values.shape != expected:
        raise ValueError(f"Unexpected BORNINFO shape {values.shape}; expected {expected}")
    return values[:3], values[3:].reshape(n_atoms, 3, 3)


def _write_numpy_ifcs(
    output: Path,
    atoms: Any,
    ph3: Any,
    fc2: np.ndarray,
    fc3: np.ndarray,
    fc2_supercell: list[int],
    fc3_supercell: list[int],
) -> dict[str, Any]:
    from ase.io import write as ase_write

    output.mkdir(parents=True, exist_ok=True)
    fc2_atoms = atoms.repeat(fc2_supercell)
    fc3_atoms = atoms.repeat(fc3_supercell)
    ase_write(output / "vca_primitive.xyz", atoms, format="extxyz")
    ase_write(output / "POSCAR.vca_primitive", atoms, format="vasp")
    ase_write(output / "replicated_atoms.xyz", fc2_atoms, format="extxyz")
    ase_write(output / "replicated_atoms_third.xyz", fc3_atoms, format="extxyz")

    fc2_primitive_map = _match_atoms(atoms, ph3.primitive)
    fc3_primitive_map = _match_atoms(atoms, ph3.primitive)
    fc2_super_map = _match_atoms(fc2_atoms, ph3.phonon_supercell)
    fc3_super_map = _match_atoms(fc3_atoms, ph3.supercell)
    fc2_ase = fc2[fc2_primitive_map][:, fc2_super_map]
    fc3_ase = fc3[fc3_primitive_map]
    fc3_ase = fc3_ase[:, fc3_super_map]
    fc3_ase = fc3_ase[:, :, fc3_super_map]

    n_primitive = len(atoms)
    n_fc2_translations = int(np.prod(fc2_supercell))
    n_fc3_atoms = len(fc3_atoms)
    second_raw = fc2_ase.transpose(0, 2, 1, 3).reshape(
        1, n_primitive, 3, n_fc2_translations, n_primitive, 3
    )
    np.save(output / "second.npy", second_raw)
    third_raw = fc3_ase.transpose(0, 3, 1, 4, 2, 5).reshape(
        n_primitive * 3 * n_fc3_atoms * 3, n_fc3_atoms * 3
    )
    sparse.save_npz(output / "third.npz", sparse.csr_matrix(third_raw))
    return {
        "fc2_primitive_ase_to_phono3py": fc2_primitive_map.tolist(),
        "fc3_primitive_ase_to_phono3py": fc3_primitive_map.tolist(),
        "fc2_supercell_ase_to_phono3py": fc2_super_map.tolist(),
        "fc3_supercell_ase_to_phono3py": fc3_super_map.tolist(),
        "second_shape": list(second_raw.shape),
        "third_shape": list(third_raw.shape),
    }


def build_aligned_algan_vca_scan(
    *,
    endpoint_aln: Path,
    endpoint_gan: Path,
    output_root: Path,
    fractions: list[float],
    fc2_supercell: list[int],
    fc3_supercell: list[int],
) -> list[Path]:
    from ase.io import read as ase_read
    import spglib

    aln_atoms = ase_read(endpoint_aln / "POSCAR.mlip_relaxed", format="vasp")
    gan_atoms = ase_read(endpoint_gan / "POSCAR.mlip_relaxed", format="vasp")
    if aln_atoms.get_chemical_symbols()[:2] != ["N", "N"] or gan_atoms.get_chemical_symbols()[:2] != ["N", "N"]:
        raise ValueError("Expected N,N,cation,cation endpoint ordering")

    aln_ph3 = _ph3(aln_atoms, fc2_supercell, fc3_supercell)
    gan_ph3 = _ph3(gan_atoms, fc2_supercell, fc3_supercell)
    aln_fc2, aln_fc3, aln_mapping = _load_compact_ifcs(
        endpoint_aln, aln_ph3, fc2_supercell, fc3_supercell
    )
    gan_fc2, gan_fc3, gan_mapping = _load_compact_ifcs(
        endpoint_gan, gan_ph3, fc2_supercell, fc3_supercell
    )

    fractional_transform = WURTZITE_GAN_TO_ALN_FRACTIONAL
    q_row = _rotation_from_fractional(np.asarray(gan_atoms.cell), fractional_transform)
    gan_aligned_atoms = gan_atoms.copy()
    gan_fractional_unwrapped = (
        gan_atoms.get_scaled_positions(wrap=True) @ fractional_transform
    )
    gan_fractional_aligned = gan_fractional_unwrapped % 1.0
    gan_aligned_atoms.set_scaled_positions(gan_fractional_aligned)
    translation_fractional = gan_fractional_aligned - gan_fractional_unwrapped
    rounded_translation = np.rint(translation_fractional).astype(int)
    if not np.allclose(translation_fractional, rounded_translation, atol=1e-10):
        raise ValueError("GaN alignment translation is not a lattice vector")
    if not np.all(rounded_translation == rounded_translation[0]):
        raise ValueError(
            "GaN endpoint atoms require different lattice translations; "
            "compact IFC alignment needs per-anchor remapping"
        )
    translation_cart = rounded_translation[0] @ np.asarray(gan_atoms.cell)
    gan_aligned_ph3 = _ph3(gan_aligned_atoms, fc2_supercell, fc3_supercell)
    gan_fc2_aligned, gan_fc3_aligned, alignment_maps = _align_gan_compact_ifcs(
        gan_fc2,
        gan_fc3,
        gan_ph3,
        gan_aligned_ph3,
        q_row,
        translation_cart,
    )

    if not np.array_equal(aln_ph3.primitive.p2s_map, gan_aligned_ph3.primitive.p2s_map):
        raise ValueError("Aligned endpoint primitive-to-supercell maps differ")
    if not np.array_equal(aln_ph3.supercell.s2u_map, gan_aligned_ph3.supercell.s2u_map):
        raise ValueError("Aligned endpoint supercell topologies differ")

    aln_eps, aln_born = _read_borninfo(
        endpoint_aln / "BORNINFO.mlip_relaxed", len(aln_atoms)
    )
    gan_eps, gan_born = _read_borninfo(
        endpoint_gan / "BORNINFO.mlip_relaxed", len(gan_atoms)
    )
    gan_eps_aligned = _rotate_second_rank(gan_eps, q_row)
    gan_born_aligned = _rotate_second_rank(gan_born, q_row)

    generated: list[Path] = []
    for x_al in fractions:
        if not 0.0 < x_al < 1.0:
            raise ValueError(f"VCA fraction must be between zero and one: {x_al}")
        composition = (
            f"Al{x_al:.3f}".rstrip("0").rstrip(".")
            + f"Ga{1.0 - x_al:.3f}".rstrip("0").rstrip(".")
            + "N"
        )
        workdir = output_root / composition
        ifc_dir = workdir / "ifc_vca_numpy"
        cell = x_al * np.asarray(aln_atoms.cell) + (1.0 - x_al) * np.asarray(
            gan_aligned_atoms.cell
        )
        fractional = _interpolate_fractional(
            aln_atoms.get_scaled_positions(wrap=True),
            gan_aligned_atoms.get_scaled_positions(wrap=True),
            x_al,
        )
        symbols = ["N" if symbol == "N" else "Ga" for symbol in aln_atoms.get_chemical_symbols()]
        vca_atoms = aln_atoms.copy()
        vca_atoms.set_chemical_symbols(symbols)
        vca_atoms.set_cell(cell, scale_atoms=False)
        vca_atoms.set_scaled_positions(fractional)
        masses = vca_atoms.get_masses()
        vca_mass = x_al * AL_MASS_AMU + (1.0 - x_al) * GA_MASS_AMU
        masses[np.asarray(symbols) != "N"] = vca_mass
        vca_atoms.set_masses(masses)

        distances = vca_atoms.get_all_distances(mic=True)
        distances[distances < 1e-12] = np.inf
        minimum_distance = float(np.min(distances))
        if minimum_distance < 1.5:
            raise ValueError(f"Unphysical {composition} minimum distance: {minimum_distance} A")
        dataset = spglib.get_symmetry_dataset(
            (np.asarray(vca_atoms.cell), vca_atoms.get_scaled_positions(), vca_atoms.numbers),
            symprec=1e-5,
        )
        spacegroup_number = int(dataset.number)
        if spacegroup_number != 186:
            raise ValueError(f"Unexpected {composition} space group: {dataset.international} ({spacegroup_number})")

        vca_ph3 = _ph3(vca_atoms, fc2_supercell, fc3_supercell)
        vca_fc2 = np.ascontiguousarray(x_al * aln_fc2 + (1.0 - x_al) * gan_fc2_aligned)
        vca_fc3 = np.ascontiguousarray(x_al * aln_fc3 + (1.0 - x_al) * gan_fc3_aligned)
        write_maps = _write_numpy_ifcs(
            ifc_dir,
            vca_atoms,
            vca_ph3,
            vca_fc2,
            vca_fc3,
            fc2_supercell,
            fc3_supercell,
        )
        epsilon = x_al * aln_eps + (1.0 - x_al) * gan_eps_aligned
        born = x_al * aln_born + (1.0 - x_al) * gan_born_aligned
        born_path = ifc_dir / "BORNINFO.vca_aligned"
        np.savetxt(born_path, np.vstack([epsilon, born.reshape(-1, 3)]), fmt="%.16e")

        reload_fc2, reload_fc3, reload_map = _load_compact_ifcs(
            workdir,
            vca_ph3,
            fc2_supercell,
            fc3_supercell,
            ifc_dir_name="ifc_vca_numpy",
            structure_name="ifc_vca_numpy/POSCAR.vca_primitive",
        )
        fc2_roundtrip = float(np.max(np.abs(reload_fc2 - vca_fc2)))
        fc3_roundtrip = float(np.max(np.abs(reload_fc3 - vca_fc3)))
        if fc2_roundtrip > 1e-12 or fc3_roundtrip > 1e-12:
            raise ValueError(
                f"IFC write/read roundtrip failed for {composition}: "
                f"fc2={fc2_roundtrip:.3e}, fc3={fc3_roundtrip:.3e}"
            )
        report = {
            "composition": composition,
            "x_al": x_al,
            "method": "orientation-aligned endpoint IFC interpolation",
            "formula": "FC_VCA = x_Al FC_AlN + (1-x_Al) R(FC_GaN)",
            "endpoint_aln": str(endpoint_aln.resolve()),
            "endpoint_gan": str(endpoint_gan.resolve()),
            "fractional_transform_gan_to_aln": fractional_transform.tolist(),
            "cartesian_row_rotation_gan_to_aln": q_row.tolist(),
            "gan_alignment_translation_fractional": rounded_translation[0].tolist(),
            "rotation_determinant": float(np.linalg.det(q_row)),
            "minimum_interatomic_distance_A": minimum_distance,
            "spacegroup": {"international": str(dataset.international), "number": spacegroup_number},
            "vca_cation_mass_amu": vca_mass,
            "fc2_supercell": fc2_supercell,
            "fc3_supercell": fc3_supercell,
            "fc2_roundtrip_max_abs_error": fc2_roundtrip,
            "fc3_roundtrip_max_abs_error": fc3_roundtrip,
            "endpoint_load_mapping": {"AlN": aln_mapping, "GaN": gan_mapping},
            "gan_alignment_maps": alignment_maps,
            "output_maps": write_maps,
            "reload_mapping": reload_map,
            "checksums": {
                "second.npy": _sha256(ifc_dir / "second.npy"),
                "third.npz": _sha256(ifc_dir / "third.npz"),
                "POSCAR.vca_primitive": _sha256(ifc_dir / "POSCAR.vca_primitive"),
                "BORNINFO.vca_aligned": _sha256(born_path),
            },
        }
        (ifc_dir / "alignment_validation.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        generated.append(workdir)
        print(
            f"[DONE] {composition}: SG={dataset.international}, "
            f"min_dist={minimum_distance:.6f} A, roundtrip=({fc2_roundtrip:.1e},{fc3_roundtrip:.1e})"
        )
    return generated
