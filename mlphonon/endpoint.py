#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-bma")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "mathtext.fontset": "cm",
    "axes.linewidth": 1.3,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "lines.solid_capstyle": "round",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

DFT_COLOR = "#AD1A60"
MACE_COLOR = "#4260CF"
DP_COLOR = "#4260CF"

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from project_config import (  # noqa: E402
    BAND_PATH,
    CM_TO_THZ,
    DPA_MODEL,
    HBAR,
    ISOTOPE_G_FACTORS,
    KB,
    MACE_MODEL,
    MATERIALS,
    PROJECT_ROOT,
    THZ_TO_CM,
)
from mlphonon.banddos import (  # noqa: E402
    clip_curve_to_window,
    select_frequency_window,
)
from mlphonon.transport_plot import (  # noqa: E402
    filter_transport_modes_for_plot,
    load_transport_frequency_window,
    summarize_transport_filter,
    write_physical_window_tau_csv,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="MLIP relaxation, finite-displacement IFCs, phonons, and transport data."
    )
    parser.add_argument("material", choices=sorted(MATERIALS))
    parser.add_argument("--model-kind", choices=["mace", "dpa"], required=True)
    parser.add_argument("--model", default=None, help="Override model path.")
    parser.add_argument("--head", default=None, help="Model head, if needed.")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--quick", action="store_true", help="Use a reduced calculation size.")
    parser.add_argument("--run-label", default=None,
                        help="Override output suffix under results/material/model-kind.")
    parser.add_argument("--skip-relax", action="store_true")
    parser.add_argument("--allow-dpa-relax", action="store_true",
                        help="Allow DPA relaxation for models trained with virial/stress.")
    parser.add_argument("--skip-fd", action="store_true", help="Reuse cached kALDo FC2/FC3.")
    parser.add_argument("--skip-banddos", action="store_true")
    parser.add_argument("--banddos-from-kaldo", action="store_true",
                        help="Build phonopy band/DOS from cached kALDo second.npy instead of recomputing FC2.")
    parser.add_argument("--skip-transport", action="store_true")
    parser.add_argument("--fc-only", action="store_true",
                        help="Build/cache kALDo FC2/FC3 and stop before transport.")
    parser.add_argument("--fc2-supercell", nargs=3, type=int, default=None)
    parser.add_argument("--fc3-supercell", nargs=3, type=int, default=None)
    parser.add_argument("--kpts", nargs=3, type=int, default=None)
    parser.add_argument("--dos-mesh", nargs=3, type=int, default=None)
    parser.add_argument("--temps", nargs="+", type=int, default=[300])
    parser.add_argument("--method", choices=["rta", "sc", "inverse", "qhgk"], default="rta")
    parser.add_argument("--finite-length-angstrom", nargs=3, type=float, default=None,
                        help="kALDo finite-size length in Angstrom for x y z; use 0 for infinity.")
    parser.add_argument("--finite-length-method", choices=["ms", "ballistic"], default="ms")
    parser.add_argument("--kaldo-unfolding", choices=["auto", "always", "never"], default="auto",
                        help="Use kALDo Wigner-Seitz unfolding for harmonic q-space assembly.")
    parser.add_argument(
        "--kaldo-max-frequency-from-banddos",
        action="store_true",
        help=(
            "Diagnostic opt-in: exclude kALDo modes above the band/DOS physical "
            "support. This changes physical_mode and may change kappa."
        ),
    )
    parser.add_argument(
        "--allow-symmetry-broken-kmesh",
        action="store_true",
        help="Allow a transport mesh that does not preserve every crystal point-group operation.",
    )
    parser.add_argument(
        "--projected-dos-plot",
        action="store_true",
        help="Plot element-projected DOS curves in addition to the total DOS curve.",
    )
    parser.add_argument("--borninfo", type=Path, default=None,
                        help="Override the default DFT-reference BORNINFO used for NAC.")
    parser.add_argument("--align-basis-to", choices=sorted(MATERIALS), default=None,
                        help="Rebuild the input primitive using this material's fractional basis "
                             "with the current material's cell and species. Use this before "
                             "generating endpoint IFCs that will be index-averaged in VCA.")
    parser.add_argument("--delta-shift-fc2", type=float, default=1e-3)
    parser.add_argument("--delta-shift-fc3", type=float, default=1e-3)
    parser.add_argument("--no-isotopes", action="store_true")
    parser.add_argument("--plot-only", action="store_true",
                        help="Regenerate figures from existing band/DOS and transport CSVs.")
    return parser.parse_args()


def model_label(model_kind):
    return {"mace": "MACE", "dpa": "DP"}[model_kind]


def model_color(model_kind):
    return {"mace": MACE_COLOR, "dpa": DP_COLOR}[model_kind]


def cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def choose_device(requested):
    if requested == "auto":
        return "cuda" if cuda_available() else "cpu"
    return requested


def build_calculator(model_kind, model_path, device, head=None):
    if model_kind == "mace":
        from mace.calculators import MACECalculator
        kwargs = {"model_paths": str(model_path), "device": device, "default_dtype": "float64"}
        # Multi-head foundation models (e.g. mace-mh-1) require an explicit head.
        if head:
            kwargs["head"] = head
        return MACECalculator(**kwargs)

    from deepmd.calculator import DP
    # Match the model's declared type_map exactly. This keeps ASE/phonopy atom
    # ordering from changing DeePMD type IDs implicitly.
    kwargs = {"model": str(model_path), "type_dict": {"In": 0, "Ga": 1, "As": 2}}
    if head:
        kwargs["head"] = head
    return DP(**kwargs)


def read_borninfo(path, n_atoms):
    data = np.loadtxt(path)
    epsilon = data[:3].reshape(3, 3)
    born = data[3:].reshape(n_atoms, 3, 3)
    return epsilon, born


def projected_dos_frequency_bounds(phonon):
    """Choose a DOS window from the actual harmonic mesh support."""
    frequencies = np.asarray(phonon.get_mesh_dict()["frequencies"], dtype=float)
    finite = frequencies[np.isfinite(frequencies)]
    if finite.size == 0:
        raise ValueError("Phonopy mesh contains no finite frequencies")
    margin = max(0.5, 0.03 * float(np.ptp(finite)))
    return min(-1.0, float(finite.min()) - margin), float(finite.max()) + margin


def align_basis_to_reference(atoms, material, reference_material):
    from ase import Atoms
    from ase.io import read as ase_read

    ref_cfg = MATERIALS[reference_material]
    ref_atoms = ase_read(str(ref_cfg["poscar"]), format="vasp")
    material_symbols = MATERIALS[material]["symbols"]
    cation = next(symbol for symbol in material_symbols if symbol != "As")

    ref_symbols = ref_atoms.get_chemical_symbols()
    ref_scaled = ref_atoms.get_scaled_positions(wrap=True)
    aligned_symbols = ["As" if symbol == "As" else cation for symbol in ref_symbols]
    aligned = Atoms(
        symbols=aligned_symbols,
        scaled_positions=ref_scaled,
        cell=atoms.cell.array,
        pbc=True,
    )

    try:
        import spglib

        for label, candidate in (("input", atoms), ("aligned", aligned)):
            cell = (
                candidate.cell.array,
                candidate.get_scaled_positions(wrap=True),
                candidate.get_atomic_numbers(),
            )
            dataset = spglib.get_symmetry_dataset(cell, symprec=1e-5)
            if dataset is not None:
                print(
                    f"[INFO] {label} basis space group: "
                    f"{dataset['international']} ({dataset['number']})"
                )
    except Exception as exc:
        print(f"[WARN] spglib basis check skipped: {exc}")

    print(
        f"[INFO] aligned {material} basis to {reference_material}: "
        f"{list(zip(aligned.get_chemical_symbols(), aligned.get_scaled_positions(wrap=True).round(8).tolist()))}"
    )
    return aligned


def attach_nac_to_kaldo(forceconstants, borninfo, n_atoms):
    if not borninfo.exists():
        print(f"[WARN] BORNINFO not found: {borninfo}; NAC disabled.")
        return
    epsilon, born = read_borninfo(borninfo, n_atoms)
    forceconstants.second.atoms.info["dielectric"] = epsilon
    forceconstants.second.atoms.set_array("charges", born, shape=(3, 3))
    print(f"[INFO] NAC enabled: eps_xx={epsilon[0, 0]:.4f}, Z0_xx={born[0, 0, 0]:.4f}")


def relax_structure(atoms, calc, out_poscar, log_path):
    from ase.filters import FrechetCellFilter
    from ase.io import write as ase_write
    from ase.optimize import BFGS

    atoms.calc = calc
    f0 = np.abs(atoms.get_forces()).max()
    print(f"[INFO] relax initial max|F|={f0:.4e} eV/A")
    filt = FrechetCellFilter(atoms)
    BFGS(filt, logfile=str(log_path)).run(fmax=1e-4, steps=500)
    f1 = np.abs(atoms.get_forces()).max()
    ase_write(str(out_poscar), atoms, format="vasp")
    print(f"[INFO] relaxed saved: {out_poscar}; max|F|={f1:.4e} eV/A")
    return atoms


def ase_to_phonopy_atoms(atoms):
    from phonopy.structure.atoms import PhonopyAtoms

    return PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        positions=atoms.get_positions(),
        cell=atoms.cell.array,
    )


def run_phonopy_banddos(atoms, calc, cfg, outdir, supercell, dos_mesh, quick=False):
    import phonopy
    from ase import Atoms as ASEAtoms
    from phonopy.file_IO import write_force_constants_to_hdf5
    from phonopy.interface.calculator import get_default_physical_units

    print("[BANDDOS] Computing MLIP FC2 with phonopy finite differences")
    ph_atoms = ase_to_phonopy_atoms(atoms)
    phonon = phonopy.Phonopy(ph_atoms, supercell_matrix=np.diag(supercell), primitive_matrix="auto")
    phonon.generate_displacements(distance=0.01)
    supercells = phonon.supercells_with_displacements
    print(f"[BANDDOS] displaced supercells: {len(supercells)}")

    forces = []
    for i, sc in enumerate(supercells):
        sc_ase = ASEAtoms(symbols=sc.symbols, positions=sc.positions, cell=sc.cell, pbc=True)
        sc_ase.calc = calc
        forces.append(sc_ase.get_forces())
        if (i + 1) % 5 == 0 or (i + 1) == len(supercells):
            print(f"[BANDDOS] {i + 1}/{len(supercells)} max|F|={np.abs(forces[-1]).max():.4e}")

    phonon.forces = forces
    phonon.produce_force_constants()
    phonon.symmetrize_force_constants()
    write_force_constants_to_hdf5(
        phonon.force_constants, filename=str(outdir / "phonopy_force_constants.hdf5")
    )

    if cfg["borninfo"].exists():
        eps, born = read_borninfo(cfg["borninfo"], len(atoms))
        phonon.nac_params = {
            "born": born,
            "dielectric": eps,
            "factor": get_default_physical_units("vasp")["nac_factor"],
        }
        with open(outdir / "BORN", "w", encoding="utf-8") as fh:
            fh.write(f"# epsilon and Born charges read from {cfg['borninfo']} for NAC\n")
            fh.write("  " + "  ".join(f"{v:.10f}" for v in eps.ravel()) + "\n")
            for z in born:
                fh.write("  " + "  ".join(f"{v:.10f}" for v in z.ravel()) + "\n")

    band_arms = [np.linspace(start, end, npts) for start, end, npts in BAND_PATH]
    phonon.run_band_structure(band_arms, with_eigenvectors=False, is_band_connection=False)
    phonon.write_yaml_band_structure(filename=str(outdir / "band.yaml"))

    mesh = [max(3, int(x)) for x in dos_mesh]
    if quick:
        mesh = [3, 3, 3]
    phonon.run_mesh(mesh, with_eigenvectors=True, is_mesh_symmetry=False)
    dos_min, dos_max = projected_dos_frequency_bounds(phonon)
    phonon.run_projected_dos(freq_min=dos_min, freq_max=dos_max, freq_pitch=0.02)
    pdos = phonon.get_projected_dos_dict()
    data = np.column_stack([pdos["frequency_points"], pdos["projected_dos"].T])
    header = "freq_THz " + " ".join(
        f"atom{i + 1}_{s}" for i, s in enumerate(atoms.get_chemical_symbols())
    )
    np.savetxt(outdir / "projected_dos.dat", data, header=header, fmt="%.8e")
    print(f"[BANDDOS] wrote {outdir / 'band.yaml'} and projected_dos.dat")


def kaldo_second_to_phonopy_compact(second, n_prim, column_order=None):
    arr = np.asarray(second)
    if arr.ndim == 6 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 5 or arr.shape[0] != n_prim or arr.shape[1] != 3 or arr.shape[3] != n_prim or arr.shape[4] != 3:
        raise ValueError(
            "Unsupported kALDo second.npy shape. Expected "
            f"(1,{n_prim},3,n_cells,{n_prim},3) or ({n_prim},3,n_cells,{n_prim},3); got {second.shape}"
        )
    # kALDo stores (i_prim, alpha, cell, j_prim, beta). Phonopy compact
    # force constants use (i_prim, j_super, alpha, beta). The initial flatten
    # is in kALDo/ASE repeat order; caller can reorder columns into phonopy's
    # own supercell order with column_order.
    compact = arr.transpose(0, 2, 3, 1, 4).reshape(n_prim, arr.shape[2] * n_prim, 3, 3)
    if column_order is not None:
        compact = compact[:, np.asarray(column_order, dtype=int), :, :]
    return compact


def phonopy_to_kaldo_supercell_order(phonopy_supercell, replicated_atoms):
    ph_symbols = list(phonopy_supercell.symbols)
    rep_symbols = replicated_atoms.get_chemical_symbols()
    if len(ph_symbols) != len(rep_symbols):
        raise ValueError(f"Supercell size mismatch: phonopy={len(ph_symbols)} kALDo={len(rep_symbols)}")

    cell = np.array(phonopy_supercell.cell)
    inv_cell = np.linalg.inv(cell)
    ph_frac = np.array(phonopy_supercell.positions) @ inv_cell
    rep_frac = np.array(replicated_atoms.positions) @ inv_cell
    used = np.zeros(len(rep_symbols), dtype=bool)
    order = []

    for i, symbol in enumerate(ph_symbols):
        candidates = [j for j, rep_symbol in enumerate(rep_symbols) if rep_symbol == symbol and not used[j]]
        if not candidates:
            raise ValueError(f"No unmatched kALDo replicated atom for phonopy atom {i} ({symbol})")
        delta = rep_frac[candidates] - ph_frac[i]
        delta -= np.round(delta)
        dist = np.linalg.norm(delta @ cell, axis=1)
        best_local = int(np.argmin(dist))
        best = candidates[best_local]
        if dist[best_local] > 1e-5:
            raise ValueError(
                f"Could not match phonopy supercell atom {i} ({symbol}); "
                f"closest kALDo atom {best} is {dist[best_local]:.3e} A away"
            )
        used[best] = True
        order.append(best)

    return order


def run_phonopy_banddos_from_kaldo(atoms, cfg, outdir, supercell, dos_mesh, quick=False):
    import phonopy
    from ase.io import read as ase_read
    from phonopy.file_IO import write_force_constants_to_hdf5
    from phonopy.interface.calculator import get_default_physical_units

    fc2_path = outdir / "kaldo_ifc" / "second.npy"
    replicated_path = outdir / "kaldo_ifc" / "replicated_atoms.xyz"
    if not fc2_path.exists():
        raise FileNotFoundError(f"Cached kALDo FC2 not found: {fc2_path}")
    if not replicated_path.exists():
        raise FileNotFoundError(f"Cached kALDo replicated atoms not found: {replicated_path}")

    print(f"[BANDDOS] Reusing cached kALDo FC2: {fc2_path}")
    ph_atoms = ase_to_phonopy_atoms(atoms)
    phonon = phonopy.Phonopy(ph_atoms, supercell_matrix=np.diag(supercell), primitive_matrix="auto")
    second = np.load(fc2_path, allow_pickle=True)
    replicated_atoms = ase_read(str(replicated_path), format="extxyz")
    column_order = phonopy_to_kaldo_supercell_order(phonon.supercell, replicated_atoms)
    phonon.force_constants = kaldo_second_to_phonopy_compact(second, len(atoms), column_order=column_order)
    phonon.symmetrize_force_constants()
    write_force_constants_to_hdf5(
        phonon.force_constants, filename=str(outdir / "phonopy_force_constants.hdf5")
    )

    if cfg["borninfo"].exists():
        eps, born = read_borninfo(cfg["borninfo"], len(atoms))
        phonon.nac_params = {
            "born": born,
            "dielectric": eps,
            "factor": get_default_physical_units("vasp")["nac_factor"],
        }
        with open(outdir / "BORN", "w", encoding="utf-8") as fh:
            fh.write(f"# epsilon and Born charges read from {cfg['borninfo']} for NAC\n")
            fh.write("  " + "  ".join(f"{v:.10f}" for v in eps.ravel()) + "\n")
            for z in born:
                fh.write("  " + "  ".join(f"{v:.10f}" for v in z.ravel()) + "\n")

    band_arms = [np.linspace(start, end, npts) for start, end, npts in BAND_PATH]
    phonon.run_band_structure(band_arms, with_eigenvectors=False, is_band_connection=False)
    phonon.write_yaml_band_structure(filename=str(outdir / "band.yaml"))

    mesh = [max(3, int(x)) for x in dos_mesh]
    if quick:
        mesh = [3, 3, 3]
    phonon.run_mesh(mesh, with_eigenvectors=True, is_mesh_symmetry=False)
    dos_min, dos_max = projected_dos_frequency_bounds(phonon)
    phonon.run_projected_dos(freq_min=dos_min, freq_max=dos_max, freq_pitch=0.02)
    pdos = phonon.get_projected_dos_dict()
    data = np.column_stack([pdos["frequency_points"], pdos["projected_dos"].T])
    header = "freq_THz " + " ".join(
        f"atom{i + 1}_{s}" for i, s in enumerate(atoms.get_chemical_symbols())
    )
    np.savetxt(outdir / "projected_dos.dat", data, header=header, fmt="%.8e")
    print(f"[BANDDOS] wrote {outdir / 'band.yaml'} and projected_dos.dat from cached FC2")


def load_phonopy_band_yaml(path):
    import yaml

    with open(path, encoding="utf-8") as fh:
        bd = yaml.safe_load(fh)
    nq = bd["segment_nqpoint"]
    phonons = bd["phonon"]
    raise RuntimeError("load_phonopy_band_yaml requires explicit segment_x_ranges")


def load_phonopy_band_yaml_segments(path, segment_x_ranges, sort_frequencies=True):
    import yaml

    with open(path, encoding="utf-8") as fh:
        bd = yaml.safe_load(fh)
    nq = bd["segment_nqpoint"]
    phonons = bd["phonon"]
    segments = []
    start = 0
    for npts, (x0, x1) in zip(nq, segment_x_ranges):
        rows = phonons[start:start + npts]
        freq_cm = np.array([[b["frequency"] * THZ_TO_CM for b in p["band"]] for p in rows])
        if sort_frequencies:
            freq_cm = np.sort(freq_cm, axis=1)
        x = np.linspace(x0, x1, npts)
        segments.append((x, freq_cm))
        start += npts
    return segments


def load_alm_bands(path):
    data = np.loadtxt(path, comments="#")
    x = data[:, 0]
    freqs = data[:, 1:]
    return [(x, freqs[:, i]) for i in range(freqs.shape[1])]


def load_alm_bands_segments(path, segment_ranges):
    data = np.loadtxt(path, comments="#")
    x_all = data[:, 0]
    freqs_all = data[:, 1:]
    # ALM encodes path discontinuities such as A|L and M|H as repeated
    # k-axis distances. Split by row order at those repeated/non-increasing
    # distances; selecting by x-value masks would put both sides of the break
    # in one segment and draw artificial vertical DFT lines.
    chunks = []
    start = 0
    for i in range(1, len(x_all)):
        if x_all[i] <= x_all[i - 1] + 1e-10:
            chunks.append((x_all[start:i], freqs_all[start:i]))
            start = i
    chunks.append((x_all[start:], freqs_all[start:]))

    out = []
    if len(chunks) > 1:
        for x, freqs in chunks:
            if len(x) == 0:
                continue
            out.append((x, freqs))
        return out

    for x0, x1 in segment_ranges:
        mask = (x_all >= x0 - 1e-8) & (x_all <= x1 + 1e-8)
        x = x_all[mask]
        freqs = freqs_all[mask]
        if len(x) == 0:
            continue
        order = np.argsort(x)
        out.append((x[order], freqs[order]))
    return out


def load_total_dos(path, model=False):
    data = np.loadtxt(path, comments="#")
    if model:
        # phonopy projected_dos.dat is DOS per THz. After x-axis conversion
        # THz -> cm^-1, density must transform as g_cm = g_THz / 33.356.
        total = data[:, 1:].sum(axis=1)
        return data[:, 0] * THZ_TO_CM, total / THZ_TO_CM
    return data[:, 0], data[:, 1]


def projected_dos_columns(path):
    """Return element-summed projected DOS curves from phonopy projected_dos.dat."""
    labels = None
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped.startswith("#"):
                labels = stripped.lstrip("#").split()
            elif stripped:
                break
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data[None, :]
    if labels is None or len(labels) != data.shape[1]:
        labels = ["freq_THz"] + [f"atom{i}" for i in range(1, data.shape[1])]

    x_cm = data[:, 0] * THZ_TO_CM
    curves = []
    grouped = {}
    order = []
    for idx, label in enumerate(labels[1:], start=1):
        if "_" in label:
            element = label.rsplit("_", 1)[-1]
        else:
            element = label
        if element not in grouped:
            grouped[element] = np.zeros_like(data[:, idx], dtype=float)
            order.append(element)
        grouped[element] += data[:, idx]
    for element in order:
        # phonopy projected_dos.dat is DOS per THz; convert density to per cm^-1.
        curves.append((element, x_cm, grouped[element] / THZ_TO_CM))
    return curves


def plot_dos_curves(dos_ax, curves, window, *, total_color, total_label, projected=False, dft_curve=None):
    if dft_curve is not None:
        x, y = clip_curve_to_window(dft_curve[0], dft_curve[1], window)
        dos_ax.plot(y, x, color=DFT_COLOR, lw=1.35, label="DFT")
    x, y = curves["total"]
    x, y = clip_curve_to_window(x, y, window)
    dos_ax.plot(y, x, color=total_color, lw=1.45, label=total_label)
    if not projected:
        return
    palette = ["#0077c2", "#d65244", "#7fb5dd", "#c6c6c6", "#005990", "#b9d7eb"]
    for i, (element, px, py) in enumerate(curves["projected"]):
        px, py = clip_curve_to_window(px, py, window)
        dos_ax.plot(
            py,
            px,
            color=palette[i % len(palette)],
            lw=1.05,
            ls="--",
            alpha=0.95,
            label=element,
        )


def dft_ticks_from_bands(path):
    labels = None
    ticks = None
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped.startswith("#"):
                break
            body = stripped[1:].strip()
            parts = body.split()
            if len(parts) >= 7 and parts[0].upper() in {"GAMMA", "Γ"}:
                labels = parts
            else:
                try:
                    vals = [float(x) for x in parts]
                except ValueError:
                    continue
                if len(vals) >= 7:
                    ticks = vals
    if ticks is None or labels is None:
        raise ValueError(f"Could not parse k-path labels/ticks from DFT bands file: {path}")

    def pretty_label(label):
        if label.upper() in {"GAMMA", "Γ"}:
            return r"$\Gamma$"
        if "_" in label:
            base, sub = label.split("_", 1)
            return rf"{base}$_{{{sub}}}$"
        return label

    labels = [pretty_label(label) for label in labels]

    segment_ranges = [
        (float(ticks[i]), float(ticks[i + 1]))
        for i in range(len(ticks) - 1)
        if abs(float(ticks[i + 1]) - float(ticks[i])) > 1e-8
    ]

    xticks = []
    xlabels = []
    for tick, label in zip(ticks, labels):
        tick = float(tick)
        if xticks and abs(xticks[-1] - tick) < 1e-8:
            xlabels[-1] = f"{xlabels[-1]}|{label}"
        else:
            xticks.append(tick)
            xlabels.append(label)

    return xticks, xlabels, segment_ranges


def compact_kpath_ticks(xticks, xlabels, min_fraction=0.085):
    """Merge visually crowded high-symmetry labels while keeping path lines."""
    if not xticks:
        return xticks, xlabels
    span = max(xticks[-1] - xticks[0], 1e-12)
    min_sep = span * min_fraction
    groups = []
    cur_x = [xticks[0]]
    cur_l = [xlabels[0]]
    for x, label in zip(xticks[1:], xlabels[1:]):
        if x - cur_x[-1] < min_sep:
            cur_x.append(x)
            cur_l.append(label)
        else:
            groups.append((cur_x, cur_l))
            cur_x = [x]
            cur_l = [label]
    groups.append((cur_x, cur_l))

    out_x = []
    out_l = []
    for xs, labels in groups:
        out_x.append(float(np.mean(xs)))
        merged = []
        for label in labels:
            for part in str(label).split("|"):
                if part and part not in merged:
                    merged.append(part)
        out_l.append("|".join(merged))
    return out_x, out_l


def set_axes_style(ax):
    ax.minorticks_on()
    ax.tick_params(which="major", direction="in", top=True, right=True, length=4.0, width=1.2)
    ax.tick_params(which="minor", direction="in", top=True, right=True, length=2.2, width=1.0)
    for spine in ax.spines.values():
        spine.set_linewidth(1.3)


def save_pubfig(fig, path):
    path = Path(path)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"[PLOT] wrote {path}")


def plot_banddos(outdir, cfg, title, model_kind):
    import matplotlib.gridspec as gridspec

    label = model_label(model_kind)
    color = model_color(model_kind)
    projected_plot = bool(cfg.get("projected_dos_plot", False))
    xticks, xlabels, segment_ranges = dft_ticks_from_bands(cfg["dft_bands"])
    # Keep labels on their exact high-symmetry coordinates. dft_ticks_from_bands
    # already merges truly repeated coordinates such as A|L and M|H; averaging
    # nearby but distinct ticks would make labels drift away from the k-points.
    label_ticks, label_text = xticks, xlabels
    model_segments = load_phonopy_band_yaml_segments(outdir / "band.yaml", segment_ranges)
    model_dos_x, model_dos_y = load_total_dos(outdir / "projected_dos.dat", model=True)
    model_dos_curves = {
        "total": (model_dos_x, model_dos_y),
        "projected": projected_dos_columns(outdir / "projected_dos.dat") if projected_plot else [],
    }
    tail_mass_fraction = float(cfg.get("dos_tail_mass_fraction", 1e-6))
    margin_fraction = float(cfg.get("frequency_margin_fraction", 0.02))
    minimum_margin = float(cfg.get("frequency_minimum_margin_cm", 5.0))
    model_window, model_window_report = select_frequency_window(
        [freqs for _, freqs in model_segments],
        [("model", model_dos_x, model_dos_y)],
        tail_mass_fraction=tail_mass_fraction,
        margin_fraction=margin_fraction,
        minimum_margin=minimum_margin,
    )
    fig = plt.figure(figsize=(7.2, 3.6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3.7, 1.0], wspace=0.06)
    ax = fig.add_subplot(gs[0])
    dos_ax = fig.add_subplot(gs[1], sharey=ax)

    for x, freqs in model_segments:
        for branch in range(freqs.shape[1]):
            ax.plot(x, freqs[:, branch], color=color, lw=1.05)
    plot_dos_curves(
        dos_ax,
        model_dos_curves,
        model_window,
        total_color=color,
        total_label=label,
        projected=projected_plot,
    )

    for xb in xticks[1:-1]:
        ax.axvline(xb, color="0.72", ls="--", lw=0.75)
    ax.axhline(0, color="0.45", ls="--", lw=0.75)
    ax.set_xlim(xticks[0], xticks[-1])
    ax.set_xticks(label_ticks)
    ax.set_xticklabels(label_text, fontsize=8)
    ax.set_ylabel("Frequency (cm$^{-1}$)")
    ax.set_title(title)
    ax.set_ylim(*model_window)
    dos_ax.set_xlabel("DOS")
    dos_ax.set_xticks([])
    dos_ax.tick_params(labelleft=False)
    dos_ax.legend(frameon=False, loc="upper right")
    set_axes_style(ax)
    set_axes_style(dos_ax)
    out = outdir / "fig_banddos_mlip.png"
    save_pubfig(fig, out)
    plt.close(fig)

    frequency_report = {"model_only": model_window_report}
    if cfg["dft_bands"].exists() and cfg["dft_dos"].exists():
        dft_segments = load_alm_bands_segments(cfg["dft_bands"], segment_ranges)
        dft_dos_x, dft_dos_y = load_total_dos(cfg["dft_dos"], model=False)
        compare_window, compare_window_report = select_frequency_window(
            [freqs for _, freqs in model_segments]
            + [freqs for _, freqs in dft_segments],
            [
                ("model", model_dos_x, model_dos_y),
                ("dft", dft_dos_x, dft_dos_y),
            ],
            tail_mass_fraction=tail_mass_fraction,
            margin_fraction=margin_fraction,
            minimum_margin=minimum_margin,
        )
        fig = plt.figure(figsize=(7.2, 3.6))
        gs = gridspec.GridSpec(1, 2, width_ratios=[3.7, 1.0], wspace=0.06)
        ax = fig.add_subplot(gs[0])
        dos_ax = fig.add_subplot(gs[1], sharey=ax)
        for x, freqs in dft_segments:
            for branch in range(freqs.shape[1]):
                ax.plot(x, freqs[:, branch], color=DFT_COLOR, lw=1.0, alpha=0.92)
        for x, freqs in model_segments:
            for branch in range(freqs.shape[1]):
                ax.plot(x, freqs[:, branch], color=color, lw=0.9, alpha=0.95)
        plot_dos_curves(
            dos_ax,
            model_dos_curves,
            compare_window,
            total_color=color,
            total_label=label,
            projected=projected_plot,
            dft_curve=(dft_dos_x, dft_dos_y),
        )
        for xb in xticks[1:-1]:
            ax.axvline(xb, color="0.72", ls="--", lw=0.75)
        ax.axhline(0, color="0.45", ls="--", lw=0.75)
        ax.set_xlim(xticks[0], xticks[-1])
        ax.set_xticks(label_ticks)
        ax.set_xticklabels(label_text, fontsize=8)
        ax.set_ylabel("Frequency (cm$^{-1}$)")
        ax.set_title(title + " vs DFT")
        ax.set_ylim(*compare_window)
        dos_ax.set_xlabel("DOS")
        dos_ax.set_xticks([])
        dos_ax.tick_params(labelleft=False)
        dos_ax.legend(frameon=False, loc="upper right")
        ax.plot([], [], color=DFT_COLOR, lw=1.2, label="DFT")
        ax.plot([], [], color=color, lw=1.2, label=label)
        ax.legend(frameon=False, loc="upper left")
        set_axes_style(ax)
        set_axes_style(dos_ax)
        out = outdir / "fig_banddos_compare_dft.png"
        save_pubfig(fig, out)
        plt.close(fig)
        frequency_report["comparison"] = compare_window_report

    with open(outdir / "banddos_frequency_window.json", "w", encoding="utf-8") as handle:
        json.dump(frequency_report, handle, indent=2)
    print(f"[BANDDOS] frequency-window report written in {outdir}")


def heat_capacity(freq_thz, temp):
    if temp == 0:
        return np.zeros_like(freq_thz)
    freq_hz = freq_thz * 1e12
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        x = HBAR * 2.0 * np.pi * freq_hz / (KB * temp)
        x = np.clip(x, 1e-10, 500.0)
        exp_x = np.exp(x)
        return np.where(freq_thz > 0.1, KB * x**2 * exp_x / (exp_x - 1.0) ** 2, KB)


def mode_kappa_from_tau(freq_thz, vel_ang_ps, tau_ps, temp, nk, volume_a3):
    v_ms = vel_ang_ps * 100.0
    c = heat_capacity(freq_thz, temp)
    tau_s = tau_ps * 1e-12
    return c[:, :, None, None] * v_ms[:, :, :, None] * v_ms[:, :, None, :] * tau_s[:, :, None, None] / (nk * volume_a3 * 1e-30)


def reshape_kaldo_vector(quantity, freq_shape, name):
    arr = np.array(quantity)
    if arr.shape[:2] == freq_shape:
        return arr
    if arr.shape[0] == int(np.prod(freq_shape)):
        return arr.reshape(freq_shape + arr.shape[1:])
    raise ValueError(f"Cannot reshape kALDo {name} with shape {arr.shape} to frequency shape {freq_shape}")


def kaldo_max_frequency_from_banddos(outdir):
    """Return a kALDo max_frequency cutoff in THz from the band/DOS physical support."""
    report_path = outdir / "banddos_frequency_window.json"
    upper_cm = None
    source = None
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            section = report.get("model_only") or report.get("comparison") or {}
            upper_cm = section.get("physical_support_max_cm-1", section.get("band_max_cm-1"))
            if upper_cm is not None:
                upper_cm = float(upper_cm)
                source = str(report_path)
        except Exception as exc:
            print(f"[WARN] could not read band/DOS frequency window: {exc}")
            upper_cm = None

    if upper_cm is None and (outdir / "band.yaml").exists():
        try:
            import yaml
            with open(outdir / "band.yaml", encoding="utf-8") as handle:
                band = yaml.safe_load(handle)
            values = [
                float(mode["frequency"]) * THZ_TO_CM
                for point in band.get("phonon", [])
                for mode in point.get("band", [])
                if np.isfinite(float(mode["frequency"]))
            ]
            if values:
                upper_cm = max(values)
                source = str(outdir / "band.yaml")
        except Exception as exc:
            print(f"[WARN] could not infer max frequency from band.yaml: {exc}")
            upper_cm = None

    if upper_cm is None or not np.isfinite(upper_cm) or upper_cm <= 0.0:
        return None, None
    return upper_cm * CM_TO_THZ, {
        "source": source,
        "max_frequency_cm-1": upper_cm,
    }


def should_use_kaldo_unfolding(atoms, mode="auto", angle_tol_deg=1e-3):
    if mode == "always":
        return True, "forced by --kaldo-unfolding=always"
    if mode == "never":
        return False, "disabled by --kaldo-unfolding=never"
    angles = np.asarray(atoms.cell.angles(), dtype=float)
    non_orthogonal = np.any(np.abs(angles - 90.0) > angle_tol_deg)
    if non_orthogonal:
        return True, f"non-orthogonal cell angles={np.round(angles, 6).tolist()}"
    return False, f"orthogonal cell angles={np.round(angles, 6).tolist()}"


def validate_kmesh_symmetry(atoms, kpts, symprec=1e-5):
    """Return how many crystal rotations map the reciprocal mesh onto itself."""
    import spglib

    symmetry = spglib.get_symmetry(
        (atoms.cell.array, atoms.get_scaled_positions(), atoms.numbers),
        symprec=symprec,
    )
    if symmetry is None:
        raise RuntimeError("spglib could not determine symmetry for k-mesh validation")
    mesh = np.asarray(kpts, dtype=float)
    mesh_matrix = np.diag(mesh)
    inverse_mesh_matrix = np.diag(1.0 / mesh)
    preserved = 0
    for rotation in symmetry["rotations"]:
        reciprocal_rotation = np.linalg.inv(rotation).T
        mapping = inverse_mesh_matrix @ reciprocal_rotation @ mesh_matrix
        if np.allclose(mapping, np.rint(mapping), atol=1e-8):
            preserved += 1
    total = len(symmetry["rotations"])
    return preserved, total


def write_spectral_csv(outdir, temp, freq_cm, mfp_nm, kappa_mode):
    kave = ((kappa_mode[:, :, 0, 0] + kappa_mode[:, :, 1, 1] + kappa_mode[:, :, 2, 2]) / 3.0).ravel()

    freq = freq_cm.ravel()
    order = np.argsort(freq)
    out = outdir / f"kspec_frequency_{temp}K.csv"
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(f"# Temperature : {temp}\nfrequency,kcumu,kspec\n")
        cum = 0.0
        for f, k in zip(freq[order], kave[order]):
            cum += k
            fh.write(f"{f:.7e},{cum:.7e},{k:.7e}\n")

    mfp = mfp_nm.ravel()
    mask = np.isfinite(mfp) & (mfp > 0)
    order = np.argsort(mfp[mask])
    out = outdir / f"kspec_mfp_{temp}K.csv"
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(f"# Temperature : {temp}\nmfp,kcumu,kspec\n")
        cum = 0.0
        for x, k in zip(mfp[mask][order], kave[mask][order]):
            cum += k
            fh.write(f"{x:.7e},{cum:.7e},{k:.7e}\n")


def run_kaldo_transport(
    atoms, calc, cfg, outdir, fc2, fc3, kpts, temps, method, delta2, delta3,
    skip_fd, isotopes, finite_length_angstrom=None, finite_length_method="ms",
    kaldo_unfolding="auto",
    max_frequency_from_banddos=False,
):
    from kaldo.conductivity import Conductivity
    from kaldo.forceconstants import ForceConstants
    from kaldo.phonons import Phonons

    fc_dir = outdir / "kaldo_ifc"
    fc_dir.mkdir(parents=True, exist_ok=True)

    if skip_fd:
        print(f"[KALDO] loading cached IFCs from {fc_dir}")
        kfc = ForceConstants.from_folder(
            folder=str(fc_dir), supercell=tuple(fc2), third_supercell=tuple(fc3),
            format="numpy", is_acoustic_sum=True
        )
    else:
        print(f"[KALDO] computing native finite-difference FC2={fc2}, FC3={fc3}")
        kfc = ForceConstants(
            atoms=atoms, supercell=np.array(fc2), third_supercell=np.array(fc3), folder=str(fc_dir)
        )
        t0 = time.time()
        kfc.second.calculate(calc, delta_shift=delta2)
        print(f"[KALDO] FC2 done in {(time.time() - t0) / 60:.2f} min")
        t0 = time.time()
        kfc.third.calculate(calc, delta_shift=delta3)
        print(f"[KALDO] FC3 done in {(time.time() - t0) / 60:.2f} min")

    attach_nac_to_kaldo(kfc, cfg["borninfo"], len(atoms))

    if getattr(run_kaldo_transport, "fc_only", False):
        print(f"[KALDO] FC-only requested; cached IFCs are ready in {fc_dir}")
        return

    try:
        del calc
        import torch
        torch.cuda.empty_cache()
        print("[KALDO] released PyTorch CUDA cache")
    except Exception:
        pass

    kappa_rows = []
    nk = int(np.prod(kpts))
    preserved_operations, total_operations = validate_kmesh_symmetry(atoms, kpts)
    print(
        "[KALDO] k-mesh point-group operations: "
        f"{preserved_operations}/{total_operations}"
    )
    if (preserved_operations != total_operations
            and not getattr(run_kaldo_transport, "allow_symmetry_broken_kmesh", False)):
        raise RuntimeError(
            f"Transport mesh {tuple(kpts)} breaks crystal point-group symmetry "
            f"({preserved_operations}/{total_operations} operations preserved). "
            "Choose a symmetry-compatible mesh or explicitly pass "
            "--allow-symmetry-broken-kmesh for a diagnostic run."
        )
    with open(outdir / "kaldo_kmesh_symmetry.md", "w", encoding="utf-8") as fh:
        fh.write("# kALDo k-mesh symmetry\n\n")
        fh.write(f"- `kpts` = `{tuple(kpts)}`\n")
        fh.write(f"- point-group operations = `{preserved_operations}/{total_operations}`\n")
        fh.write(f"- symmetry preserving = `{preserved_operations == total_operations}`\n")
    use_unfolding, unfolding_reason = should_use_kaldo_unfolding(atoms, kaldo_unfolding)
    print(f"[KALDO] is_unfolding={use_unfolding} ({unfolding_reason})")
    with open(outdir / "kaldo_unfolding.md", "w", encoding="utf-8") as fh:
        fh.write("# kALDo Unfolding\n\n")
        fh.write(f"- `mode` = `{kaldo_unfolding}`\n")
        fh.write(f"- `is_unfolding` = `{use_unfolding}`\n")
        fh.write(f"- `reason` = `{unfolding_reason}`\n")
        fh.write(
            "- Non-orthogonal primitive or conventional cells can require Wigner-Seitz "
            "unfolding when kALDo assembles harmonic dynamical matrices from finite-supercell IFCs.\n"
        )
    finite_length = None
    if finite_length_angstrom is not None:
        finite_length = tuple(None if float(v) == 0.0 else float(v) for v in finite_length_angstrom)
        print(f"[KALDO] finite length = {finite_length} Angstrom, method={finite_length_method}")
        with open(outdir / "finite_length_scattering.md", "w", encoding="utf-8") as fh:
            fh.write("# kALDo finite-length scattering\n\n")
            fh.write(f"- `length_angstrom` = `{finite_length}`\n")
            fh.write(f"- `length_nm` = `{tuple(None if v is None else v * 0.1 for v in finite_length)}`\n")
            fh.write(f"- `finite_length_method` = `{finite_length_method}`\n")
            fh.write("- This run reuses the same IFCs and changes only kALDo transport finite-size scattering.\n")

    max_frequency_thz = None
    max_frequency_report = None
    if max_frequency_from_banddos:
        max_frequency_thz, max_frequency_report = kaldo_max_frequency_from_banddos(outdir)
    if max_frequency_thz is not None and max_frequency_report is not None:
        print(
            "[KALDO] max_frequency from band/DOS support = "
            f"{max_frequency_thz:.6f} THz "
            f"({max_frequency_report['max_frequency_cm-1']:.3f} cm^-1)"
        )
        with open(outdir / "kaldo_transport_frequency_window.json", "w", encoding="utf-8") as fh:
            json.dump(
                {
                    **max_frequency_report,
                    "max_frequency_THz": max_frequency_thz,
                    "purpose": (
                        "Passed to kALDo Phonons(max_frequency=...) so NAC-created "
                        "transport modes outside the band/DOS physical support do not "
                        "enter physical_mode or conductivity."
                    ),
                },
                fh,
                indent=2,
            )
    elif max_frequency_from_banddos:
        print("[KALDO] no band/DOS frequency window found; kALDo max_frequency is unconstrained")
    else:
        print(
            "[KALDO] max_frequency remains unconstrained; band/DOS filtering, if "
            "available, applies only to plot copies"
        )

    for temp in temps:
        tdir = outdir / f"T{temp}"
        tdir.mkdir(exist_ok=True)
        print(f"[KALDO] transport T={temp} K, kpts={kpts}, method={method}")
        g_factor = None
        if isotopes:
            g_factor = np.array(
                [ISOTOPE_G_FACTORS.get(symbol, 0.0) for symbol in atoms.get_chemical_symbols()],
                dtype=float,
            )
        phonons_kwargs = {
            "forceconstants": kfc,
            "kpts": kpts,
            "is_classic": False,
            "temperature": float(temp),
            "folder": str(tdir),
            "storage": "numpy",
            "g_factor": g_factor,
            "is_unfolding": use_unfolding,
        }
        if max_frequency_thz is not None:
            phonons_kwargs["max_frequency"] = max_frequency_thz
        phonons = Phonons(**phonons_kwargs)
        if isotopes:
            phonons.include_isotopes = True
        cond_kwargs = {"method": method, "storage": "numpy"}
        if finite_length is not None:
            cond_kwargs["length"] = finite_length
            cond_kwargs["finite_length_method"] = finite_length_method
        cond = Conductivity(phonons=phonons, **cond_kwargs)
        raw = np.array(cond.conductivity)
        freq_thz = np.array(phonons.frequency)
        bw_thz = np.array(phonons.bandwidth)
        vel = np.array(phonons.velocity)
        mfp_ang = reshape_kaldo_vector(cond.mean_free_path, freq_thz.shape, "mean_free_path")
        if raw.ndim == 4:
            kt = raw.sum(axis=(0, 1))
            kappa_mode = raw
        elif raw.ndim == 3 and raw.shape[0] == freq_thz.size:
            kt = raw.sum(axis=0)
            kappa_mode = raw.reshape(freq_thz.shape + (3, 3))
        elif raw.ndim == 3:
            kt = raw.sum(axis=0)
            kappa_mode = None
        elif raw.ndim == 2:
            kt = raw
            kappa_mode = None
        else:
            raise ValueError(f"Unexpected kALDo conductivity shape: {raw.shape}")
        kappa_rows.append((temp,) + tuple(kt.ravel()))
        print(f"[KALDO] kxx={kt[0,0]:.3f} kyy={kt[1,1]:.3f} kzz={kt[2,2]:.3f} W/mK")

        v_abs = np.linalg.norm(vel * 100.0, axis=-1)
        material_symbols = set(cfg.get("symbols", []))
        if material_symbols == {"In", "As"}:
            bad_velocity = (freq_thz * THZ_TO_CM < 10.0) & (v_abs > 6000.0)
            if np.any(bad_velocity):
                print(
                    "[KALDO] masking "
                    f"{int(np.count_nonzero(bad_velocity))} InAs near-Gamma velocity spike(s) "
                    "in tau CSV output only"
                )
                v_abs = v_abs.copy()
                v_abs[bad_velocity] = np.nan
        mfp_abs_ang = np.linalg.norm(mfp_ang, axis=-1)
        tau_from_bw_ps = np.where((bw_thz > 0) & np.isfinite(bw_thz) & (freq_thz > 0.1), 1.0 / bw_thz, 0.0)
        # kALDo defines bandwidth as inverse lifetime. Use kALDo's native
        # mean_free_path separately rather than deriving MFP as v*tau.
        tau_ps = tau_from_bw_ps
        mfp_nm = mfp_abs_ang * 0.1
        if finite_length is not None:
            tau_eff_ps = np.zeros_like(tau_ps, dtype=float)
            np.divide(mfp_nm * 1000.0, v_abs, out=tau_eff_ps, where=(v_abs > 1e-12) & (mfp_nm > 0))
            tau_ps = tau_eff_ps
        if kappa_mode is None or kappa_mode.shape[:2] != freq_thz.shape:
            kappa_mode = mode_kappa_from_tau(freq_thz, vel, tau_ps, temp, nk, atoms.get_volume())

        tau_csv = outdir / f"tau_{temp}K.csv"
        with open(tau_csv, "w", encoding="utf-8") as fh:
            fh.write("# Created by MLIP+kALDo native finite-difference pipeline\n")
            fh.write(f"# Temperature  : {temp} K\n")
            fh.write("# ik,is,frequency[cm^-1],lifetime[ps],|velocity|[m/s],MFP[nm],multiplicity,kxx,kxy,kxz,kyx,kyy,kyz,kzx,kzy,kzz\n")
            for ik in range(freq_thz.shape[0]):
                for im in range(freq_thz.shape[1]):
                    km = kappa_mode[ik, im]
                    fh.write(
                        f"{ik + 1},{im + 1},{freq_thz[ik, im] * THZ_TO_CM:.7e},"
                        f"{tau_ps[ik, im]:.7e},{v_abs[ik, im]:.7e},{mfp_nm[ik, im]:.7e},1,"
                        f"{km[0,0]:.7e},{km[0,1]:.7e},{km[0,2]:.7e},"
                        f"{km[1,0]:.7e},{km[1,1]:.7e},{km[1,2]:.7e},"
                        f"{km[2,0]:.7e},{km[2,1]:.7e},{km[2,2]:.7e}\n"
                    )
        write_spectral_csv(outdir, temp, freq_thz * THZ_TO_CM, mfp_nm, kappa_mode)

    with open(outdir / "kappa.dat", "w", encoding="utf-8") as fh:
        fh.write("# Temperature [K], kxx kxy kxz kyx kyy kyz kzx kzy kzz [W/mK]\n")
        for row in kappa_rows:
            fh.write(f"{row[0]:10.2f}  " + "  ".join(f"{v:14.6f}" for v in row[1:]) + "\n")
    print(f"[KALDO] wrote {outdir / 'kappa.dat'}")
    if os.environ.get("KALDO_KEEP_CACHE", "").lower() not in {"1", "true", "yes"}:
        removed_dirs = 0
        for path in outdir.glob("T[0-9]*"):
            if path.is_dir():
                shutil.rmtree(path)
                removed_dirs += 1
        print(f"[KALDO] removed {removed_dirs} completed kALDo T* cache directories under {outdir}")


def load_numeric_csv(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            first = stripped[0]
            if not (first.isdigit() or first in "+-."):
                continue
            rows.append([float(x) for x in stripped.split(",")])
    if not rows:
        return np.empty((0, 0))
    return np.array(rows, dtype=float)


def sum_tau_kappa(path):
    if not path.exists():
        return None
    data = load_numeric_csv(path)
    if data.size == 0:
        return None
    if data.ndim == 1:
        data = data[None, :]
    multiplicity = data[:, 6] if data.shape[1] > 6 else 1.0
    kt = (data[:, 7:16] * multiplicity[:, None]).sum(axis=0).reshape(3, 3)
    return (kt[0, 0] + kt[1, 1] + kt[2, 2]) / 3.0


def cumulative_xy(data):
    if data.size == 0:
        return np.empty(0), np.empty(0), np.empty(0)
    if data.ndim == 1:
        data = data[None, :]
    data = data[np.isfinite(data[:, 0]) & np.isfinite(data[:, 1]) & np.isfinite(data[:, 2])]
    if data.size == 0:
        return np.empty(0), np.empty(0), np.empty(0)
    order = np.argsort(data[:, 0])
    return data[order, 0], data[order, 1], data[order, 2]


def binned_spectral_xy(x, spec, bins=80, logx=False):
    x = np.asarray(x, dtype=float)
    spec = np.asarray(spec, dtype=float)
    mask = np.isfinite(x) & np.isfinite(spec)
    if logx:
        mask &= x > 0
    if mask.sum() < 2:
        return np.empty(0), np.empty(0)
    x = x[mask]
    spec = spec[mask]
    if logx:
        lo = max(np.nanmin(x), 1e-6)
        hi = np.nanmax(x)
        if hi <= lo:
            return np.empty(0), np.empty(0)
        edges = np.logspace(np.log10(lo), np.log10(hi), bins + 1)
        centers = np.sqrt(edges[:-1] * edges[1:])
    else:
        lo = np.nanmin(x)
        hi = np.nanmax(x)
        if hi <= lo:
            return np.empty(0), np.empty(0)
        edges = np.linspace(lo, hi, bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
    values, _ = np.histogram(x, bins=edges, weights=spec)
    return centers, values


def plot_cumulative_spectrum(
    model_csv, dft_csv, out_png, xlabel, title, model_label_text, model_color_value,
    logx=False, spectral_bins=80, xmin=None,
):
    if not model_csv.exists():
        return
    mx, mkc, mks = cumulative_xy(load_numeric_csv(model_csv))
    if mx.size == 0:
        return

    fig, ax = plt.subplots(figsize=(4.25, 3.15))
    ax2 = ax.twinx()
    if not logx:
        mx = np.r_[0.0, mx]
        mkc = np.r_[0.0, mkc]
        mks = np.r_[0.0, mks]
    ax.plot(mx, mkc, color=model_color_value, lw=1.65, label=model_label_text)
    bx, by = binned_spectral_xy(mx, mks, bins=spectral_bins, logx=logx)
    if bx.size:
        ax2.fill_between(bx, by, color=model_color_value, alpha=0.20, lw=0)

    if dft_csv.exists():
        dx, dkc, dks = cumulative_xy(load_numeric_csv(dft_csv))
        if dx.size:
            if not logx:
                dx = np.r_[0.0, dx]
                dkc = np.r_[0.0, dkc]
                dks = np.r_[0.0, dks]
            ax.plot(dx, dkc, color=DFT_COLOR, lw=1.65, label="DFT")
            bx, by = binned_spectral_xy(dx, dks, bins=spectral_bins, logx=logx)
            if bx.size:
                ax2.fill_between(bx, by, color=DFT_COLOR, alpha=0.18, lw=0)

    if logx:
        ax.set_xscale("log")
        ax2.set_xscale("log")
    if xmin is not None:
        ax.set_xlim(left=xmin)
        ax2.set_xlim(left=xmin)
    elif not logx:
        ax.set_xlim(left=0.0)
        ax2.set_xlim(left=0.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Cumulative $\kappa_\mathrm{avg}$ (W m$^{-1}$ K$^{-1}$)")
    ax2.set_ylabel(r"Spectral $\kappa_\mathrm{avg}$ (W m$^{-1}$ K$^{-1}$ bin$^{-1}$)")
    ax.set_title(title)
    lines, labels = ax.get_legend_handles_labels()
    ax.legend(lines, labels, frameon=False)
    set_axes_style(ax)
    set_axes_style(ax2)
    # twin axes: left axis owns the left side, right axis owns the right side —
    # turn off the mirrored tick marks so the right side does not show two
    # overlapping scales (cumulative + spectral).
    ax.tick_params(which="both", right=False)
    ax2.tick_params(which="both", left=False)
    ax2.grid(False)
    save_pubfig(fig, out_png)
    plt.close(fig)


def set_lifetime_ylim(ax, *arrays):
    values = []
    for arr in arrays:
        if arr is None or np.size(arr) == 0:
            continue
        tau = np.asarray(arr)[:, 3]
        tau = tau[np.isfinite(tau) & (tau > 0)]
        if tau.size:
            values.append(tau)
    if not values:
        return
    tau = np.concatenate(values)
    lo = np.percentile(tau, 1.0)
    if np.isfinite(lo) and lo > 0:
        ax.set_ylim(bottom=max(lo * 0.75, 1e-3))


def set_velocity_ylim(ax, *arrays, outlier_cap=12000.0):
    vals = []
    for arr in arrays:
        if arr is None:
            continue
        a = np.asarray(arr)
        if a.ndim == 1:
            a = a[None, :]
        if a.shape[1] <= 4:
            continue
        v = a[:, 4]
        vals.append(v[np.isfinite(v) & (v >= 0.0) & (v <= outlier_cap)])
    if not vals:
        ax.set_ylim(bottom=0)
        return
    vel = np.concatenate(vals)
    if vel.size == 0:
        ax.set_ylim(bottom=0)
        return
    ymax = float(np.max(vel))
    if ymax <= 0:
        ax.set_ylim(bottom=0)
        return
    ax.set_ylim(0, min(outlier_cap, ymax * 1.05))


def plot_transport(outdir, cfg, title, temps, model_kind):
    label = model_label(model_kind)
    color = model_color(model_kind)
    kappa_file = outdir / "kappa.dat"
    if not kappa_file.exists():
        return
    data = np.loadtxt(kappa_file, comments="#")
    if data.ndim == 1:
        data = data[None, :]
    t = data[:, 0]
    kave = (data[:, 1] + data[:, 5] + data[:, 9]) / 3.0

    fig, ax = plt.subplots(figsize=(4.0, 3.1))
    ax.plot(t, kave, "o-", color=color, ms=4.5, lw=1.5, label=label)
    dft_k300 = sum_tau_kappa(cfg["dft_transport"] / "tau_300K.csv")
    if dft_k300 is not None:
        ax.plot([300], [dft_k300], "s", color=DFT_COLOR, ms=4.8, label="DFT")
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(r"$\kappa_\mathrm{avg}$ (W m$^{-1}$ K$^{-1}$)")
    ax.set_title(title)
    ax.legend(frameon=False)
    set_axes_style(ax)
    save_pubfig(fig, outdir / "fig_kappa_compare.png")
    plt.close(fig)

    temp = 300 if 300 in temps else temps[0]
    tau_file = outdir / f"tau_{temp}K.csv"
    if tau_file.exists():
        arr = load_numeric_csv(tau_file)
        if arr.ndim == 1:
            arr = arr[None, :]
        freq_window = load_transport_frequency_window(outdir)
        arr_plot, model_filtered = filter_transport_modes_for_plot(arr, freq_window)
        write_physical_window_tau_csv(tau_file, arr_plot, temp, freq_window)
        fig, ax = plt.subplots(figsize=(4.0, 3.1))
        ax.scatter(arr_plot[:, 2], arr_plot[:, 3], s=9, c=color, alpha=0.58, label=label, linewidths=0)
        dft_tau = cfg["dft_transport"] / f"tau_{temp}K.csv"
        d = None
        d_plot = None
        dft_filtered = 0
        if dft_tau.exists():
            d = load_numeric_csv(dft_tau)
            d_plot, dft_filtered = filter_transport_modes_for_plot(d, freq_window)
            ax.scatter(d_plot[:, 2], d_plot[:, 3], s=8, c=DFT_COLOR, alpha=0.38, label="DFT", linewidths=0)
        ax.set_yscale("log")
        set_lifetime_ylim(ax, arr_plot, d_plot)
        if freq_window is not None:
            ax.set_xlim(freq_window[0], freq_window[1])
        ax.set_xlabel("Frequency (cm$^{-1}$)")
        ax.set_ylabel("Lifetime (ps)")
        ax.set_title(f"{title} lifetime {temp} K")
        ax.legend(frameon=False)
        set_axes_style(ax)
        save_pubfig(fig, outdir / f"fig_lifetime_compare_{temp}K.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(4.0, 3.1))
        ax.scatter(arr_plot[:, 2], arr_plot[:, 4], s=9, c=color, alpha=0.58, label=label, linewidths=0)
        if d_plot is not None:
            ax.scatter(d_plot[:, 2], d_plot[:, 4], s=8, c=DFT_COLOR, alpha=0.38, label="DFT", linewidths=0)
        ax.set_xlabel("Frequency (cm$^{-1}$)")
        ax.set_ylabel("|velocity| (m/s)")
        set_velocity_ylim(ax, arr_plot, d_plot)
        if freq_window is not None:
            ax.set_xlim(freq_window[0], freq_window[1])
        ax.set_title(f"{title} group velocity {temp} K")
        ax.legend(frameon=False)
        set_axes_style(ax)
        save_pubfig(fig, outdir / f"fig_velocity_compare_{temp}K.png")
        plt.close(fig)

        if freq_window is not None:
            with open(outdir / f"transport_plot_frequency_filter_{temp}K.json", "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "frequency_window_cm-1": list(freq_window),
                        "source": "banddos_frequency_window.json",
                        "model_points_filtered": model_filtered,
                        "dft_points_filtered": dft_filtered,
                        "raw_transport_and_kappa_unchanged": True,
                        "model": summarize_transport_filter(arr, freq_window),
                        "dft": summarize_transport_filter(d, freq_window),
                    },
                    handle,
                    indent=2,
                )

    plot_cumulative_spectrum(
        outdir / f"kspec_frequency_{temp}K.csv",
        cfg["dft_transport"] / f"kspec_frequency_{temp}K.csv",
        outdir / f"fig_spectral_kappa_compare_{temp}K.png",
        "Frequency (cm$^{-1}$)",
        f"{title} spectral kappa {temp} K",
        label,
        color,
        logx=False,
        spectral_bins=90,
    )
    plot_cumulative_spectrum(
        outdir / f"kspec_mfp_{temp}K.csv",
        cfg["dft_transport"] / f"kspec_mfp_{temp}K.csv",
        outdir / f"fig_cumulative_mfp_compare_{temp}K.png",
        "Mean free path (nm)",
        f"{title} cumulative kappa by MFP {temp} K",
        label,
        color,
        logx=True,
        spectral_bins=80,
        xmin=0.1,
    )

    print(f"[PLOT] transport figures written in {outdir}")


def archive_memory():
    mem = PROJECT_ROOT / "MEMORY.md"
    if mem.exists():
        stamp = time.strftime("%Y%m%dT%H%M%S")
        shutil.copy2(mem, PROJECT_ROOT / "archive" / f"MEMORY.md.bak.{stamp}")


def main():
    args = parse_args()
    cfg = MATERIALS[args.material]
    if args.borninfo is not None or args.projected_dos_plot:
        cfg = dict(cfg)
    if args.borninfo is not None:
        cfg["borninfo"] = args.borninfo
    if args.projected_dos_plot:
        cfg["projected_dos_plot"] = True
    model_path = Path(args.model) if args.model else (MACE_MODEL if args.model_kind == "mace" else DPA_MODEL)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    device = choose_device(args.device)
    if args.model_kind == "dpa" and device == "cpu":
        os.environ.setdefault("DP_INTRA_OP_PARALLELISM_THREADS", os.environ.get("OMP_NUM_THREADS", "32"))

    suffix = args.run_label or ("quick" if args.quick else "full")
    outdir = PROJECT_ROOT / "results" / args.material / args.model_kind / suffix
    outdir.mkdir(parents=True, exist_ok=True)
    archive_memory()

    fc2 = args.fc2_supercell or cfg["fc2_supercell"]
    fc3 = args.fc3_supercell or cfg["fc3_supercell"]
    kpts = args.kpts or cfg["kaldo_kmesh"]
    dos_mesh = args.dos_mesh or cfg["dos_mesh"]
    if args.quick:
        fc2 = args.fc2_supercell or [2, 2, 2]
        fc3 = args.fc3_supercell or [1, 1, 1]
        kpts = args.kpts or [2, 2, 2]
        dos_mesh = args.dos_mesh or [3, 3, 3]
        args.temps = args.temps or [300]

    print(f"[INFO] material={args.material} model={args.model_kind} path={model_path}")
    print(f"[INFO] output={outdir}")
    print(f"[INFO] device={device} fc2={fc2} fc3={fc3} kpts={kpts} temps={args.temps}")

    title = f"{args.material} {model_label(args.model_kind)}"

    if args.plot_only:
        if (outdir / "band.yaml").exists() and (outdir / "projected_dos.dat").exists():
            plot_banddos(outdir, cfg, title, args.model_kind)
        plot_transport(outdir, cfg, title, args.temps, args.model_kind)
        print(f"[DONE] Replotted figures in {outdir}")
        return

    from ase.io import read as ase_read

    cached_banddos_only = args.banddos_from_kaldo and args.skip_transport
    calc = None if cached_banddos_only else build_calculator(args.model_kind, model_path, device, args.head)
    relaxed_poscar = outdir / "POSCAR.mlip_relaxed"
    if args.model_kind == "dpa" and not args.allow_dpa_relax:
        atoms = ase_read(str(cfg["poscar"]), format="vasp")
        if relaxed_poscar.exists():
            print(f"[INFO] ignoring cached DP relaxed structure: {relaxed_poscar}")
        print("[INFO] DP model was trained without virial loss; using DFT-relaxed cell.")
    elif relaxed_poscar.exists():
        atoms = ase_read(str(relaxed_poscar), format="vasp")
        print(f"[INFO] reusing relaxed structure: {relaxed_poscar}")
    else:
        atoms = ase_read(str(cfg["poscar"]), format="vasp")
        if cached_banddos_only:
            print("[INFO] cached-banddos-only run has no relaxed POSCAR; using DFT-relaxed POSCAR")
        elif not args.skip_relax:
            atoms = relax_structure(atoms, calc, relaxed_poscar, outdir / "relax.log")
        else:
            print("[INFO] using DFT-relaxed POSCAR without MLIP relaxation")

    if args.align_basis_to:
        atoms = align_basis_to_reference(atoms, args.material, args.align_basis_to)
        from ase.io import write as ase_write
        ase_write(str(outdir / f"POSCAR.basis_aligned_to_{args.align_basis_to}"), atoms, format="vasp")

    if not args.skip_banddos:
        if args.banddos_from_kaldo:
            run_phonopy_banddos_from_kaldo(atoms.copy(), cfg, outdir, fc2, dos_mesh, quick=args.quick)
        else:
            run_phonopy_banddos(atoms.copy(), calc, cfg, outdir, fc2, dos_mesh, quick=args.quick)
        plot_banddos(outdir, cfg, title, args.model_kind)

    if not args.skip_transport:
        run_kaldo_transport.fc_only = args.fc_only
        run_kaldo_transport.allow_symmetry_broken_kmesh = args.allow_symmetry_broken_kmesh
        run_kaldo_transport(
            atoms.copy(), calc, cfg, outdir, fc2, fc3, kpts, args.temps,
            args.method, args.delta_shift_fc2, args.delta_shift_fc3,
            args.skip_fd, not args.no_isotopes,
            finite_length_angstrom=args.finite_length_angstrom,
            finite_length_method=args.finite_length_method,
            kaldo_unfolding=args.kaldo_unfolding,
            max_frequency_from_banddos=args.kaldo_max_frequency_from_banddos,
        )
        plot_transport(outdir, cfg, title, args.temps, args.model_kind)

    print(f"[DONE] Results in {outdir}")


if __name__ == "__main__":
    main()
