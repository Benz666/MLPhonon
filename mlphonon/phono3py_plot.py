from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-mlip-ld")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODEL_COLOR = "#4260CF"
DFT_COLOR = "#AD1A60"


def _load_csv(path: Path) -> np.ndarray:
    if not path.exists():
        return np.empty((0, 0))
    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            rows.append([float(value) for value in stripped.split(",")])
        except ValueError:
            continue
    if not rows:
        return np.empty((0, 0))
    data = np.asarray(rows, dtype=float)
    return data[None, :] if data.ndim == 1 else data


def _set_style(ax: plt.Axes) -> None:
    ax.minorticks_on()
    ax.tick_params(which="major", direction="in", top=True, right=True, length=4)
    ax.tick_params(which="minor", direction="in", top=True, right=True, length=2)
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _mode_kappa_average(data: np.ndarray) -> np.ndarray:
    if data.shape[1] >= 16:
        diagonal = data[:, [7, 11, 15]]
    elif data.shape[1] >= 10:
        diagonal = data[:, 7:10]
    else:
        raise ValueError(f"Mode table has too few columns for kappa: {data.shape}")
    return diagonal.mean(axis=1) * data[:, 6]


def _kappa_tensor(data: np.ndarray) -> np.ndarray | None:
    if not data.size:
        return None
    if data.shape[1] >= 16:
        return (data[:, 7:16] * data[:, 6, None]).sum(axis=0).reshape(3, 3)
    if data.shape[1] >= 13:
        voigt = (data[:, 7:13] * data[:, 6, None]).sum(axis=0)
        return np.array(
            [
                [voigt[0], voigt[5], voigt[4]],
                [voigt[5], voigt[1], voigt[3]],
                [voigt[4], voigt[3], voigt[2]],
            ]
        )
    return None


def _mfp_nm(data: np.ndarray) -> np.ndarray:
    """Reconstruct MFP from the declared ps and m/s columns in nanometres."""
    return data[:, 3] * data[:, 4] * 1.0e-3


def _mfp_diagnostic(data: np.ndarray) -> dict[str, float | int | None]:
    reconstructed = _mfp_nm(data)
    valid = (
        np.isfinite(data[:, 5])
        & np.isfinite(reconstructed)
        & (data[:, 5] > 0)
        & (reconstructed > 0)
    )
    if not np.any(valid):
        return {
            "compared_modes": 0,
            "reported_to_reconstructed_median": None,
        }
    ratio = data[valid, 5] / reconstructed[valid]
    return {
        "compared_modes": int(np.count_nonzero(valid)),
        "reported_to_reconstructed_median": float(np.median(ratio)),
    }


def _frequency_limit(model: np.ndarray, dft: np.ndarray) -> float:
    maxima = [float(np.nanmax(model[:, 2]))]
    if dft.size:
        maxima.append(float(np.nanmax(dft[:, 2])))
    return max(maxima) * 1.02


def _write_spectrum(path: Path, x: np.ndarray, contribution: np.ndarray) -> None:
    order = np.argsort(x)
    cumulative = np.cumsum(contribution[order])
    with path.open("w", encoding="utf-8") as handle:
        handle.write("x,kcumu,kspec\n")
        for x_value, kcumu, kspec in zip(
            x[order], cumulative, contribution[order], strict=True
        ):
            handle.write(f"{x_value:.8e},{kcumu:.8e},{kspec:.8e}\n")


def _binned(
    x: np.ndarray, values: np.ndarray, *, bins: int = 80, log_x: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(x) & np.isfinite(values)
    if log_x:
        valid &= x > 0
    x = x[valid]
    values = values[valid]
    if x.size < 2:
        return np.empty(0), np.empty(0)
    if log_x:
        edges = np.logspace(np.log10(x.min()), np.log10(x.max()), bins + 1)
        centers = np.sqrt(edges[:-1] * edges[1:])
    else:
        edges = np.linspace(x.min(), x.max(), bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
    histogram, _ = np.histogram(x, bins=edges, weights=values)
    return centers, histogram


def _plot_cumulative(
    output: Path,
    model_x: np.ndarray,
    model_k: np.ndarray,
    dft_x: np.ndarray | None,
    dft_k: np.ndarray | None,
    *,
    xlabel: str,
    title: str,
    log_x: bool,
    xmin: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(4.25, 3.2))
    ax2 = ax.twinx()
    for x, values, color, label in (
        (model_x, model_k, MODEL_COLOR, "MACE + phono3py"),
        (dft_x, dft_k, DFT_COLOR, "DFT"),
    ):
        if x is None or values is None or not len(x):
            continue
        valid = np.isfinite(x) & np.isfinite(values)
        if log_x:
            valid &= x > 0
        x_valid = x[valid]
        values_valid = values[valid]
        order = np.argsort(x_valid)
        x_sorted = x_valid[order]
        cumulative = np.cumsum(values_valid[order])
        visible = np.ones(len(x_sorted), dtype=bool)
        if xmin is not None:
            visible &= x_sorted >= xmin
        ax.plot(
            x_sorted[visible], cumulative[visible], color=color, lw=1.6,
            label=label,
        )
        bin_mask = (
            x_valid >= xmin
            if xmin is not None
            else np.ones(len(x_valid), dtype=bool)
        )
        centers, binned = _binned(
            x_valid[bin_mask], values_valid[bin_mask], log_x=log_x
        )
        if centers.size:
            ax2.fill_between(centers, binned, color=color, alpha=0.18, lw=0)
    if log_x:
        ax.set_xscale("log")
        ax2.set_xscale("log")
    else:
        ax.set_xlim(left=0)
        ax2.set_xlim(left=0)
    if xmin is not None:
        ax.set_xlim(left=xmin)
        ax2.set_xlim(left=xmin)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Cumulative $\kappa_\mathrm{avg}$ (W m$^{-1}$ K$^{-1}$)")
    ax2.set_ylabel(r"Spectral $\kappa_\mathrm{avg}$ (W m$^{-1}$ K$^{-1}$ bin$^{-1}$)")
    ax.set_title(title)
    ax.legend(frameon=False)
    _set_style(ax)
    _set_style(ax2)
    ax.tick_params(which="both", right=False)
    ax2.tick_params(which="both", left=False)
    ax2.grid(False)
    _save(fig, output)


def _copy_harmonic_figures(source: Path, output: Path) -> list[str]:
    copied: list[str] = []
    for name in (
        "fig_banddos_mlip.png",
        "fig_banddos_compare_dft.png",
        "banddos_frequency_window.json",
    ):
        path = source / name
        if path.exists():
            shutil.copy2(path, output / name)
            copied.append(name)
    return copied


def postprocess_phono3py_result(
    output: Path,
    *,
    source: Path,
    material: str,
    dft_transport: Path | None = None,
) -> dict[str, Any]:
    """Create the standard endpoint plot set from an existing phono3py result."""
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    temperatures = [float(value) for value in result["temperatures_K"]]
    report: dict[str, Any] = {
        "backend": "phono3py",
        "source_result": str(source.resolve()),
        "dft_transport": str(dft_transport.resolve()) if dft_transport else None,
        "harmonic_files_copied": _copy_harmonic_figures(source, output),
        "temperatures": {},
    }
    for temperature in temperatures:
        tag = f"{temperature:g}K"
        model = _load_csv(output / f"tau_{tag}.csv")
        if not model.size:
            raise FileNotFoundError(output / f"tau_{tag}.csv")
        dft_path = dft_transport / f"tau_{tag}.csv" if dft_transport else None
        dft = _load_csv(dft_path) if dft_path else np.empty((0, 0))
        frequency_max = _frequency_limit(model, dft)

        fig, ax = plt.subplots(figsize=(4.1, 3.2))
        mask = np.isfinite(model[:, 3]) & (model[:, 3] > 0)
        ax.scatter(
            model[mask, 2], model[mask, 3], s=8, color=MODEL_COLOR, alpha=0.5,
            linewidths=0, label="MACE + phono3py",
        )
        if dft.size:
            dmask = np.isfinite(dft[:, 3]) & (dft[:, 3] > 0)
            ax.scatter(
                dft[dmask, 2], dft[dmask, 3], s=7, color=DFT_COLOR, alpha=0.3,
                linewidths=0, label="DFT",
            )
        positive = model[mask, 3]
        if dft.size:
            positive = np.r_[positive, dft[dmask, 3]]
        ax.set_yscale("log")
        if positive.size:
            ax.set_ylim(
                max(float(np.percentile(positive, 0.5)) * 0.7, 1e-3),
                float(np.percentile(positive, 99.7)) * 1.4,
            )
        ax.set_xlim(0, frequency_max)
        ax.set_xlabel(r"Frequency (cm$^{-1}$)")
        ax.set_ylabel("Lifetime (ps)")
        ax.set_title(f"{material} lifetime {temperature:g} K")
        ax.legend(frameon=False)
        _set_style(ax)
        _save(fig, output / f"fig_lifetime_compare_{tag}.png")

        fig, ax = plt.subplots(figsize=(4.1, 3.2))
        ax.scatter(
            model[:, 2], model[:, 4], s=8, color=MODEL_COLOR, alpha=0.5,
            linewidths=0, label="MACE + phono3py",
        )
        velocity_values = [model[:, 4]]
        if dft.size:
            ax.scatter(
                dft[:, 2], dft[:, 4], s=7, color=DFT_COLOR, alpha=0.3,
                linewidths=0, label="DFT",
            )
            velocity_values.append(dft[:, 4])
        velocity = np.concatenate(velocity_values)
        velocity = velocity[np.isfinite(velocity) & (velocity >= 0) & (velocity <= 12000)]
        ax.set_ylim(0, min(12000, float(np.percentile(velocity, 99.9)) * 1.08))
        ax.set_xlim(0, frequency_max)
        ax.set_xlabel(r"Frequency (cm$^{-1}$)")
        ax.set_ylabel(r"$|v_g|$ (m s$^{-1}$)")
        ax.set_title(f"{material} group velocity")
        ax.legend(frameon=False)
        _set_style(ax)
        _save(fig, output / f"fig_velocity_compare_{tag}.png")

        model_k = _mode_kappa_average(model)
        dft_k = _mode_kappa_average(dft) if dft.size else None
        model_mfp = _mfp_nm(model)
        dft_mfp = _mfp_nm(dft) if dft.size else None
        _write_spectrum(
            output / f"kspec_frequency_{tag}.csv", model[:, 2], model_k
        )
        valid_mfp = np.isfinite(model_mfp) & (model_mfp > 0)
        _write_spectrum(
            output / f"kspec_mfp_{tag}.csv", model_mfp[valid_mfp], model_k[valid_mfp]
        )
        _plot_cumulative(
            output / f"fig_spectral_kappa_compare_{tag}.png",
            model[:, 2], model_k,
            dft[:, 2] if dft.size else None, dft_k,
            xlabel=r"Frequency (cm$^{-1}$)",
            title=f"{material} spectral conductivity {temperature:g} K",
            log_x=False,
        )
        dft_mfp_mask = (
            np.isfinite(dft_mfp) & (dft_mfp > 0) if dft.size else None
        )
        _plot_cumulative(
            output / f"fig_cumulative_mfp_compare_{tag}.png",
            model_mfp[valid_mfp], model_k[valid_mfp],
            dft_mfp[dft_mfp_mask] if dft.size else None,
            dft_k[dft_mfp_mask] if dft.size else None,
            xlabel="Mean free path (nm)",
            title=f"{material} MFP conductivity {temperature:g} K",
            log_x=True,
            xmin=0.1,
        )

        model_tensor = _kappa_tensor(model)
        dft_tensor = _kappa_tensor(dft)
        labels = ["In-plane", "Along c", "Average"]
        model_values = [
            float((model_tensor[0, 0] + model_tensor[1, 1]) / 2),
            float(model_tensor[2, 2]),
            float(np.trace(model_tensor) / 3),
        ]
        fig, ax = plt.subplots(figsize=(4.8, 3.4))
        positions = np.arange(3)
        width = 0.36 if dft_tensor is not None else 0.55
        ax.bar(
            positions - (width / 2 if dft_tensor is not None else 0),
            model_values, width, color=MODEL_COLOR, label="MACE + phono3py",
        )
        dft_values = None
        if dft_tensor is not None:
            dft_values = [
                float((dft_tensor[0, 0] + dft_tensor[1, 1]) / 2),
                float(dft_tensor[2, 2]),
                float(np.trace(dft_tensor) / 3),
            ]
            ax.bar(
                positions + width / 2, dft_values, width,
                color=DFT_COLOR, label="DFT",
            )
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(r"$\kappa$ (W m$^{-1}$ K$^{-1}$)")
        ax.set_title(f"{material} thermal conductivity {temperature:g} K")
        all_values = model_values + (dft_values or [])
        ax.set_ylim(0, max(all_values) * 1.28)
        ax.legend(frameon=False, loc="upper center", ncol=2, fontsize=9)
        _set_style(ax)
        _save(fig, output / "fig_kappa_compare.png")
        report["temperatures"][tag] = {
            "model_kappa_W_mK": model_values,
            "dft_kappa_W_mK": dft_values,
            "model_modes": int(len(model)),
            "dft_modes": int(len(dft)),
            "mfp_unit_diagnostic": {
                "formula": "lifetime_ps * velocity_m_per_s * 1e-3",
                "model": _mfp_diagnostic(model),
                "dft": _mfp_diagnostic(dft) if dft.size else None,
            },
        }
    (output / "postprocess.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report
