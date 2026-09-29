from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_transport_frequency_window(outdir: Path) -> tuple[float, float] | None:
    """Read the physical plotting window selected by the band/DOS stage."""
    report_path = Path(outdir) / "banddos_frequency_window.json"
    if not report_path.exists():
        return None
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    section = report.get("comparison") or report.get("model_only") or {}
    lower = section.get("plot_min_cm-1", section.get("physical_support_min_cm-1"))
    upper = section.get("physical_support_max_cm-1", section.get("band_max_cm-1"))
    try:
        lower = float(lower)
        upper = float(upper)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        return None
    return lower, upper


def filter_transport_modes_for_plot(
    values: np.ndarray | None,
    frequency_window: tuple[float, float] | None,
) -> tuple[np.ndarray | None, int]:
    """Filter a plot copy by frequency without modifying the source table."""
    if values is None or frequency_window is None:
        return values, 0
    array = np.asarray(values)
    if array.ndim == 1:
        array = array[None, :]
    if array.shape[1] <= 2:
        return array, 0
    lower, upper = frequency_window
    mask = (
        np.isfinite(array[:, 2])
        & (array[:, 2] >= lower)
        & (array[:, 2] <= upper)
    )
    return array[mask], int(array.shape[0] - np.count_nonzero(mask))


def summarize_transport_filter(
    values: np.ndarray | None,
    frequency_window: tuple[float, float] | None,
) -> dict[str, object]:
    """Quantify removed points and their conductivity without changing data."""
    if values is None:
        return {"available": False}
    array = np.asarray(values)
    if array.ndim == 1:
        array = array[None, :]
    summary: dict[str, object] = {
        "available": True,
        "raw_points": int(array.shape[0]),
        "raw_frequency_min_cm-1": (
            float(np.nanmin(array[:, 2])) if array.shape[1] > 2 else None
        ),
        "raw_frequency_max_cm-1": (
            float(np.nanmax(array[:, 2])) if array.shape[1] > 2 else None
        ),
    }
    filtered, removed = filter_transport_modes_for_plot(array, frequency_window)
    summary["retained_points"] = int(len(filtered))
    summary["filtered_points"] = removed
    if frequency_window is None or array.shape[1] < 16:
        return summary
    lower, upper = frequency_window
    outside = (
        ~np.isfinite(array[:, 2])
        | (array[:, 2] < lower)
        | (array[:, 2] > upper)
    )
    mode_tensors = array[:, 7:16] * array[:, 6, None]
    total_tensor = mode_tensors.sum(axis=0).reshape(3, 3)
    filtered_tensor = mode_tensors[outside].sum(axis=0).reshape(3, 3)
    total_average = float(np.trace(total_tensor) / 3.0)
    filtered_average = float(np.trace(filtered_tensor) / 3.0)
    summary.update(
        {
            "raw_kappa_average_W_mK": total_average,
            "filtered_kappa_average_W_mK": filtered_average,
            "filtered_kappa_fraction": (
                filtered_average / total_average if total_average != 0 else None
            ),
            "raw_kappa_tensor_W_mK": total_tensor.tolist(),
            "filtered_kappa_tensor_W_mK": filtered_tensor.tolist(),
        }
    )
    return summary


def write_physical_window_tau_csv(
    path: Path,
    values: np.ndarray | None,
    temperature: int,
    frequency_window: tuple[float, float] | None,
) -> Path | None:
    """Write an explicitly labeled filtered copy while preserving raw tau CSV."""
    if values is None or frequency_window is None:
        return None
    output = path.with_name(f"{path.stem}.physical_window{path.suffix}")
    header = (
        "# Filtered plot copy; the raw tau CSV and kappa are unchanged\n"
        f"# Temperature  : {temperature} K\n"
        f"# frequency_window_cm^-1 : {frequency_window[0]:.10e}, "
        f"{frequency_window[1]:.10e}\n"
        "# ik,is,frequency[cm^-1],lifetime[ps],|velocity|[m/s],MFP[nm],"
        "multiplicity,kxx,kxy,kxz,kyx,kyy,kyz,kzx,kzy,kzz\n"
    )
    np.savetxt(
        output,
        values,
        delimiter=",",
        header=header.rstrip("\n"),
        comments="",
        fmt="%.10e",
    )
    return output
