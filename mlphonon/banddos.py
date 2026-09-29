from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def integrated_dos_bounds(
    frequencies: np.ndarray,
    dos: np.ndarray,
    *,
    tail_mass_fraction: float = 1e-6,
) -> dict[str, float]:
    """Return DOS support bounds from integrated spectral weight."""
    if not 0 <= tail_mass_fraction < 0.5:
        raise ValueError("tail_mass_fraction must satisfy 0 <= value < 0.5")
    x = np.asarray(frequencies, dtype=float).reshape(-1)
    y = np.asarray(dos, dtype=float).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = np.maximum(y[valid], 0.0)
    if x.size < 2:
        raise ValueError("DOS curve must contain at least two finite points")
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    dx = np.diff(x)
    interval_weight = 0.5 * (y[:-1] + y[1:]) * np.maximum(dx, 0.0)
    cumulative = np.r_[0.0, np.cumsum(interval_weight)]
    total = float(cumulative[-1])
    if not np.isfinite(total) or total <= 0:
        raise ValueError("DOS integral is not positive and finite")
    lower_target = tail_mass_fraction * total
    upper_target = (1.0 - tail_mass_fraction) * total
    lower_index = min(int(np.searchsorted(cumulative, lower_target)), len(x) - 1)
    upper_index = min(int(np.searchsorted(cumulative, upper_target)), len(x) - 1)
    return {
        "raw_min": float(x[0]),
        "raw_max": float(x[-1]),
        "integral": total,
        "lower_mass_bound": float(x[lower_index]),
        "upper_mass_bound": float(x[upper_index]),
    }


def _outside_mass_fraction(
    frequencies: np.ndarray, dos: np.ndarray, lower: float, upper: float
) -> float:
    x = np.asarray(frequencies, dtype=float).reshape(-1)
    y = np.maximum(np.asarray(dos, dtype=float).reshape(-1), 0.0)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    dx = np.maximum(np.diff(x), 0.0)
    weights = 0.5 * (y[:-1] + y[1:]) * dx
    centers = 0.5 * (x[:-1] + x[1:])
    total = weights.sum()
    if total <= 0:
        return 0.0
    outside = weights[(centers < lower) | (centers > upper)].sum()
    return float(outside / total)


def select_frequency_window(
    band_frequency_sets: Sequence[np.ndarray],
    dos_curves: Sequence[tuple[str, np.ndarray, np.ndarray]],
    *,
    tail_mass_fraction: float = 1e-6,
    margin_fraction: float = 0.02,
    minimum_margin: float = 5.0,
    lower_floor: float = -15.0,
) -> tuple[tuple[float, float], dict[str, Any]]:
    """Select plot limits without treating Gaussian DOS tails as phonon modes."""
    finite_bands = []
    for values in band_frequency_sets:
        array = np.asarray(values, dtype=float)
        finite = array[np.isfinite(array)]
        if finite.size:
            finite_bands.append(finite)
    if not finite_bands:
        raise ValueError("No finite band frequencies were supplied")
    all_bands = np.concatenate(finite_bands)
    band_min = float(all_bands.min())
    band_max = float(all_bands.max())

    dos_reports: dict[str, dict[str, float]] = {}
    physical_min = band_min
    physical_max = band_max
    for label, frequencies, dos in dos_curves:
        report = integrated_dos_bounds(
            frequencies, dos, tail_mass_fraction=tail_mass_fraction
        )
        dos_reports[label] = report
        physical_min = min(physical_min, report["lower_mass_bound"])
        physical_max = max(physical_max, report["upper_mass_bound"])

    span = max(physical_max - physical_min, 1.0)
    margin = max(float(minimum_margin), float(margin_fraction) * span)
    lower = min(float(lower_floor), physical_min - margin if physical_min < 0 else lower_floor)
    upper = physical_max + margin
    for label, frequencies, dos in dos_curves:
        dos_reports[label]["mass_fraction_outside_plot"] = _outside_mass_fraction(
            frequencies, dos, lower, upper
        )

    report: dict[str, Any] = {
        "settings": {
            "tail_mass_fraction": float(tail_mass_fraction),
            "margin_fraction": float(margin_fraction),
            "minimum_margin_cm-1": float(minimum_margin),
            "lower_floor_cm-1": float(lower_floor),
        },
        "band_min_cm-1": band_min,
        "band_max_cm-1": band_max,
        "physical_support_min_cm-1": physical_min,
        "physical_support_max_cm-1": physical_max,
        "plot_min_cm-1": lower,
        "plot_max_cm-1": upper,
        "dos": dos_reports,
    }
    return (lower, upper), report


def clip_curve_to_window(
    frequencies: np.ndarray,
    values: np.ndarray,
    window: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Remove curve points outside a selected plotting window."""
    x = np.asarray(frequencies)
    y = np.asarray(values)
    mask = np.isfinite(x) & np.isfinite(y) & (x >= window[0]) & (x <= window[1])
    return x[mask], y[mask]
