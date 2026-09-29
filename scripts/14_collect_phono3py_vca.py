#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402


REFERENCE_COLORS = {
    "Dagli et al.": "#c6c6c6",
    "Tran et al. (bulk)": "#7fb5dd",
    "Tran et al. (L=0.5um)": "#b9d7eb",
    "Tran et al. (L=0.5 um)": "#b9d7eb",
}
REFERENCE_LINESTYLES = {
    "Dagli et al.": (0, (1.2, 2.2)),
    "Tran et al. (bulk)": (0, (6.0, 2.4)),
    "Tran et al. (L=0.5um)": (0, (3.2, 2.0, 1.2, 2.0)),
    "Tran et al. (L=0.5 um)": (0, (3.2, 2.0, 1.2, 2.0)),
}


def read_summary(path: Path) -> tuple[float, float, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = data["tensor_summaries"][0]
    tensor = np.asarray(
        summary.get("cartesian_tensor_W_mK", summary.get("tensor_W_mK")), dtype=float
    )
    return float(np.trace(tensor) / 3.0), float((tensor[0, 0] + tensor[1, 1]) / 2.0), float(tensor[2, 2])


def fit_dagli(xs: np.ndarray, ks: np.ndarray) -> dict[str, float]:
    k_gan = float(ks[np.argmin(np.abs(xs))])
    k_aln = float(ks[np.argmin(np.abs(xs - 1.0))])
    rho_gan, rho_aln = 1.0 / k_gan, 1.0 / k_aln

    def model(params: np.ndarray, x: np.ndarray) -> np.ndarray:
        amplitude, p_al, q_ga = params
        disorder = np.zeros_like(x)
        mask = (x > 1e-12) & (x < 1.0 - 1e-12)
        disorder[mask] = amplitude * x[mask] ** p_al * (1.0 - x[mask]) ** q_ga
        return 1.0 / ((1.0 - x) * rho_gan + x * rho_aln + disorder)

    result = least_squares(
        lambda params: 1.0 / model(params, xs) - 1.0 / ks,
        x0=[0.35, 0.85, 0.65],
        bounds=([0.0, 0.05, 0.05], [100.0, 5.0, 5.0]),
        max_nfev=100000,
    )
    prediction = model(result.x, xs)
    residual = ks - prediction
    ss_tot = float(np.sum((ks - np.mean(ks)) ** 2))
    grid = np.linspace(0.0, 1.0, 1001)
    fitted = model(result.x, grid)
    minimum = int(np.argmin(fitted))
    return {
        "amplitude": float(result.x[0]),
        "p_al": float(result.x[1]),
        "q_ga": float(result.x[2]),
        "kappa_gan_W_mK": k_gan,
        "kappa_aln_W_mK": k_aln,
        "mae_W_mK": float(np.mean(np.abs(residual))),
        "rmse_W_mK": float(np.sqrt(np.mean(residual**2))),
        "r2": float(1.0 - np.sum(residual**2) / ss_tot),
        "fit_min_x_al": float(grid[minimum]),
        "fit_min_kappa_W_mK": float(fitted[minimum]),
    }


def read_benchmarks(path: Path | None) -> list[dict[str, object]]:
    if path is None:
        return []
    if not path.is_file():
        raise FileNotFoundError(f"Literature benchmark CSV does not exist: {path}")
    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                x_al = float(row["x_Al"])
                kappa = float(row["kappa_W_mK"])
            except (KeyError, TypeError, ValueError):
                continue
            rows.append(
                {
                    "x": x_al,
                    "kappa": kappa,
                    "label": row.get("label") or "Literature",
                    "kind": row.get("kind") or "line",
                }
            )
    return rows


def plot_benchmarks(ax, rows: list[dict[str, object]]) -> None:
    for label in sorted({str(row["label"]) for row in rows}):
        group = sorted(
            (row for row in rows if row["label"] == label),
            key=lambda row: float(row["x"]),
        )
        line_rows = [row for row in group if row["kind"] == "line"]
        point_rows = [row for row in group if row["kind"] != "line"]
        color = REFERENCE_COLORS.get(label, "#777777")
        if line_rows:
            ax.plot(
                [row["x"] for row in line_rows],
                [row["kappa"] for row in line_rows],
                color=color,
                lw=1.7,
                linestyle=REFERENCE_LINESTYLES.get(label, (0, (4.0, 2.0))),
                label=label,
                zorder=1,
            )
        if point_rows:
            ax.scatter(
                [row["x"] for row in point_rows],
                [row["kappa"] for row in point_rows],
                color=color,
                marker="d",
                s=28,
                label=None if line_rows else label,
                zorder=2,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect aligned phono3py AlGaN VCA results.")
    parser.add_argument("--vca-root", type=Path, required=True)
    parser.add_argument("--gan-result", type=Path, required=True)
    parser.add_argument("--aln-result", type=Path, required=True)
    parser.add_argument("--title", default="MACE-MH-1 MATPES-R2SCAN")
    parser.add_argument("--benchmark-csv", type=Path)
    parser.add_argument("--dft-gan-kappa", type=float)
    parser.add_argument("--dft-aln-kappa", type=float)
    args = parser.parse_args()

    rows: list[tuple[float, str, float, float, float]] = []
    for x_al, label, result in (
        (0.0, "GaN", args.gan_result),
        (1.0, "AlN", args.aln_result),
    ):
        rows.append((x_al, label, *read_summary(result)))
    expected = {
        0.125: "Al0.125Ga0.875N",
        0.23: "Al0.23Ga0.77N",
        0.25: "Al0.25Ga0.75N",
        0.375: "Al0.375Ga0.625N",
        0.5: "Al0.5Ga0.5N",
        0.625: "Al0.625Ga0.375N",
        0.75: "Al0.75Ga0.25N",
        0.875: "Al0.875Ga0.125N",
    }
    for x_al, composition in expected.items():
        result = args.vca_root / composition / "result.json"
        if not result.exists():
            raise FileNotFoundError(f"Missing completed VCA result: {result}")
        rows.append((x_al, composition, *read_summary(result)))
    rows.sort()

    csv_path = args.vca_root / "algan_vca_kappa_300K_vs_x.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x_Al", "composition", "kappa_avg_W_mK", "kappa_a_W_mK", "kappa_c_W_mK"])
        writer.writerows(rows)
    xs = np.asarray([row[0] for row in rows])
    ks = np.asarray([row[2] for row in rows])
    fit = fit_dagli(xs, ks)
    fit["equation"] = "kappa(x)=1/[(1-x)/kappa_GaN+x/kappa_AlN+A*x^p*(1-x)^q]"
    (args.vca_root / "algan_vca_kappa_300K_fit.json").write_text(
        json.dumps(fit, indent=2) + "\n", encoding="utf-8"
    )
    grid = np.linspace(0.0, 1.0, 501)
    rho = (
        (1.0 - grid) / fit["kappa_gan_W_mK"]
        + grid / fit["kappa_aln_W_mK"]
        + fit["amplitude"] * grid ** fit["p_al"] * (1.0 - grid) ** fit["q_ga"]
    )
    benchmarks = read_benchmarks(args.benchmark_csv)
    fig, ax = plt.subplots(figsize=(6.7, 4.05), constrained_layout=True)
    plot_benchmarks(ax, benchmarks)
    ax.plot(grid, 1.0 / rho, color="#0077c2", lw=2.2, label="MLIP fitting (bulk)")
    ax.scatter(xs[1:-1], ks[1:-1], color="#0C2340", s=34, label="MLIP", zorder=3)
    ax.scatter(
        [0.0, 1.0], [ks[0], ks[-1]], color="#d65244", marker="s", s=45,
        label="MLIP endpoints", zorder=4,
    )
    dft_values: list[float] = []
    if args.dft_gan_kappa is not None and args.dft_aln_kappa is not None:
        dft_values = [args.dft_gan_kappa, args.dft_aln_kappa]
        ax.scatter(
            [0.0, 1.0], dft_values, color="#fb6a4a", marker="^", s=52,
            label="DFT endpoints", zorder=5,
        )
    ax.set(
        xlabel=r"Al fraction $x$ in Al$_x$Ga$_{1-x}$N",
        ylabel=r"$\kappa_\mathrm{avg}$ at 300 K (W m$^{-1}$ K$^{-1}$)",
        title=args.title,
    )
    ax.set_xlim(-0.03, 1.03)
    all_y = list(ks) + dft_values + [float(row["kappa"]) for row in benchmarks]
    positive = np.asarray([value for value in all_y if value > 0], dtype=float)
    if positive.size and positive.max() / positive.min() > 15.0:
        ax.set_yscale("log")
        ax.set_ylim(positive.min() * 0.85, positive.max() * 1.18)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(
        frameon=False, fontsize=7.6, handlelength=2.4, loc="upper center",
        bbox_to_anchor=(0.58, 0.985), ncol=2, columnspacing=1.0,
    )
    fig.savefig(args.vca_root / "fig_algan_vca_kappa_300K_vs_x.png", dpi=300)
    plt.close(fig)
    print(f"[DONE] wrote {csv_path}")
