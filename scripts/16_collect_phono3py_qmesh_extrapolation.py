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


COMPOSITIONS = {
    0.125: "Al0.125Ga0.875N",
    0.23: "Al0.23Ga0.77N",
    0.25: "Al0.25Ga0.75N",
    0.375: "Al0.375Ga0.625N",
    0.5: "Al0.5Ga0.5N",
    0.625: "Al0.625Ga0.375N",
    0.75: "Al0.75Ga0.25N",
    0.875: "Al0.875Ga0.125N",
}
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


def read_summary(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    tensor = np.asarray(data["tensor_summaries"][0]["cartesian_tensor_W_mK"], dtype=float)
    return {
        "kavg": float(np.trace(tensor) / 3.0),
        "ka": float((tensor[0, 0] + tensor[1, 1]) / 2.0),
        "kc": float(tensor[2, 2]),
    }


def linear_extrapolate(meshes: list[tuple[int, int, int]], values: list[float]) -> dict[str, float]:
    x = np.asarray([np.prod(mesh) ** (-1.0 / 3.0) for mesh in meshes], dtype=float)
    y = np.asarray(values, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    fitted = intercept + slope * x
    residual = y - fitted
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    dof = len(x) - 2
    sxx = float(np.sum((x - np.mean(x)) ** 2))
    intercept_stderr = float(
        np.sqrt((ss_res / dof) * (1.0 / len(x) + np.mean(x) ** 2 / sxx))
    ) if dof > 0 and sxx > 0 else float("nan")
    return {
        "intercept": float(intercept),
        "slope": float(slope),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 1.0,
        "intercept_stderr": intercept_stderr,
    }


def fit_dagli(xs: np.ndarray, ks: np.ndarray) -> dict[str, object]:
    k_gan, k_aln = float(ks[0]), float(ks[-1])

    def rho(params: np.ndarray, x: np.ndarray) -> np.ndarray:
        amplitude, p_al, q_ga = params
        return (
            (1.0 - x) / k_gan + x / k_aln
            + amplitude * x**p_al * (1.0 - x) ** q_ga
        )

    result = least_squares(
        lambda params: rho(params, xs) - 1.0 / ks,
        x0=[0.2, 0.85, 0.65],
        bounds=([0.0, 0.05, 0.05], [100.0, 5.0, 5.0]),
        max_nfev=100000,
    )
    grid = np.linspace(0.0, 1.0, 1001)
    prediction = 1.0 / rho(result.x, xs)
    fitted = 1.0 / rho(result.x, grid)
    residual = ks - prediction
    ss_tot = float(np.sum((ks - np.mean(ks)) ** 2))
    minimum = int(np.argmin(fitted))
    return {
        "amplitude": float(result.x[0]),
        "p_al": float(result.x[1]),
        "q_ga": float(result.x[2]),
        "kappa_gan_inf_W_mK": k_gan,
        "kappa_aln_inf_W_mK": k_aln,
        "mae_W_mK": float(np.mean(np.abs(residual))),
        "rmse_W_mK": float(np.sqrt(np.mean(residual**2))),
        "r2": float(1.0 - np.sum(residual**2) / ss_tot),
        "fit_min_x_al": float(grid[minimum]),
        "fit_min_kappa_W_mK": float(fitted[minimum]),
        "grid": grid,
        "fitted": fitted,
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
                rows.append(
                    {
                        "x": float(row["x_Al"]),
                        "kappa": float(row["kappa_W_mK"]),
                        "label": row.get("label") or "Literature",
                        "kind": row.get("kind") or "line",
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
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


def plot_reference_comparison(
    *,
    xs: np.ndarray,
    k_inf: np.ndarray,
    dagli: dict[str, object],
    benchmarks: list[dict[str, object]],
    dft_gan: float | None,
    dft_aln: float | None,
    path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.7, 4.05), constrained_layout=True)
    plot_benchmarks(ax, benchmarks)
    ax.plot(
        dagli["grid"], dagli["fitted"], color="#0077c2", lw=2.3,
        label="MLIP extrapolated fitting", zorder=4,
    )
    ax.scatter(
        xs[1:-1], k_inf[1:-1], color="#0C2340", s=36,
        label="MLIP extrapolated", zorder=5,
    )
    ax.scatter(
        [0.0, 1.0], [k_inf[0], k_inf[-1]], color="#d65244", marker="s",
        s=47, label="MLIP extrapolated endpoints", zorder=6,
    )
    dft_values: list[float] = []
    if dft_gan is not None and dft_aln is not None:
        dft_values = [dft_gan, dft_aln]
        ax.scatter(
            [0.0, 1.0], dft_values, color="#fb6a4a", marker="^", s=52,
            label="DFT endpoints", zorder=7,
        )
    ax.set(
        xlabel=r"Al fraction $x$ in Al$_x$Ga$_{1-x}$N",
        ylabel=r"$\kappa_\mathrm{avg}$ at 300 K (W m$^{-1}$ K$^{-1}$)",
        title=title,
    )
    ax.set_xlim(-0.03, 1.03)
    all_y = list(k_inf) + dft_values + [float(row["kappa"]) for row in benchmarks]
    positive = np.asarray([value for value in all_y if value > 0], dtype=float)
    if positive.size and positive.max() / positive.min() > 15.0:
        ax.set_yscale("log")
        ax.set_ylim(positive.min() * 0.85, positive.max() * 1.18)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(
        frameon=False, fontsize=7.4, handlelength=2.4, loc="upper center",
        bbox_to_anchor=(0.58, 0.985), ncol=2, columnspacing=1.0,
    )
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extrapolate aligned phono3py VCA qmesh scans.")
    parser.add_argument(
        "--mesh-result", action="append", nargs=7, metavar=("TAG", "N1", "N2", "N3", "VCA_ROOT", "GAN_JSON", "ALN_JSON"), required=True
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--title", default="MACE-MH-1 MATPES-R2SCAN")
    parser.add_argument("--benchmark-csv", type=Path)
    parser.add_argument("--dft-gan-kappa", type=float)
    parser.add_argument("--dft-aln-kappa", type=float)
    args = parser.parse_args()

    specs = []
    for tag, n1, n2, n3, vca_root, gan_json, aln_json in args.mesh_result:
        specs.append((tag, (int(n1), int(n2), int(n3)), Path(vca_root), Path(gan_json), Path(aln_json)))
    specs.sort(key=lambda item: np.prod(item[1]))
    if len(specs) < 3:
        raise ValueError("At least three qmesh results are required for extrapolation")

    points = [(0.0, "GaN"), *COMPOSITIONS.items(), (1.0, "AlN")]
    rows: list[dict[str, object]] = []
    for x_al, composition in points:
        values = {key: [] for key in ("kavg", "ka", "kc")}
        meshes = []
        row: dict[str, object] = {"x_Al": x_al, "composition": composition}
        for tag, mesh, root, gan_json, aln_json in specs:
            path = gan_json if x_al == 0.0 else aln_json if x_al == 1.0 else root / composition / "result.json"
            if not path.is_file():
                raise FileNotFoundError(f"Missing qmesh result: {path}")
            summary = read_summary(path)
            meshes.append(mesh)
            for key, value in summary.items():
                values[key].append(value)
                row[f"{key}_{tag}"] = value
        for key in values:
            fit = linear_extrapolate(meshes, values[key])
            row[f"{key}_inf"] = fit["intercept"]
            row[f"{key}_slope"] = fit["slope"]
            row[f"{key}_r2"] = fit["r2"]
            row[f"{key}_inf_stderr"] = fit["intercept_stderr"]
        rows.append(row)

    args.output_root.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_root / "algan_kappa_qmesh_extrapolated_300K.csv"
    fieldnames = list(rows[0])
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    xs = np.asarray([float(row["x_Al"]) for row in rows])
    k_inf = np.asarray([float(row["kavg_inf"]) for row in rows])
    dagli = fit_dagli(xs, k_inf)
    fit_json = {key: value for key, value in dagli.items() if key not in {"grid", "fitted"}}
    fit_json["equation"] = "kappa(x)=1/[(1-x)/kappa_GaN+x/kappa_AlN+A*x^p*(1-x)^q]"
    fit_json["qmesh_equation"] = "kappa(N)=kappa_inf+S/(N1*N2*N3)^(1/3)"
    (args.output_root / "algan_kappa_qmesh_extrapolated_300K_fit.json").write_text(
        json.dumps(fit_json, indent=2) + "\n", encoding="utf-8"
    )

    fig, ax = plt.subplots(figsize=(6.7, 4.2), constrained_layout=True)
    for index, (tag, mesh, *_rest) in enumerate(specs):
        ax.plot(
            xs, [float(row[f"kavg_{tag}"]) for row in rows], "--", lw=1.0,
            alpha=0.34 + 0.08 * index, label=f"MLIP {mesh[0]}x{mesh[1]}x{mesh[2]}",
        )
    ax.scatter(xs, k_inf, color="#0C2340", s=36, label="MLIP extrapolated", zorder=5)
    ax.plot(dagli["grid"], dagli["fitted"], color="#0077c2", lw=2.3, label="MLIP extrapolated fitting", zorder=4)
    ax.set(
        xlabel=r"Al fraction $x$ in Al$_x$Ga$_{1-x}$N",
        ylabel=r"$\kappa_\mathrm{avg}$ at 300 K (W m$^{-1}$ K$^{-1}$)",
        title=args.title,
    )
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(frameon=False, fontsize=7.4, ncol=2)
    fig.savefig(args.output_root / "fig_algan_kappa_qmesh_extrapolated_300K.png", dpi=300)
    plt.close(fig)
    if args.benchmark_csv is not None or (
        args.dft_gan_kappa is not None and args.dft_aln_kappa is not None
    ):
        plot_reference_comparison(
            xs=xs,
            k_inf=k_inf,
            dagli=dagli,
            benchmarks=read_benchmarks(args.benchmark_csv),
            dft_gan=args.dft_gan_kappa,
            dft_aln=args.dft_aln_kappa,
            path=args.output_root / "fig_algan_kappa_qmesh_extrapolated_300K_compare_reference.png",
            title=args.title.replace(" qmesh extrapolation", ""),
        )
    print(f"[DONE] wrote {csv_path}")


if __name__ == "__main__":
    main()
