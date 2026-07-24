"""Figuras del reporte. Todas se guardan en ``artifacts/figures/``.

Requisito de la entrega: los resultados agregados se muestran con barras de error
(desviacion estandar entre folds).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # backend headless, sin ventana.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config, io_utils

_SHORT = {
    "custom_water_domain_r1_robust": "custom_r1",
    "custom_water_domain_r2_reupload_robust": "custom_r2_reup",
    "zz_ring_r2_robust": "zz_ring_r2",
    "zz_ring_r1_robust": "zz_ring_r1",
    "zz_linear_r1_robust": "zz_linear_r1",
    "zz_full_r1_robust": "zz_full_r1",
    "pauli_z_zz_linear_r1_robust": "pauli_lin_r1",
    "pauli_z_zz_ring_r1_robust": "pauli_ring_r1",
    "pauli_xz_xxzz_linear_r1_robust": "pauli_xz_lin",
    "pauli_z_zz_ring_r1_minmax": "pauli_ring_mm",
}


def short(name: str) -> str:
    return _SHORT.get(name, name)


def kernel_heatmaps() -> list:
    paths = []
    for map_name in io_utils.available_maps():
        kernel = io_utils.load_kernel(map_name)
        fig, ax = plt.subplots(figsize=(4.6, 4.0))
        im = ax.imshow(kernel, cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_title(f"Kernel exacto\n{short(map_name)}", fontsize=9)
        ax.set_xlabel("muestra j")
        ax.set_ylabel("muestra i")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="fidelidad")
        fig.tight_layout()
        out = config.ART_FIGURES / f"kernel_heatmap_{map_name}.png"
        fig.savefig(out, dpi=140)
        plt.close(fig)
        paths.append(out)
    return paths


def f1_ranking(best: pd.DataFrame) -> object:
    data = best.sort_values("f1_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.barh(
        [short(m) for m in data["map"]],
        data["f1_mean"],
        xerr=data["f1_std"],
        color="#3a7ca5",
        capsize=3,
    )
    ax.set_xlabel("F1 medio (CV 5 folds) ± desv. est.")
    ax.set_title("Ranking de mapas cuanticos por F1 (subconjunto 80 filas)")
    ax.axvline(0.5573, color="crimson", linestyle="--", linewidth=1,
               label="SVM-RBF chem5_v01 (CV F1 0.557)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out = config.ART_FIGURES / "map_f1_ranking.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def classical_vs_quantum(best: pd.DataFrame, quantum80_cv: pd.DataFrame) -> object:
    top = best.sort_values("f1_mean", ascending=False).head(3)
    labels = ["SVM-RBF\n(80 filas)"] + [short(m) for m in top["map"]]
    means = [float(quantum80_cv["f1_mean"].iloc[0])] + list(top["f1_mean"])
    errs = [float(quantum80_cv["f1_std"].iloc[0])] + list(top["f1_std"])
    colors = ["#8d99ae"] + ["#2a9d8f"] * len(top)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.bar(labels, means, yerr=errs, color=colors, capsize=4)
    ax.set_ylabel("F1 medio (CV) ± desv. est.")
    ax.set_title("Clasico vs. QSVM en el mismo subconjunto de 80 filas")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    out = config.ART_FIGURES / "classical_vs_quantum_f1.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def holdout_confusion(holdout_row: pd.DataFrame) -> object:
    row = holdout_row.iloc[0]
    matrix = np.array([[int(row["tn"]), int(row["fp"])],
                       [int(row["fn"]), int(row["tp"])]])
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    im = ax.imshow(matrix, cmap="Blues")
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, str(value), ha="center", va="center",
                color="white" if value > matrix.max() / 2 else "black", fontsize=13)
    ax.set_xticks([0, 1], ["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], ["Real 0", "Real 1"])
    ax.set_title("SVM-RBF chem5_v01 - holdout (656 filas)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out = config.ART_FIGURES / "classical_holdout_confusion.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def run(quantum_results: dict, classical_results: dict) -> list:
    config.ART_FIGURES.mkdir(parents=True, exist_ok=True)
    outputs = []
    outputs.extend(kernel_heatmaps())
    outputs.append(f1_ranking(quantum_results["best"]))
    outputs.append(classical_vs_quantum(quantum_results["best"], classical_results["quantum80_cv"]))
    outputs.append(holdout_confusion(classical_results["holdout"]))
    return outputs
