"""Etapa cuantica: consolida las cifras QSVM reportadas y las verifica.

Los kernels de fidelidad (statevector, sin shots ni ruido) se calcularon con
``pytket`` en la generacion ``last_version`` (Partes 3-4) y quedaron congelados en
``artifacts_v3_4/``. Estrategia de reproducibilidad:

1. Las cifras REPORTADAS (ranking por mapa, grilla completa de C, geometria del
   kernel y ablaciones) se cargan desde los artefactos congelados y se copian a
   ``artifacts/quantum/``. Son la fuente de verdad del informe.
2. Como verificacion independiente, se vuelve a ajustar una QSVM con kernel
   precomputado sobre los mismos kernels y los folds congelados. El resultado se
   guarda como ``qsvm_recompute_check.csv`` y se reporta la diferencia maxima de
   F1 frente a lo congelado. Corrobora la conclusion (cuantico ~= clasico) sin
   depender de ``pytket``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, io_utils
from .metrics import evaluate_precomputed_kernel, kernel_diagnostics


# ---------------------------------------------------------------------------
# 1. Cifras reportadas (congeladas)
# ---------------------------------------------------------------------------
def load_frozen() -> dict[str, pd.DataFrame]:
    v34 = config.ARTIFACTS_V34
    return {
        "best": io_utils.load_csv(v34 / "map_cv_best_by_map.csv"),
        "summary_all_c": io_utils.load_csv(v34 / "map_cv_summary_all_C.csv"),
        "geometry": io_utils.load_csv(v34 / "kernel_geometry_summary.csv"),
        "ablation_family": io_utils.load_csv(v34 / "ablation_by_family.csv"),
        "ablation_topology": io_utils.load_csv(v34 / "ablation_by_topology.csv"),
        "ablation_scaling": io_utils.load_csv(v34 / "ablation_by_scaling.csv"),
        "resources": io_utils.load_csv(v34 / "circuit_resource_summary.csv"),
        "shots": io_utils.load_csv(v34 / "shot_sensitivity_summary.csv"),
        "noise": io_utils.load_csv(v34 / "noise_proxy_sensitivity_summary.csv"),
    }


# ---------------------------------------------------------------------------
# 2. Verificacion independiente desde los kernels cacheados
# ---------------------------------------------------------------------------
def recompute_check() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Re-ajusta la QSVM (kernel precomputado) sobre la grilla de C y folds."""
    labels, folds, _ = io_utils.load_labels_folds()
    parts: list[pd.DataFrame] = []
    diag_rows: list[dict] = []
    for map_name in io_utils.available_maps():
        kernel = io_utils.load_kernel(map_name)
        if kernel.shape[0] != len(labels):
            raise ValueError(
                f"El kernel de {map_name} ({kernel.shape[0]}) no coincide con "
                f"las {len(labels)} filas del subconjunto."
            )
        for c_value in config.C_GRID:
            _, summary = evaluate_precomputed_kernel(kernel, labels, folds, c_value, map_name)
            parts.append(summary)
        diag_rows.append({"map": map_name, **kernel_diagnostics(kernel)})

    summary_all_c = pd.concat(parts, ignore_index=True)
    best = (
        summary_all_c.sort_values(config.RANK_KEYS, ascending=False)
        .drop_duplicates("map")
        .sort_values("f1_mean", ascending=False)
        .reset_index(drop=True)
    )
    return best, pd.DataFrame(diag_rows)


def _f1_agreement(frozen_best: pd.DataFrame, recomputed_best: pd.DataFrame) -> float:
    """Diferencia absoluta maxima de F1 por mapa entre congelado y recomputo."""
    a = frozen_best.set_index("map")["f1_mean"]
    b = recomputed_best.set_index("map")["f1_mean"]
    common = a.index.intersection(b.index)
    if len(common) == 0:
        return float("nan")
    return float(np.max(np.abs(a.loc[common].to_numpy() - b.loc[common].to_numpy())))


def run() -> dict[str, object]:
    config.ART_QUANTUM.mkdir(parents=True, exist_ok=True)
    frozen = load_frozen()

    # Cifras reportadas -> artifacts/quantum/
    frozen["best"].to_csv(config.ART_QUANTUM / "map_cv_best_by_map.csv", index=False)
    frozen["summary_all_c"].to_csv(config.ART_QUANTUM / "map_cv_summary_all_C.csv", index=False)
    frozen["geometry"].to_csv(config.ART_QUANTUM / "kernel_geometry_summary.csv", index=False)
    frozen["ablation_family"].to_csv(config.ART_QUANTUM / "ablation_by_family.csv", index=False)
    frozen["ablation_topology"].to_csv(config.ART_QUANTUM / "ablation_by_topology.csv", index=False)
    frozen["ablation_scaling"].to_csv(config.ART_QUANTUM / "ablation_by_scaling.csv", index=False)
    frozen["resources"].to_csv(config.ART_QUANTUM / "circuit_resource_summary.csv", index=False)
    frozen["shots"].to_csv(config.ART_QUANTUM / "shot_sensitivity_summary.csv", index=False)
    frozen["noise"].to_csv(config.ART_QUANTUM / "noise_proxy_sensitivity_summary.csv", index=False)

    # Verificacion independiente.
    recomputed_best, diagnostics = recompute_check()
    recomputed_best.to_csv(config.ART_QUANTUM / "qsvm_recompute_check.csv", index=False)
    diagnostics.to_csv(config.ART_QUANTUM / "kernel_diagnostics.csv", index=False)
    max_diff = _f1_agreement(frozen["best"], recomputed_best)

    return {
        "best": frozen["best"],
        "summary_all_c": frozen["summary_all_c"],
        "geometry": frozen["geometry"],
        "ablation_family": frozen["ablation_family"],
        "ablation_topology": frozen["ablation_topology"],
        "ablation_scaling": frozen["ablation_scaling"],
        "recomputed_best": recomputed_best,
        "recompute_max_f1_diff": max_diff,
    }
