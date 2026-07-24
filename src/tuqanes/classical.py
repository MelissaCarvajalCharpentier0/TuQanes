"""Etapa clasica: consolida el baseline SVM-RBF congelado y sus figuras.

El baseline clasico completo (division train/holdout, submuestreo, grilla C x
gamma y evaluacion en holdout) se ejecuto en la generacion ``last_version``
(Partes 1-2). Aqui se consolida la evidencia congelada del espacio de features
``chem5_v01`` sobre el dataset completo y sobre el subconjunto cuantico de 80
filas, y se regeneran las tablas de reporte. El recomputo desde datos crudos vive
en el notebook intacto ``Hacia_el_agua_limpia_partes_1_2_SVM_RBF.ipynb``.
"""

from __future__ import annotations

import pandas as pd

from . import config, io_utils


def full_dataset_baseline() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fila CV y fila holdout del espacio ``chem5_v01`` sobre el dataset completo."""
    cv = io_utils.load_csv(config.ARTIFACTS_V12 / "feature_cv_summary.csv")
    holdout = io_utils.load_csv(config.ARTIFACTS_V12 / "feature_holdout_metrics.csv")

    cv_v01 = cv.loc[cv["feature_set"] == config.FEATURE_SET].copy()
    # Config congelada del README: C=10, gamma=auto (mejor F1 de chem5_v01).
    cv_best = (
        cv_v01.sort_values(config.RANK_KEYS, ascending=False)
        .head(1)
        .reset_index(drop=True)
    )
    holdout_v01 = holdout.loc[holdout["feature_set"] == config.FEATURE_SET].reset_index(drop=True)
    return cv_best, holdout_v01


def quantum80_baseline() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Baseline clasico sobre el subconjunto de 80 filas (referencia para la QSVM)."""
    cv = io_utils.load_csv(config.ARTIFACTS_V12 / "quantum80_svm_cv_summary.csv")
    locked = cv.loc[
        (cv["C"] == config.LOCKED_CLASSICAL_C)
        & (cv["gamma"].astype(str) == str(config.LOCKED_CLASSICAL_GAMMA))
    ].reset_index(drop=True)
    oof = io_utils.load_csv(config.ARTIFACTS_V12 / "quantum80_svm_oof_metrics.csv")
    return locked, oof


def run() -> dict[str, pd.DataFrame]:
    config.ART_CLASSICAL.mkdir(parents=True, exist_ok=True)
    cv_best, holdout_v01 = full_dataset_baseline()
    q80_cv, q80_oof = quantum80_baseline()

    cv_best.to_csv(config.ART_CLASSICAL / "full_dataset_cv_chem5_v01.csv", index=False)
    holdout_v01.to_csv(config.ART_CLASSICAL / "full_dataset_holdout_chem5_v01.csv", index=False)
    q80_cv.to_csv(config.ART_CLASSICAL / "quantum80_svm_cv_locked.csv", index=False)
    q80_oof.to_csv(config.ART_CLASSICAL / "quantum80_svm_oof_metrics.csv", index=False)

    return {
        "cv_best": cv_best,
        "holdout": holdout_v01,
        "quantum80_cv": q80_cv,
        "quantum80_oof": q80_oof,
    }
