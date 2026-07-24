"""Metricas de clasificacion y evaluacion de kernels precomputados.

La logica replica exactamente la usada en la fase Nexus
(``nexus_reproducible/nexus_qsvm.py``) para que las cifras reproducidas coincidan
con las reportadas en los artefactos congelados.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.svm import SVC

from .config import CLASS_LABELS

METRIC_NAMES = (
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "specificity",
    "f1",
    "mcc",
)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    """Metricas por fold, incluyendo la matriz de confusion desagregada."""
    matrix = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)
    tn, fp, fn, tp = [int(value) for value in matrix.ravel()]
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def summarize_folds(fold_metrics: pd.DataFrame) -> dict[str, float]:
    """Media y desviacion estandar (ddof=1) de cada metrica entre folds."""
    summary: dict[str, float] = {}
    for metric in METRIC_NAMES:
        summary[f"{metric}_mean"] = float(fold_metrics[metric].mean())
        summary[f"{metric}_std"] = float(fold_metrics[metric].std(ddof=1))
    return summary


def evaluate_precomputed_kernel(
    kernel: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    c_value: float,
    map_name: str,
    result_kind: str = "exact",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """QSVM con kernel precomputado y validacion cruzada por folds congelados.

    Devuelve (metricas_por_fold, resumen). Un ``SVC`` con ``kernel='precomputed'``
    se entrena con la submatriz train-train y se evalua con la submatriz
    valid-train, exactamente como en la fase de seleccion.
    """
    fold_rows: list[dict[str, Any]] = []
    for fold in sorted(np.unique(folds)):
        train_idx = np.flatnonzero(folds != fold)
        valid_idx = np.flatnonzero(folds == fold)
        model = SVC(kernel="precomputed", C=float(c_value))
        model.fit(kernel[np.ix_(train_idx, train_idx)], labels[train_idx])
        prediction = model.predict(kernel[np.ix_(valid_idx, train_idx)])
        row: dict[str, Any] = {
            "map": map_name,
            "result_kind": result_kind,
            "C": float(c_value),
            "fold": int(fold),
            "n_train": int(len(train_idx)),
            "n_valid": int(len(valid_idx)),
        }
        row.update(classification_metrics(labels[valid_idx], prediction))
        fold_rows.append(row)
    fold_df = pd.DataFrame(fold_rows)
    summary = {
        "map": map_name,
        "result_kind": result_kind,
        "C": float(c_value),
        **summarize_folds(fold_df),
    }
    return fold_df, pd.DataFrame([summary])


def kernel_diagnostics(kernel: np.ndarray) -> dict[str, float]:
    """Diagnosticos de simetria, diagonal y espectro de un kernel."""
    symmetric = 0.5 * (kernel + kernel.T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    mask = ~np.eye(len(kernel), dtype=bool)
    return {
        "symmetry_max_abs": float(np.max(np.abs(kernel - kernel.T))),
        "diagonal_max_abs_error": float(np.max(np.abs(np.diag(kernel) - 1.0))),
        "minimum": float(kernel.min()),
        "maximum": float(kernel.max()),
        "offdiagonal_mean": float(kernel[mask].mean()),
        "offdiagonal_std": float(kernel[mask].std(ddof=1)),
        "minimum_eigenvalue": float(eigenvalues.min()),
        "negative_eigenvalue_count": int(np.sum(eigenvalues < -1e-10)),
    }


def kernel_geometry(kernel: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Alineamiento kernel-target y separacion intra/inter clase."""
    n = len(kernel)
    y = labels.astype(float)
    ideal = np.equal.outer(y, y).astype(float)  # 1 si misma clase, 0 si no.
    # Alineamiento centrado de kernel (Cristianini): <K, YY^T> / (||K|| ||YY^T||).
    kf = float(np.sum(kernel * ideal))
    denom = float(np.linalg.norm(kernel) * np.linalg.norm(ideal))
    alignment = kf / denom if denom else float("nan")

    mask_off = ~np.eye(n, dtype=bool)
    same = ideal.astype(bool) & mask_off
    diff = (~ideal.astype(bool)) & mask_off
    intra_mean = float(kernel[same].mean()) if same.any() else float("nan")
    inter_mean = float(kernel[diff].mean()) if diff.any() else float("nan")
    eigenvalues = np.linalg.eigvalsh(0.5 * (kernel + kernel.T))
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    total = float(eigenvalues.sum())
    effective_rank = (total ** 2) / float(np.sum(eigenvalues ** 2)) if total else float("nan")
    return {
        "alignment": alignment,
        "intra_mean": intra_mean,
        "inter_mean": inter_mean,
        "intra_minus_inter": intra_mean - inter_mean,
        "effective_rank": effective_rank,
        "min_eigenvalue": float(eigenvalues.min()),
        "max_eigenvalue": float(eigenvalues.max()),
    }
