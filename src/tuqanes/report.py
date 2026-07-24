"""Consolidacion final: tabla maestra clasico-vs-cuantico y resumen en Markdown."""

from __future__ import annotations

import pandas as pd

from . import config


def _to_markdown(frame: pd.DataFrame) -> str:
    """Tabla Markdown simple sin depender de ``tabulate``."""
    headers = list(frame.columns)
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in headers) + " |")
    return "\n".join(lines)


def master_table(classical: dict, quantum: dict, nexus: dict) -> pd.DataFrame:
    """Tabla comparativa unica (CV, mismo protocolo de folds cuando aplica)."""
    rows: list[dict] = []

    cvb = classical["cv_best"].iloc[0]
    rows.append({
        "stage": "clasico",
        "model": "SVM-RBF chem5_v01 (dataset completo, CV)",
        "n": 2620,
        "f1_mean": round(float(cvb["f1_mean"]), 4),
        "f1_std": round(float(cvb.get("f1_std", float("nan"))), 4),
        "balanced_accuracy_mean": round(float(cvb["balanced_accuracy_mean"]), 4),
        "mcc_mean": round(float(cvb["mcc_mean"]), 4),
    })

    hold = classical["holdout"].iloc[0]
    rows.append({
        "stage": "clasico",
        "model": "SVM-RBF chem5_v01 (holdout 656)",
        "n": 656,
        "f1_mean": round(float(hold["f1"]), 4),
        "f1_std": float("nan"),
        "balanced_accuracy_mean": round(float(hold["balanced_accuracy"]), 4),
        "mcc_mean": round(float(hold["mcc"]), 4),
    })

    q80 = classical["quantum80_cv"].iloc[0]
    rows.append({
        "stage": "clasico",
        "model": "SVM-RBF (subconjunto 80, C=10 gamma=0.01)",
        "n": 80,
        "f1_mean": round(float(q80["f1_mean"]), 4),
        "f1_std": round(float(q80["f1_std"]), 4),
        "balanced_accuracy_mean": round(float(q80["balanced_accuracy_mean"]), 4),
        "mcc_mean": round(float(q80["mcc_mean"]), 4),
    })

    for _, r in quantum["best"].sort_values("f1_mean", ascending=False).iterrows():
        rows.append({
            "stage": "cuantico",
            "model": f"QSVM {r['map']} (C={r['C']})",
            "n": 80,
            "f1_mean": round(float(r["f1_mean"]), 4),
            "f1_std": round(float(r["f1_std"]), 4),
            "balanced_accuracy_mean": round(float(r["balanced_accuracy_mean"]), 4),
            "mcc_mean": round(float(r["mcc_mean"]), 4),
        })

    return pd.DataFrame(rows)


def markdown_summary(master: pd.DataFrame, quantum: dict, nexus: dict) -> str:
    best_q = quantum["best"].sort_values("f1_mean", ascending=False).iloc[0]
    lines = [
        "# Resumen consolidado de resultados (TuQanes)",
        "",
        "Generado por `main.py`. Todas las cifras se reproducen desde datos y kernels",
        "congelados. La fase Nexus/H2 NO se ejecuta desde este flujo.",
        "",
        "## Tabla maestra",
        "",
        _to_markdown(master),
        "",
        "## Lectura",
        "",
        f"- Mejor mapa cuantico por F1 (CV, 80 filas): **{best_q['map']}** "
        f"con F1 = {best_q['f1_mean']:.4f} ± {best_q['f1_std']:.4f} (C={best_q['C']}).",
        "- El baseline clasico completo alcanza mayor F1/MCC en holdout independiente; "
        "la ventaja cuantica **no** esta demostrada.",
        "- La fase Nexus queda empaquetada y modular en "
        "`notebooks/Johnny/nexus_reproducible/TuQanes_Package_Nexus` (no ejecutada aqui).",
        "",
    ]
    return "\n".join(lines)


def run(classical: dict, quantum: dict, nexus: dict) -> dict:
    config.ART_COMPARISON.mkdir(parents=True, exist_ok=True)
    master = master_table(classical, quantum, nexus)
    master.to_csv(config.ART_COMPARISON / "master_metrics.csv", index=False)

    summary_md = markdown_summary(master, quantum, nexus)
    (config.ARTIFACTS / "RESULTS_SUMMARY.md").write_text(summary_md, encoding="utf-8")
    return {"master": master, "summary_md": summary_md}
