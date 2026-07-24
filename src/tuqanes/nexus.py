"""Etapa Nexus (solo lectura): resume los resultados locales, sin ejecutar Nexus.

La fase de hardware/emulador vive de forma autocontenida en el paquete
``notebooks/Johnny/nexus_reproducible/TuQanes_Package_Nexus`` y NO se ejecuta desde
este pipeline (requiere cuota de Quantinuum). Aqui solo se recogen los resultados
exactos locales ya calculados (referencia exacto-vs-Nexus) para incluirlos en el
reporte consolidado. Todo queda claramente etiquetado como no proveniente de H2.
"""

from __future__ import annotations

import pandas as pd

from . import config


def _first_existing(*candidates):
    for path in candidates:
        if path.exists():
            return path
    return None


def local_comparison() -> pd.DataFrame | None:
    """Comparacion local exacta 16 vs 64 muestras (no son resultados de H2)."""
    path = _first_existing(
        config.NEXUS_ROOT / "comparacion_local_16_64.csv",
        config.NEXUS_PACKAGE / "comparacion_local_16_64.csv",
    )
    if path is None:
        return None
    return pd.read_csv(path)


def package_status() -> pd.DataFrame:
    """Inventario del paquete Nexus modular: presente pero no ejecutado."""
    present = config.NEXUS_PACKAGE.exists()
    files = (
        sorted(p.name for p in config.NEXUS_PACKAGE.iterdir())
        if present else []
    )
    return pd.DataFrame([
        {
            "component": "TuQanes_Package_Nexus",
            "present": present,
            "executed_by_pipeline": False,
            "note": "Copiar a Nexus Lab y ejecutar por etapas (local_only/cost/submit/collect).",
            "contents": ", ".join(files),
        }
    ])


def run() -> dict[str, pd.DataFrame]:
    config.ART_NEXUS.mkdir(parents=True, exist_ok=True)
    status = package_status()
    status.to_csv(config.ART_NEXUS / "package_status.csv", index=False)

    comparison = local_comparison()
    if comparison is not None:
        comparison.to_csv(config.ART_NEXUS / "local_exact_comparison_16_64.csv", index=False)
    return {"status": status, "comparison": comparison}
