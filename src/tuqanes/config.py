"""Rutas y constantes compartidas del pipeline reproducible de TuQanes.

Todo el paquete se ancla a la raiz del repositorio (donde vive ``main.py``) para
que la ejecucion sea independiente del directorio de trabajo. Los artefactos de
entrada se leen desde la generacion vigente ``last_version`` (intacta) y toda la
salida se consolida en ``artifacts/`` en la raiz.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
# src/tuqanes/config.py -> parents[0]=tuqanes, [1]=src, [2]=raiz del repo.
REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "data"
DATASET_CSV = DATA_DIR / "water_potability.csv"

# Generacion vigente de Johnny (solo lectura, nunca se modifica).
LAST_VERSION = REPO_ROOT / "notebooks" / "Johnny" / "last_version"
ARTIFACTS_V12 = LAST_VERSION / "artifacts_v1_2"        # baseline clasico + subconjunto 80
ARTIFACTS_V34 = LAST_VERSION / "artifacts_v3_4"        # kernels cuanticos + analisis
KERNELS_DIR = ARTIFACTS_V34 / "kernels"

# Fase Nexus: paquete modular (NO se ejecuta desde este pipeline).
NEXUS_ROOT = REPO_ROOT / "notebooks" / "Johnny" / "nexus_reproducible"
NEXUS_PACKAGE = NEXUS_ROOT / "TuQanes_Package_Nexus"

# Trabajo multiseed de Luis (contexto complementario).
LUIS_DIR = REPO_ROOT / "notebooks" / "Luis"

# Salida unificada.
ARTIFACTS = REPO_ROOT / "artifacts"
ART_CLASSICAL = ARTIFACTS / "classical"
ART_QUANTUM = ARTIFACTS / "quantum"
ART_COMPARISON = ARTIFACTS / "comparison"
ART_NEXUS = ARTIFACTS / "nexus"
ART_FIGURES = ARTIFACTS / "figures"

ALL_OUTPUT_DIRS = (ART_CLASSICAL, ART_QUANTUM, ART_COMPARISON, ART_NEXUS, ART_FIGURES)

# ---------------------------------------------------------------------------
# Contrato de datos
# ---------------------------------------------------------------------------
TARGET = "Potability"
CLASS_LABELS = (0, 1)
FEATURE_SET = "chem5_v01"
FEATURES = ["Sulfate", "ph", "Conductivity", "Chloramines", "Hardness"]
EXPECTED_DATA_SHA256 = "904004bde729bfe3d2e195f46343bceead09e32a0eb95bb8184e7e20e029b2bf"

# Grilla de C usada en toda la seleccion (clasica y cuantica).
C_GRID = (0.1, 1.0, 10.0)

# Baseline clasico congelado sobre el subconjunto cuantico de 80 filas.
LOCKED_CLASSICAL_C = 10.0
LOCKED_CLASSICAL_GAMMA = 0.01

# Criterio de desempate para elegir el mejor C por mapa.
RANK_KEYS = ["f1_mean", "balanced_accuracy_mean", "mcc_mean"]

# Sufijo de los kernels exactos cacheados (statevector, sin shots ni ruido).
KERNEL_SUFFIX = "_global_kernel.npy"
