"""TuQanes - pipeline reproducible del Challenge 2 (agua potable, SVM-RBF vs QSVM).

Paquete modular usado por ``main.py`` (raiz) y por el notebook de punto de entrada
``notebooks/TuQanes_entrypoint.ipynb``. La fase Nexus/H2 se mantiene modular y no
se ejecuta desde este flujo.
"""

from __future__ import annotations

from . import classical, config, io_utils, metrics, nexus, plots, quantum, report

__all__ = [
    "classical",
    "config",
    "io_utils",
    "metrics",
    "nexus",
    "plots",
    "quantum",
    "report",
]
