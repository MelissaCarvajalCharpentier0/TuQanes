"""Carga de datos congelados: subconjunto de 80 filas, folds, kernels y metadatos."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_dataset() -> dict[str, object]:
    """Verifica el hash del dataset canonico. No aborta el flujo si falta."""
    if not config.DATASET_CSV.exists():
        return {"present": False, "hash": None, "matches": False}
    digest = sha256_file(config.DATASET_CSV)
    return {
        "present": True,
        "hash": digest,
        "matches": digest == config.EXPECTED_DATA_SHA256,
    }


def load_quantum80() -> pd.DataFrame:
    """Subconjunto cuantico de 80 filas con etiqueta y fold congelados.

    Se ordena por ``position`` para alinear con los kernels globales cacheados.
    """
    folds = pd.read_csv(config.ARTIFACTS_V12 / "quantum80_folds.csv")
    return folds.sort_values("position").reset_index(drop=True)


def load_labels_folds() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = load_quantum80()
    labels = frame[config.TARGET].to_numpy(dtype=int)
    fold_ids = frame["validation_fold"].to_numpy(dtype=int)
    source_index = frame["source_index"].to_numpy(dtype=int)
    return labels, fold_ids, source_index


def map_metadata() -> pd.DataFrame:
    """Metadatos por mapa (familia, topologia, reps, scaling) desde el artefacto.

    Es la unica fuente de la lista de mapas evaluados; se reutiliza para las
    ablaciones sin fijar nombres a mano.
    """
    best = pd.read_csv(config.ARTIFACTS_V34 / "map_cv_best_by_map.csv")
    cols = ["map", "family", "scaling", "reps", "topology"]
    return best[cols].drop_duplicates("map").reset_index(drop=True)


def available_maps() -> list[str]:
    return sorted(p.name[: -len(config.KERNEL_SUFFIX)]
                  for p in config.KERNELS_DIR.glob(f"*{config.KERNEL_SUFFIX}"))


def load_kernel(map_name: str) -> np.ndarray:
    """Carga el kernel exacto (statevector) cacheado de 80x80 para un mapa."""
    path = config.KERNELS_DIR / f"{map_name}{config.KERNEL_SUFFIX}"
    kernel = np.load(path)
    if kernel.shape[0] != kernel.shape[1]:
        raise ValueError(f"Kernel no cuadrado para {map_name}: {kernel.shape}")
    return kernel


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)
