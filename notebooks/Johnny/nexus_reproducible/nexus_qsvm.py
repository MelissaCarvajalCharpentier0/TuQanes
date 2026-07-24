"""Reproducible Nexus evaluation utilities for the TuQanes QSVM study.

The module deliberately does not import ``qnexus`` at import time.  Local exact
validation can therefore run without Nexus credentials or quota.  Functions
whose name starts with ``nexus_`` import the client lazily and require an
explicit quota acknowledgement in :class:`ExperimentConfig`.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import itertools
import json
import math
import platform
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import pandas as pd
import pytket
import sklearn
from pytket import Circuit, OpType
from pytket.circuit import Pauli, PauliExpBox
from pytket.passes import AutoRebase, DecomposeBoxes, RemoveRedundancies, SequencePass
from pytket.qasm import circuit_to_qasm_str
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


TARGET = "Potability"
CLASS_LABELS = (0, 1)
FEATURES = ["Sulfate", "ph", "Conductivity", "Chloramines", "Hardness"]
FEATURE_SET = "chem5_v01"
EXPECTED_DATA_SHA256 = "904004bde729bfe3d2e195f46343bceead09e32a0eb95bb8184e7e20e029b2bf"
NEXUS_ACKNOWLEDGEMENT = "I_ACCEPT_NEXUS_QUOTA_USAGE"

C_GRID = (0.1, 1.0, 10.0)
GAMMA_GRID: tuple[str | float, ...] = ("scale", "auto", 0.01)
LOCKED_CLASSICAL_C = 10.0
LOCKED_CLASSICAL_GAMMA = 0.01

DOMAIN_EDGES = [
    ("ph", "Sulfate", 1.10),
    ("Conductivity", "Hardness", 1.20),
    ("Chloramines", "Sulfate", 0.90),
    ("ph", "Hardness", 0.95),
    ("Chloramines", "Conductivity", 0.80),
]


@dataclass(frozen=True)
class MapConfig:
    name: str
    family: str
    scaling: str = "robust_atan"
    reps: int = 1
    topology: str = "linear"
    alpha: float = 0.8
    local_terms: tuple[str, ...] = ("Z",)
    pair_terms: tuple[str, ...] = ("ZZ",)
    data_reupload: bool = False


LOCKED_MAPS = {
    "custom_water_domain_r1_robust": MapConfig(
        "custom_water_domain_r1_robust",
        "custom_water_domain",
        topology="domain",
        alpha=0.8,
    ),
    "zz_ring_r2_robust": MapConfig(
        "zz_ring_r2_robust",
        "ZZFeatureMap",
        reps=2,
        topology="ring",
        alpha=0.8,
    ),
    "pauli_z_zz_linear_r1_robust": MapConfig(
        "pauli_z_zz_linear_r1_robust",
        "PauliFeatureMap",
        topology="linear",
        alpha=0.8,
        local_terms=("Z",),
        pair_terms=("ZZ",),
    ),
}

LOCKED_C = {
    "custom_water_domain_r1_robust": 10.0,
    "zz_ring_r2_robust": 0.1,
    "pauli_z_zz_linear_r1_robust": 1.0,
}

MAP_TAGS = {
    "custom_water_domain_r1_robust": "custom-r1",
    "zz_ring_r2_robust": "zz-ring-r2",
    "pauli_z_zz_linear_r1_robust": "pauli-z-zz-linear-r1",
}


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration frozen into every local and Nexus run manifest."""

    subset_n: int = 16
    subset_seed: int = 20260807
    split_seed: int = 20260801
    preprocessing_protocol: str = "global_median_external_pool"
    shots: int = 4096
    repeats: int = 3
    backend_name: str = "H2-1LE"
    project_name: str = "TuQanes-QSVM-Nexus"
    run_id: str = "pilot16-strict-v1"
    max_programs_per_job: int = 300
    nexus_stage: str = "local_only"
    quota_acknowledgement: str = ""

    def validate(self) -> None:
        if self.subset_n not in {16, 64}:
            raise ValueError("subset_n debe ser 16 (piloto) o 64 (evaluacion).")
        if self.subset_n % 2:
            raise ValueError("subset_n debe mantener balance exacto entre las dos clases.")
        if self.repeats < 3:
            raise ValueError("El protocolo exige al menos tres repeticiones.")
        if self.shots <= 0:
            raise ValueError("shots debe ser positivo.")
        if self.preprocessing_protocol not in {
            "global_median_external_pool",
            "class_median_external_pool",
        }:
            raise ValueError("Protocolo de preprocesamiento no soportado.")
        if self.nexus_stage not in {"local_only", "cost", "submit", "collect"}:
            raise ValueError("nexus_stage debe ser local_only, cost, submit o collect.")
        if not 1 <= self.max_programs_per_job <= 300:
            raise ValueError("Nexus admite como maximo 300 programas por job.")


@dataclass
class PreparedExperiment:
    config: ExperimentConfig
    repo_root: Path
    output_dir: Path
    source_pool: pd.DataFrame
    subset: pd.DataFrame
    x_z: np.ndarray
    x_angles: np.ndarray
    labels: np.ndarray
    folds: np.ndarray
    source_index: np.ndarray
    preprocessing_manifest: dict[str, Any]
    input_hashes: dict[str, str]


def discover_repo_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (
            (candidate / "data" / "water_potability.csv").exists()
            and (candidate / "notebooks" / "Johnny" / "last_version").exists()
        ):
            return candidate
    raise FileNotFoundError("No se encontro la raiz de TuQanes desde la ruta actual.")


def default_paths(repo_root: Path) -> dict[str, Path]:
    current = repo_root / "notebooks" / "Johnny" / "last_version"
    return {
        "water": current / "water_potability.csv",
        "pool": current / "artifacts_v1_2" / "quantum80_raw.csv",
        "folds": current / "artifacts_v1_2" / "quantum80_folds.csv",
        "pool_manifest": current / "artifacts_v1_2" / "quantum80_manifest.json",
        "best_maps": current / "artifacts_v3_4" / "map_cv_best_by_map.csv",
        "maps_manifest": current / "artifacts_v3_4" / "parts_3_4_manifest.json",
        "classical_grid": current / "artifacts_v1_2" / "quantum80_svm_cv_summary.csv",
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(_jsonable(payload), ensure_ascii=False) + "\n")


def _priority(source_index: int, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{source_index}".encode("ascii")).hexdigest()


def _allocate_per_fold(total_per_class: int, fold_values: Sequence[int]) -> dict[int, int]:
    ordered = sorted(int(value) for value in fold_values)
    base, remainder = divmod(total_per_class, len(ordered))
    return {fold: base + int(position < remainder) for position, fold in enumerate(ordered)}


def load_source_pool(pool_path: Path, folds_path: Path) -> pd.DataFrame:
    pool = pd.read_csv(pool_path)
    fold_table = pd.read_csv(folds_path)
    required_pool = {"source_index", TARGET, *FEATURES}
    required_folds = {"position", "source_index", TARGET, "validation_fold"}
    if missing := sorted(required_pool - set(pool.columns)):
        raise ValueError(f"Columnas faltantes en quantum80_raw.csv: {missing}")
    if missing := sorted(required_folds - set(fold_table.columns)):
        raise ValueError(f"Columnas faltantes en quantum80_folds.csv: {missing}")
    if pool["source_index"].duplicated().any() or fold_table["source_index"].duplicated().any():
        raise ValueError("source_index debe ser unico en los artifacts quantum80.")

    fold_indexed = fold_table.set_index("source_index")
    pool = pool.copy()
    pool["position"] = pool["source_index"].map(fold_indexed["position"])
    pool["validation_fold"] = pool["source_index"].map(fold_indexed["validation_fold"])
    expected_y = pool["source_index"].map(fold_indexed[TARGET])
    if pool[["position", "validation_fold"]].isna().any().any():
        raise ValueError("Los artifacts raw y folds no contienen las mismas filas.")
    if not np.array_equal(pool[TARGET].to_numpy(dtype=int), expected_y.to_numpy(dtype=int)):
        raise ValueError("Las etiquetas no coinciden entre quantum80_raw y quantum80_folds.")
    if len(pool) != 80 or pool[TARGET].value_counts().sort_index().to_dict() != {0: 40, 1: 40}:
        raise ValueError("El pool fuente debe contener 80 filas balanceadas 40/40.")
    return pool.sort_values("position").reset_index(drop=True)


def select_nested_subset(pool: pd.DataFrame, subset_n: int, seed: int) -> pd.DataFrame:
    """Select a balanced, fold-aware subset without looking at model outcomes.

    A deterministic hash orders each ``(validation_fold, class)`` stratum.  The
    16-row pilot is consequently nested in the 64-row evaluation set.
    """

    if subset_n not in {16, 64}:
        raise ValueError("Solo se permiten los tamanos preregistrados 16 y 64.")
    folds = sorted(pool["validation_fold"].astype(int).unique().tolist())
    allocation = _allocate_per_fold(subset_n // 2, folds)
    selected_parts: list[pd.DataFrame] = []
    for cls in CLASS_LABELS:
        for fold in folds:
            stratum = pool.loc[
                pool[TARGET].eq(cls) & pool["validation_fold"].eq(fold)
            ].copy()
            stratum["selection_priority"] = stratum["source_index"].map(
                lambda value: _priority(int(value), seed)
            )
            count = allocation[fold]
            if len(stratum) < count:
                raise ValueError(f"Estrato fold={fold}, clase={cls} no tiene {count} filas.")
            selected_parts.append(stratum.sort_values("selection_priority").head(count))
    subset = pd.concat(selected_parts, ignore_index=True)
    subset = subset.sort_values(["validation_fold", TARGET, "position"]).reset_index(drop=True)
    if len(subset) != subset_n:
        raise AssertionError("La seleccion no produjo el tamano esperado.")
    if subset[TARGET].value_counts().sort_index().to_dict() != {
        0: subset_n // 2,
        1: subset_n // 2,
    }:
        raise AssertionError("La seleccion perdio el balance de clases.")
    fold_balance = subset.groupby(["validation_fold", TARGET]).size().unstack(fill_value=0)
    if not (fold_balance[0] == fold_balance[1]).all():
        raise AssertionError("Cada fold debe permanecer balanceado por clase.")
    return subset


def verify_locked_candidates(best_maps_path: Path, maps_manifest_path: Path) -> pd.DataFrame:
    best = pd.read_csv(best_maps_path)
    selected = best.loc[best["map"].isin(LOCKED_MAPS)].copy()
    if set(selected["map"]) != set(LOCKED_MAPS):
        raise ValueError("No se encontraron los tres candidatos congelados.")
    for row in selected.itertuples(index=False):
        expected = LOCKED_MAPS[row.map]
        if row.family != expected.family or not math.isclose(float(row.C), LOCKED_C[row.map]):
            raise ValueError(f"El artifact de seleccion cambio para {row.map}.")

    manifest = json.loads(maps_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("feature_set") != FEATURE_SET or manifest.get("features") != FEATURES:
        raise ValueError("El manifiesto de mapas no coincide con chem5_v01.")
    manifest_configs = {entry["name"]: entry for entry in manifest.get("maps_evaluated", [])}
    for name, expected in LOCKED_MAPS.items():
        actual = manifest_configs.get(name)
        if actual is None:
            raise ValueError(f"El manifiesto no contiene {name}.")
        for field in ("family", "scaling", "reps", "topology", "alpha", "data_reupload"):
            if actual.get(field) != asdict(expected)[field]:
                raise ValueError(f"Configuracion inesperada para {name}: campo {field}.")
    return selected.sort_values("f1_mean", ascending=False).reset_index(drop=True)


def verify_locked_classical(classical_grid_path: Path) -> pd.Series:
    """Verify the classical baseline selected before the Nexus subsets existed."""

    grid = pd.read_csv(classical_grid_path)
    required = {"feature_set", "features", "C", "gamma", "f1_mean"}
    if missing := sorted(required - set(grid.columns)):
        raise ValueError(f"Columnas faltantes en el artifact clasico: {missing}")
    expected_features = "|".join(FEATURES)
    grid = grid.loc[
        grid["feature_set"].eq(FEATURE_SET) & grid["features"].eq(expected_features)
    ].copy()
    locked = grid.loc[
        grid["C"].astype(float).eq(LOCKED_CLASSICAL_C)
        & grid["gamma"].astype(str).eq(str(LOCKED_CLASSICAL_GAMMA))
    ]
    if len(locked) != 1:
        raise ValueError("No se encontro una unica configuracion clasica congelada.")
    ranked = grid.sort_values(
        ["f1_mean", "balanced_accuracy_mean", "mcc_mean"], ascending=False
    ).reset_index(drop=True)
    if not (
        math.isclose(float(ranked.iloc[0]["C"]), LOCKED_CLASSICAL_C)
        and str(ranked.iloc[0]["gamma"]) == str(LOCKED_CLASSICAL_GAMMA)
    ):
        raise ValueError("El ganador clasico del artifact fuente cambio.")
    return locked.iloc[0]


def _fit_external_preprocessor(
    water_path: Path,
    source_pool: pd.DataFrame,
    subset: pd.DataFrame,
    split_seed: int,
    protocol: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    data = pd.read_csv(water_path).reset_index(names="source_index")
    required = {TARGET, *FEATURES}
    if missing := sorted(required - set(data.columns)):
        raise ValueError(f"Columnas faltantes en water_potability.csv: {missing}")

    development, holdout = train_test_split(
        data,
        test_size=0.20,
        stratify=data[TARGET],
        random_state=split_seed,
    )
    pool_ids = set(source_pool["source_index"].astype(int))
    development_ids = set(development["source_index"].astype(int))
    holdout_ids = set(holdout["source_index"].astype(int))
    if not pool_ids.issubset(development_ids) or pool_ids.intersection(holdout_ids):
        raise ValueError("El pool quantum80 debe estar exclusivamente dentro del development training.")

    fit_data = development.loc[~development["source_index"].isin(pool_ids)].copy()
    if len(fit_data) != len(development) - len(source_pool):
        raise AssertionError("No se excluyeron exactamente las 80 filas del ajuste externo.")

    global_medians = fit_data[FEATURES].median(numeric_only=True)
    class_medians = {
        int(cls): fit_data.loc[fit_data[TARGET].eq(cls), FEATURES].median(numeric_only=True)
        for cls in CLASS_LABELS
    }

    def transform(frame: pd.DataFrame, use_labels: bool) -> pd.DataFrame:
        transformed = frame[FEATURES].copy()
        if use_labels:
            for cls in CLASS_LABELS:
                mask = frame[TARGET].eq(cls)
                transformed.loc[mask] = transformed.loc[mask].fillna(class_medians[cls])
        return transformed.fillna(global_medians)

    label_aware = protocol == "class_median_external_pool"
    fit_imputed = transform(fit_data, use_labels=label_aware)
    subset_imputed = transform(subset, use_labels=label_aware)
    if fit_imputed.isna().any().any() or subset_imputed.isna().any().any():
        raise ValueError("La imputacion dejo valores faltantes.")

    scaler = StandardScaler().fit(fit_imputed)
    x_z = scaler.transform(subset_imputed)
    x_angles = (2.0 / np.pi) * np.arctan(x_z)
    manifest = {
        "protocol": protocol,
        "imputation_uses_evaluation_labels": label_aware,
        "scaler_fit_uses_evaluation_rows": False,
        "development_rows": int(len(development)),
        "excluded_quantum_pool_rows": int(len(source_pool)),
        "preprocessor_fit_rows": int(len(fit_data)),
        "holdout_rows": int(len(holdout)),
        "holdout_intersection": int(len(pool_ids.intersection(holdout_ids))),
        "features": FEATURES,
        "global_medians": global_medians.to_dict(),
        "class_medians": {str(key): value.to_dict() for key, value in class_medians.items()},
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "angle_transform": "(2/pi) * atan(z)",
    }
    return x_z, x_angles, manifest


def prepare_experiment(
    config: ExperimentConfig,
    repo_root: Path | None = None,
    output_dir: Path | None = None,
) -> PreparedExperiment:
    config.validate()
    root = (repo_root or discover_repo_root()).resolve()
    paths = default_paths(root)
    for label, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Falta el input {label}: {path}")

    hashes = {label: sha256_file(path) for label, path in paths.items()}
    if hashes["water"] != EXPECTED_DATA_SHA256:
        raise ValueError("El hash del dataset no coincide con la generacion chem5_v01.")

    verify_locked_candidates(paths["best_maps"], paths["maps_manifest"])
    verify_locked_classical(paths["classical_grid"])
    pool = load_source_pool(paths["pool"], paths["folds"])
    subset = select_nested_subset(pool, config.subset_n, config.subset_seed)
    x_z, x_angles, prep_manifest = _fit_external_preprocessor(
        paths["water"],
        pool,
        subset,
        split_seed=config.split_seed,
        protocol=config.preprocessing_protocol,
    )

    target_output = output_dir or (
        root / "notebooks" / "Johnny" / "nexus_reproducible" / "artifacts_nexus" / config.run_id
    )
    target_output.mkdir(parents=True, exist_ok=True)
    return PreparedExperiment(
        config=config,
        repo_root=root,
        output_dir=target_output.resolve(),
        source_pool=pool,
        subset=subset,
        x_z=x_z,
        x_angles=x_angles,
        labels=subset[TARGET].to_numpy(dtype=int),
        folds=subset["validation_fold"].to_numpy(dtype=int),
        source_index=subset["source_index"].to_numpy(dtype=int),
        preprocessing_manifest=prep_manifest,
        input_hashes=hashes,
    )


PAULI_LOOKUP = {"X": Pauli.X, "Y": Pauli.Y, "Z": Pauli.Z}


def topology_pairs(n_qubits: int, topology: str) -> list[tuple[int, int]]:
    if topology == "linear":
        return [(i, i + 1) for i in range(n_qubits - 1)]
    if topology == "ring":
        return [(i, i + 1) for i in range(n_qubits - 1)] + [(n_qubits - 1, 0)]
    if topology == "full":
        return list(itertools.combinations(range(n_qubits), 2))
    if topology == "domain":
        feature_to_position = {feature: position for position, feature in enumerate(FEATURES)}
        return [(feature_to_position[a], feature_to_position[b]) for a, b, _ in DOMAIN_EDGES]
    raise ValueError(f"Topologia no soportada: {topology}")


def domain_weight(i: int, j: int) -> float:
    feature_to_position = {feature: position for position, feature in enumerate(FEATURES)}
    for left, right, weight in DOMAIN_EDGES:
        if {i, j} == {feature_to_position[left], feature_to_position[right]}:
            return float(weight)
    return 1.0


def add_pauli_exp(
    circuit: Circuit,
    pauli_string: str,
    qubits: list[int],
    angle_halfturns: float,
) -> None:
    circuit.add_pauliexpbox(
        PauliExpBox([PAULI_LOOKUP[pauli] for pauli in pauli_string], float(angle_halfturns)),
        qubits,
    )


def build_feature_circuit(x_angles: np.ndarray, config: MapConfig) -> Circuit:
    x = np.asarray(x_angles, dtype=float)
    circuit = Circuit(len(x), name=f"feature-{MAP_TAGS[config.name]}")
    for qubit in range(len(x)):
        circuit.H(qubit)
    for repetition in range(config.reps):
        rep_scale = config.alpha / np.sqrt(repetition + 1)
        if config.family == "ZZFeatureMap":
            for i in range(len(x)):
                circuit.Rz(rep_scale * x[i], i)
            for i, j in topology_pairs(len(x), config.topology):
                circuit.ZZPhase(rep_scale * x[i] * x[j], i, j)
        elif config.family == "PauliFeatureMap":
            for term in config.local_terms:
                for i in range(len(x)):
                    add_pauli_exp(circuit, term, [i], rep_scale * x[i])
            for term in config.pair_terms:
                if len(term) != 2:
                    raise ValueError("Solo se admiten terminos Pauli de pares de longitud 2.")
                for i, j in topology_pairs(len(x), config.topology):
                    add_pauli_exp(circuit, term, [i, j], rep_scale * x[i] * x[j])
        elif config.family == "custom_water_domain":
            for i in range(len(x)):
                circuit.Ry(0.75 * rep_scale * x[i], i)
                circuit.Rz(0.35 * rep_scale * np.sign(x[i]) * x[i] ** 2, i)
            for i, j in topology_pairs(len(x), "domain"):
                circuit.ZZPhase(rep_scale * domain_weight(i, j) * x[i] * x[j], i, j)
            if config.data_reupload:
                for i in range(len(x)):
                    circuit.Rx(0.25 * rep_scale * x[i], i)
        else:
            raise ValueError(f"Familia no soportada: {config.family}")
    return circuit


def build_overlap_circuit(
    x_i: np.ndarray,
    x_j: np.ndarray,
    config: MapConfig,
    name: str,
    measure: bool = True,
) -> Circuit:
    """Construct ``U(x_j)^dagger U(x_i)|0>`` for an all-zero overlap test."""

    circuit = Circuit(len(x_i), name=name)
    circuit.append(build_feature_circuit(x_i, config))
    circuit.append(build_feature_circuit(x_j, config).dagger())
    if measure:
        circuit.measure_all()
    return circuit


def exact_fidelity_kernel(x_angles: np.ndarray, config: MapConfig) -> np.ndarray:
    states = np.asarray(
        [build_feature_circuit(row, config).get_statevector() for row in np.asarray(x_angles)]
    )
    overlaps = states @ states.conj().T
    kernel = np.clip(np.abs(overlaps) ** 2, 0.0, 1.0).real
    kernel = 0.5 * (kernel + kernel.T)
    np.fill_diagonal(kernel, 1.0)
    return kernel


def validate_overlap_circuits(
    x_angles: np.ndarray,
    config: MapConfig,
    kernel: np.ndarray,
    max_pairs: int = 12,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pairs = list(itertools.combinations(range(len(x_angles)), 2))[:max_pairs]
    for i, j in pairs:
        overlap = build_overlap_circuit(
            x_angles[i], x_angles[j], config, name=f"validation-{i}-{j}", measure=False
        )
        state = overlap.get_statevector()
        measured = float(np.abs(state[0]) ** 2)
        rows.append(
            {
                "map": config.name,
                "i": i,
                "j": j,
                "statevector_kernel": float(kernel[i, j]),
                "overlap_zero_probability": measured,
                "absolute_error": abs(measured - float(kernel[i, j])),
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty and result["absolute_error"].max() > 1e-10:
        raise AssertionError("El circuito de overlap no reproduce el kernel exacto.")
    return result


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
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


METRIC_NAMES = (
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "specificity",
    "f1",
    "mcc",
)


def summarize_folds(fold_metrics: pd.DataFrame) -> dict[str, float]:
    summary: dict[str, float] = {}
    for metric in METRIC_NAMES:
        summary[f"{metric}_mean"] = float(fold_metrics[metric].mean())
        summary[f"{metric}_std"] = float(fold_metrics[metric].std(ddof=1))
    return summary


def evaluate_precomputed_kernel(
    kernel: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    source_index: np.ndarray,
    c_value: float,
    map_name: str,
    repeat: int | None = None,
    result_kind: str = "exact",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fold_rows: list[dict[str, Any]] = []
    prediction_parts: list[pd.DataFrame] = []
    for fold in sorted(np.unique(folds)):
        train_idx = np.flatnonzero(folds != fold)
        valid_idx = np.flatnonzero(folds == fold)
        model = SVC(kernel="precomputed", C=float(c_value))
        model.fit(kernel[np.ix_(train_idx, train_idx)], labels[train_idx])
        prediction = model.predict(kernel[np.ix_(valid_idx, train_idx)])
        row: dict[str, Any] = {
            "map": map_name,
            "result_kind": result_kind,
            "repeat": repeat,
            "C": float(c_value),
            "fold": int(fold),
            "n_train": int(len(train_idx)),
            "n_valid": int(len(valid_idx)),
        }
        row.update(classification_metrics(labels[valid_idx], prediction))
        fold_rows.append(row)
        prediction_parts.append(
            pd.DataFrame(
                {
                    "map": map_name,
                    "result_kind": result_kind,
                    "repeat": repeat,
                    "C": float(c_value),
                    "fold": int(fold),
                    "source_index": source_index[valid_idx],
                    "y_true": labels[valid_idx],
                    "y_pred": prediction,
                }
            )
        )
    fold_df = pd.DataFrame(fold_rows)
    summary = {
        "map": map_name,
        "result_kind": result_kind,
        "repeat": repeat,
        "C": float(c_value),
        **summarize_folds(fold_df),
    }
    return fold_df, pd.concat(prediction_parts, ignore_index=True), pd.DataFrame([summary])


def evaluate_classical_grid(
    x_z: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    source_index: np.ndarray,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
]:
    fold_rows: list[dict[str, Any]] = []
    prediction_parts: list[pd.DataFrame] = []
    for c_value, gamma in itertools.product(C_GRID, GAMMA_GRID):
        for fold in sorted(np.unique(folds)):
            train_idx = np.flatnonzero(folds != fold)
            valid_idx = np.flatnonzero(folds == fold)
            model = SVC(kernel="rbf", C=c_value, gamma=gamma)
            model.fit(x_z[train_idx], labels[train_idx])
            prediction = model.predict(x_z[valid_idx])
            row: dict[str, Any] = {
                "model": "svm_rbf",
                "C": float(c_value),
                "gamma": gamma,
                "fold": int(fold),
                "n_train": int(len(train_idx)),
                "n_valid": int(len(valid_idx)),
            }
            row.update(classification_metrics(labels[valid_idx], prediction))
            fold_rows.append(row)
            prediction_parts.append(
                pd.DataFrame(
                    {
                        "model": "svm_rbf",
                        "C": float(c_value),
                        "gamma": gamma,
                        "fold": int(fold),
                        "source_index": source_index[valid_idx],
                        "y_true": labels[valid_idx],
                        "y_pred": prediction,
                    }
                )
            )
    folds_df = pd.DataFrame(fold_rows)
    summary_rows: list[dict[str, Any]] = []
    for (c_value, gamma), group in folds_df.groupby(["C", "gamma"], sort=False):
        summary_rows.append(
            {"model": "svm_rbf", "C": float(c_value), "gamma": gamma, **summarize_folds(group)}
        )
    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["f1_mean", "balanced_accuracy_mean", "mcc_mean"], ascending=False
    ).reset_index(drop=True)
    best = summary_df.iloc[0]
    mask = folds_df["C"].eq(float(best["C"])) & folds_df["gamma"].astype(str).eq(str(best["gamma"]))
    best_folds = folds_df.loc[mask].copy()
    best_predictions = pd.concat(
        [
            part
            for part in prediction_parts
            if float(part["C"].iloc[0]) == float(best["C"])
            and str(part["gamma"].iloc[0]) == str(best["gamma"])
        ],
        ignore_index=True,
    )
    locked = summary_df.loc[
        summary_df["C"].eq(LOCKED_CLASSICAL_C)
        & summary_df["gamma"].astype(str).eq(str(LOCKED_CLASSICAL_GAMMA))
    ].iloc[0]
    locked_mask = folds_df["C"].eq(LOCKED_CLASSICAL_C) & folds_df["gamma"].astype(str).eq(
        str(LOCKED_CLASSICAL_GAMMA)
    )
    locked_folds = folds_df.loc[locked_mask].copy()
    locked_predictions = pd.concat(
        [
            part
            for part in prediction_parts
            if float(part["C"].iloc[0]) == LOCKED_CLASSICAL_C
            and str(part["gamma"].iloc[0]) == str(LOCKED_CLASSICAL_GAMMA)
        ],
        ignore_index=True,
    )
    return (
        best_folds,
        best_predictions,
        summary_df,
        best,
        locked_folds,
        locked_predictions,
        locked,
    )


def kernel_diagnostics(kernel: np.ndarray) -> dict[str, float]:
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


def simulate_binomial_repeats(
    exact_kernel: np.ndarray,
    experiment: PreparedExperiment,
    map_name: str,
) -> tuple[list[np.ndarray], pd.DataFrame, pd.DataFrame]:
    """Local shot-only preview.  These are never labelled as Nexus results."""

    kernels: list[np.ndarray] = []
    summary_parts: list[pd.DataFrame] = []
    fold_parts: list[pd.DataFrame] = []
    for repeat in range(experiment.config.repeats):
        map_offset = int(hashlib.sha256(map_name.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(experiment.config.subset_seed + map_offset + repeat)
        # Nexus executes one circuit per unique pair, so the preview samples each
        # upper-triangle probability exactly once and mirrors the observed value.
        sampled = np.eye(len(exact_kernel), dtype=float)
        for i, j in itertools.combinations(range(len(exact_kernel)), 2):
            probability = float(np.clip(exact_kernel[i, j], 0.0, 1.0))
            observed = rng.binomial(experiment.config.shots, probability) / experiment.config.shots
            sampled[i, j] = observed
            sampled[j, i] = observed
        kernels.append(sampled)
        folds_df, _, summary_df = evaluate_precomputed_kernel(
            sampled,
            experiment.labels,
            experiment.folds,
            experiment.source_index,
            LOCKED_C[map_name],
            map_name,
            repeat=repeat,
            result_kind="local_binomial_preview",
        )
        summary_df["kernel_mae_vs_exact"] = float(np.mean(np.abs(sampled - exact_kernel)))
        fold_parts.append(folds_df)
        summary_parts.append(summary_df)
    return kernels, pd.concat(fold_parts, ignore_index=True), pd.concat(summary_parts, ignore_index=True)


def pair_manifest(experiment: PreparedExperiment, kernels: dict[str, np.ndarray]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for map_name in LOCKED_MAPS:
        tag = MAP_TAGS[map_name]
        for pair_index, (i, j) in enumerate(itertools.combinations(range(len(experiment.subset)), 2)):
            rows.append(
                {
                    "map": map_name,
                    "map_tag": tag,
                    "pair_index": pair_index,
                    "i": i,
                    "j": j,
                    "source_index_i": int(experiment.source_index[i]),
                    "source_index_j": int(experiment.source_index[j]),
                    "circuit_name": f"tq-{experiment.config.run_id}-{tag}-p{pair_index:04d}",
                    "exact_fidelity": float(kernels[map_name][i, j]),
                    "shots": experiment.config.shots,
                }
            )
    return pd.DataFrame(rows)


def chunk_records(frame: pd.DataFrame, chunk_size: int) -> Iterator[pd.DataFrame]:
    for start in range(0, len(frame), chunk_size):
        yield frame.iloc[start : start + chunk_size].copy()


def expected_workload(experiment: PreparedExperiment) -> pd.DataFrame:
    pairs = experiment.config.subset_n * (experiment.config.subset_n - 1) // 2
    chunks = math.ceil(pairs / experiment.config.max_programs_per_job)
    rows = []
    for map_name in LOCKED_MAPS:
        rows.append(
            {
                "map": map_name,
                "samples": experiment.config.subset_n,
                "unique_offdiagonal_pairs": pairs,
                "compile_jobs": chunks,
                "execute_jobs_per_repeat": chunks,
                "repeats": experiment.config.repeats,
                "total_execute_jobs": chunks * experiment.config.repeats,
                "shots_per_circuit": experiment.config.shots,
                "total_requested_shots": pairs
                * experiment.config.repeats
                * experiment.config.shots,
            }
        )
    return pd.DataFrame(rows)


def portable_overlap_qasm(
    x_i: np.ndarray,
    x_j: np.ndarray,
    config: MapConfig,
    name: str,
) -> str:
    circuit = build_overlap_circuit(x_i, x_j, config, name=name, measure=True)
    portable = circuit.copy()
    SequencePass(
        [
            DecomposeBoxes(),
            AutoRebase({OpType.H, OpType.Rx, OpType.Rz, OpType.CX}),
            RemoveRedundancies(),
        ]
    ).apply(portable)
    return circuit_to_qasm_str(portable)


def save_local_run(
    experiment: PreparedExperiment,
    candidate_selection: pd.DataFrame,
    exact_kernels: dict[str, np.ndarray],
    overlap_validation: pd.DataFrame,
    classical_summary: pd.DataFrame,
    classical_best_folds: pd.DataFrame,
    classical_best_predictions: pd.DataFrame,
    classical_locked_folds: pd.DataFrame,
    classical_locked_predictions: pd.DataFrame,
    qsvm_summary: pd.DataFrame,
    qsvm_folds: pd.DataFrame,
    qsvm_predictions: pd.DataFrame,
    preview_summary: pd.DataFrame,
    preview_folds: pd.DataFrame,
) -> dict[str, Path]:
    output = experiment.output_dir
    local_dir = output / "local_exact"
    circuit_dir = output / "representative_circuits"
    local_dir.mkdir(parents=True, exist_ok=True)
    circuit_dir.mkdir(parents=True, exist_ok=True)

    subset_columns = [
        "position",
        "source_index",
        TARGET,
        "validation_fold",
        "selection_priority",
        *FEATURES,
    ]
    experiment.subset[subset_columns].to_csv(output / "selected_subset.csv", index=False)
    candidate_selection.to_csv(output / "candidate_selection.csv", index=False)
    overlap_validation.to_csv(local_dir / "overlap_circuit_validation.csv", index=False)
    classical_summary.to_csv(local_dir / "classical_grid_summary.csv", index=False)
    classical_best_folds.to_csv(local_dir / "classical_best_fold_metrics.csv", index=False)
    classical_best_predictions.to_csv(local_dir / "classical_best_oof_predictions.csv", index=False)
    classical_locked_folds.to_csv(local_dir / "classical_locked_fold_metrics.csv", index=False)
    classical_locked_predictions.to_csv(
        local_dir / "classical_locked_oof_predictions.csv", index=False
    )
    qsvm_summary.to_csv(local_dir / "qsvm_exact_summary.csv", index=False)
    qsvm_folds.to_csv(local_dir / "qsvm_exact_fold_metrics.csv", index=False)
    qsvm_predictions.to_csv(local_dir / "qsvm_exact_oof_predictions.csv", index=False)
    preview_summary.to_csv(local_dir / "shot_preview_summary.csv", index=False)
    preview_folds.to_csv(local_dir / "shot_preview_fold_metrics.csv", index=False)

    diagnostic_rows = []
    for map_name, kernel in exact_kernels.items():
        np.save(local_dir / f"{map_name}_exact_kernel.npy", kernel)
        pd.DataFrame(kernel).to_csv(local_dir / f"{map_name}_exact_kernel.csv", index=False)
        diagnostic_rows.append({"map": map_name, **kernel_diagnostics(kernel)})
        (circuit_dir / f"{map_name}_representative_overlap.qasm").write_text(
            portable_overlap_qasm(
                experiment.x_angles[0],
                experiment.x_angles[1],
                LOCKED_MAPS[map_name],
                name=f"representative-{MAP_TAGS[map_name]}",
            ),
            encoding="utf-8",
        )
    pd.DataFrame(diagnostic_rows).to_csv(local_dir / "kernel_diagnostics.csv", index=False)

    manifest = {
        "config": asdict(experiment.config),
        "features": FEATURES,
        "feature_set": FEATURE_SET,
        "locked_maps": {name: asdict(config) for name, config in LOCKED_MAPS.items()},
        "locked_C": LOCKED_C,
        "locked_classical": {
            "C": LOCKED_CLASSICAL_C,
            "gamma": LOCKED_CLASSICAL_GAMMA,
            "source": "artifacts_v1_2/quantum80_svm_cv_summary.csv",
        },
        "input_hashes": experiment.input_hashes,
        "preprocessing": experiment.preprocessing_manifest,
        "subset": {
            "rows": int(len(experiment.subset)),
            "class_counts": experiment.subset[TARGET].value_counts().sort_index().to_dict(),
            "fold_class_counts": experiment.subset.groupby(["validation_fold", TARGET])
            .size()
            .to_dict(),
            "source_indices": experiment.source_index.tolist(),
        },
        "environment": environment_versions(),
        "result_labels": {
            "exact": "statevector local sin shots ni ruido",
            "local_binomial_preview": "muestreo binomial local; no es Nexus ni H2",
            "nexus": "reservado para counts descargados de Nexus",
        },
    }
    write_json(output / "run_manifest.json", manifest)
    return {
        "output_dir": output,
        "manifest": output / "run_manifest.json",
        "exact_summary": local_dir / "qsvm_exact_summary.csv",
        "classical_summary": local_dir / "classical_grid_summary.csv",
    }


def environment_versions() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "pytket": pytket.__version__,
    }
    try:
        versions["qnexus"] = importlib.metadata.version("qnexus")
    except importlib.metadata.PackageNotFoundError:
        versions["qnexus"] = "not-installed-local-validation"
    return versions


def run_local_analysis(experiment: PreparedExperiment) -> dict[str, Any]:
    """Run and persist every quota-free validation step."""

    paths = default_paths(experiment.repo_root)
    candidates = verify_locked_candidates(paths["best_maps"], paths["maps_manifest"])
    (
        best_classical_folds,
        best_classical_predictions,
        classical_summary,
        classical_best,
        locked_classical_folds,
        locked_classical_predictions,
        classical_locked,
    ) = evaluate_classical_grid(
        experiment.x_z,
        experiment.labels,
        experiment.folds,
        experiment.source_index,
    )

    exact_kernels: dict[str, np.ndarray] = {}
    overlap_parts: list[pd.DataFrame] = []
    qsvm_summary_parts: list[pd.DataFrame] = []
    qsvm_fold_parts: list[pd.DataFrame] = []
    qsvm_prediction_parts: list[pd.DataFrame] = []
    preview_summary_parts: list[pd.DataFrame] = []
    preview_fold_parts: list[pd.DataFrame] = []

    for map_name, map_config in LOCKED_MAPS.items():
        kernel = exact_fidelity_kernel(experiment.x_angles, map_config)
        exact_kernels[map_name] = kernel
        overlap_parts.append(validate_overlap_circuits(experiment.x_angles, map_config, kernel))
        fold_df, prediction_df, summary_df = evaluate_precomputed_kernel(
            kernel,
            experiment.labels,
            experiment.folds,
            experiment.source_index,
            LOCKED_C[map_name],
            map_name,
            result_kind="exact",
        )
        qsvm_fold_parts.append(fold_df)
        qsvm_prediction_parts.append(prediction_df)
        qsvm_summary_parts.append(summary_df)
        _, preview_folds, preview_summary = simulate_binomial_repeats(kernel, experiment, map_name)
        preview_fold_parts.append(preview_folds)
        preview_summary_parts.append(preview_summary)

    overlap_validation = pd.concat(overlap_parts, ignore_index=True)
    qsvm_summary = pd.concat(qsvm_summary_parts, ignore_index=True)
    qsvm_folds = pd.concat(qsvm_fold_parts, ignore_index=True)
    qsvm_predictions = pd.concat(qsvm_prediction_parts, ignore_index=True)
    preview_summary = pd.concat(preview_summary_parts, ignore_index=True)
    preview_folds = pd.concat(preview_fold_parts, ignore_index=True)
    pair_table = pair_manifest(experiment, exact_kernels)
    pair_table.to_csv(experiment.output_dir / "circuit_pair_manifest.csv", index=False)
    workload = expected_workload(experiment)
    workload.to_csv(experiment.output_dir / "expected_workload.csv", index=False)

    saved = save_local_run(
        experiment,
        candidates,
        exact_kernels,
        overlap_validation,
        classical_summary,
        best_classical_folds,
        best_classical_predictions,
        locked_classical_folds,
        locked_classical_predictions,
        qsvm_summary,
        qsvm_folds,
        qsvm_predictions,
        preview_summary,
        preview_folds,
    )
    return {
        "candidate_selection": candidates,
        "classical_summary": classical_summary,
        "classical_best": classical_best,
        "classical_grid_best": classical_best,
        "classical_locked": classical_locked,
        "qsvm_exact_summary": qsvm_summary,
        "qsvm_exact_folds": qsvm_folds,
        "shot_preview_summary": preview_summary,
        "overlap_validation": overlap_validation,
        "exact_kernels": exact_kernels,
        "pair_manifest": pair_table,
        "workload": workload,
        "saved": saved,
    }


def _require_nexus(config: ExperimentConfig, allowed_stages: set[str]):
    config.validate()
    if config.nexus_stage not in allowed_stages:
        raise RuntimeError(
            f"Esta operacion requiere nexus_stage en {sorted(allowed_stages)}; "
            f"valor actual: {config.nexus_stage}."
        )
    if config.quota_acknowledgement != NEXUS_ACKNOWLEDGEMENT:
        raise RuntimeError(
            "Falta el reconocimiento explicito de cuota. Revise costos y establezca "
            f"quota_acknowledgement='{NEXUS_ACKNOWLEDGEMENT}'."
        )
    try:
        import qnexus as qnx
    except ImportError as exc:
        raise RuntimeError("qnexus no esta instalado en este entorno.") from exc
    return qnx


def _save_ref(qnx: Any, path: Path, ref: Any) -> None:
    if not path.exists():
        qnx.filesystem.save(path=path, ref=ref, mkdir=True)


def _load_ref(qnx: Any, path: Path) -> Any:
    return qnx.filesystem.load(path=path)


def nexus_context(config: ExperimentConfig, allowed_stages: set[str]) -> tuple[Any, Any, Any]:
    """Return ``(qnexus, project_ref, backend_config)`` without performing login."""

    qnx = _require_nexus(config, allowed_stages)
    project = qnx.projects.get_or_create(name=config.project_name)
    qnx.context.set_active_project(project)
    backend_kwargs: dict[str, Any] = {
        "device_name": config.backend_name,
        "attempt_batching": False,
    }
    if config.backend_name.endswith("-Emulator"):
        backend_kwargs["noisy_simulation"] = True
    backend = qnx.QuantinuumConfig(**backend_kwargs)
    return qnx, project, backend


def nexus_upload_map_circuits(
    experiment: PreparedExperiment,
    map_name: str,
    pair_table: pd.DataFrame,
) -> list[Any]:
    """Upload one map's overlap circuits with local reference checkpoints."""

    qnx, project, _ = nexus_context(experiment.config, {"cost", "submit"})
    map_rows = pair_table.loc[pair_table["map"].eq(map_name)].sort_values("pair_index")
    ref_root = experiment.output_dir / "nexus" / "refs" / "uploaded" / MAP_TAGS[map_name]
    registry = experiment.output_dir / "nexus" / "upload_registry.jsonl"
    references: list[Any] = []
    for row in map_rows.itertuples(index=False):
        ref_path = ref_root / f"p{int(row.pair_index):04d}"
        if ref_path.exists():
            ref = _load_ref(qnx, ref_path)
        else:
            circuit = build_overlap_circuit(
                experiment.x_angles[int(row.i)],
                experiment.x_angles[int(row.j)],
                LOCKED_MAPS[map_name],
                name=row.circuit_name,
                measure=True,
            )
            ref = qnx.circuits.upload(circuit=circuit, name=row.circuit_name, project=project)
            _save_ref(qnx, ref_path, ref)
            append_jsonl(
                registry,
                {
                    "map": map_name,
                    "pair_index": int(row.pair_index),
                    "circuit_name": row.circuit_name,
                    "ref_id": str(getattr(ref, "id", "")),
                },
            )
        references.append(ref)
    return references


def nexus_compile_map_circuits(
    experiment: PreparedExperiment,
    map_name: str,
    uploaded_refs: Sequence[Any],
) -> list[Any]:
    """Compile uploaded circuits once and checkpoint every output reference."""

    qnx, project, backend = nexus_context(experiment.config, {"cost", "submit"})
    compiled: list[Any] = []
    tag = MAP_TAGS[map_name]
    ref_root = experiment.output_dir / "nexus" / "refs"
    registry = experiment.output_dir / "nexus" / "compile_registry.jsonl"
    for chunk_index, start in enumerate(
        range(0, len(uploaded_refs), experiment.config.max_programs_per_job)
    ):
        chunk = list(uploaded_refs[start : start + experiment.config.max_programs_per_job])
        output_paths = [
            ref_root / "compiled" / tag / f"p{pair_index:04d}"
            for pair_index in range(start, start + len(chunk))
        ]
        if all(path.exists() for path in output_paths):
            compiled.extend(_load_ref(qnx, path) for path in output_paths)
            continue
        job_path = ref_root / "compile_jobs" / tag / f"chunk-{chunk_index:03d}"
        if job_path.exists():
            job_ref = _load_ref(qnx, job_path)
        else:
            job_name = f"tq-{experiment.config.run_id}-{tag}-compile-{chunk_index:03d}"
            job_ref = qnx.start_compile_job(
                programs=chunk,
                backend_config=backend,
                optimisation_level=2,
                name=job_name,
                project=project,
            )
            _save_ref(qnx, job_path, job_ref)
            append_jsonl(
                registry,
                {
                    "map": map_name,
                    "chunk": chunk_index,
                    "job_name": job_name,
                    "job_id": str(getattr(job_ref, "id", "")),
                    "programs": len(chunk),
                },
            )
        qnx.jobs.wait_for(job_ref)
        outputs = [item.get_output() for item in qnx.jobs.results(job_ref)]
        if len(outputs) != len(chunk):
            raise RuntimeError("El compile job no devolvio un output por circuito.")
        for path, output_ref in zip(output_paths, outputs):
            _save_ref(qnx, path, output_ref)
            compiled.append(output_ref)
    return compiled


def nexus_estimate_map_cost(
    experiment: PreparedExperiment,
    map_name: str,
    compiled_refs: Sequence[Any],
) -> pd.DataFrame:
    """Run explicit Nexus costing jobs; this operation consumes costing quota."""

    qnx, project, backend = nexus_context(experiment.config, {"cost"})
    rows: list[dict[str, Any]] = []
    for chunk_index, start in enumerate(
        range(0, len(compiled_refs), experiment.config.max_programs_per_job)
    ):
        chunk = list(compiled_refs[start : start + experiment.config.max_programs_per_job])
        estimate = qnx.circuits.cost(
            circuit_ref=chunk,
            n_shots=[experiment.config.shots] * len(chunk),
            backend_config=backend,
            project=project,
        )
        rows.append(
            {
                "map": map_name,
                "chunk": chunk_index,
                "programs": len(chunk),
                "shots_per_program": experiment.config.shots,
                "estimated_cost": estimate,
                "backend_name": experiment.config.backend_name,
            }
        )
    frame = pd.DataFrame(rows)
    output_path = experiment.output_dir / "nexus" / "cost_estimates.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        existing = pd.read_csv(output_path)
        frame = pd.concat([existing.loc[~existing["map"].eq(map_name)], frame], ignore_index=True)
    frame.to_csv(output_path, index=False)
    return frame.loc[frame["map"].eq(map_name)].copy()


def nexus_submit_map_repeats(
    experiment: PreparedExperiment,
    map_name: str,
    compiled_refs: Sequence[Any],
) -> list[Any]:
    """Submit all preregistered repeats, reusing the same compiled circuits."""

    qnx, project, backend = nexus_context(experiment.config, {"submit"})
    tag = MAP_TAGS[map_name]
    ref_root = experiment.output_dir / "nexus" / "refs" / "execute_jobs" / tag
    registry = experiment.output_dir / "nexus" / "execute_registry.jsonl"
    jobs: list[Any] = []
    for repeat in range(experiment.config.repeats):
        for chunk_index, start in enumerate(
            range(0, len(compiled_refs), experiment.config.max_programs_per_job)
        ):
            chunk = list(compiled_refs[start : start + experiment.config.max_programs_per_job])
            job_path = ref_root / f"repeat-{repeat:02d}-chunk-{chunk_index:03d}"
            if job_path.exists():
                job_ref = _load_ref(qnx, job_path)
            else:
                job_name = (
                    f"tq-{experiment.config.run_id}-{tag}-"
                    f"r{repeat:02d}-execute-{chunk_index:03d}"
                )
                job_ref = qnx.start_execute_job(
                    programs=chunk,
                    n_shots=[experiment.config.shots] * len(chunk),
                    backend_config=backend,
                    name=job_name,
                    project=project,
                )
                _save_ref(qnx, job_path, job_ref)
                append_jsonl(
                    registry,
                    {
                        "map": map_name,
                        "repeat": repeat,
                        "chunk": chunk_index,
                        "job_name": job_name,
                        "job_id": str(getattr(job_ref, "id", "")),
                        "programs": len(chunk),
                        "shots_per_program": experiment.config.shots,
                    },
                )
            jobs.append(job_ref)
    return jobs


def _outcome_bitstring(outcome: Any) -> str:
    if hasattr(outcome, "to_readouts"):
        values = np.asarray(outcome.to_readouts()).reshape(-1)
        return "".join(str(int(value)) for value in values)
    if isinstance(outcome, tuple):
        return "".join(str(int(value)) for value in outcome)
    text = str(outcome).replace(" ", "").replace(",", "")
    digits = "".join(character for character in text if character in "01")
    if digits:
        return digits
    raise TypeError(f"No se pudo serializar el outcome: {outcome!r}")


def backend_result_counts(result: Any) -> dict[str, int]:
    counts = result.get_counts()
    serialized: Counter[str] = Counter()
    for outcome, count in counts.items():
        serialized[_outcome_bitstring(outcome)] += int(count)
    return dict(sorted(serialized.items()))


def nexus_collect_map_results(
    experiment: PreparedExperiment,
    map_name: str,
    execute_jobs: Sequence[Any],
    pair_table: pd.DataFrame,
) -> pd.DataFrame:
    """Download all counts and preserve them before any kernel correction."""

    qnx, _, _ = nexus_context(experiment.config, {"collect"})
    map_pairs = pair_table.loc[pair_table["map"].eq(map_name)].sort_values("pair_index")
    chunk_size = experiment.config.max_programs_per_job
    expected_jobs = experiment.config.repeats * math.ceil(len(map_pairs) / chunk_size)
    if len(execute_jobs) != expected_jobs:
        raise ValueError(f"Se esperaban {expected_jobs} execute jobs para {map_name}.")

    rows: list[dict[str, Any]] = []
    job_position = 0
    for repeat in range(experiment.config.repeats):
        for chunk_index, pair_chunk in enumerate(chunk_records(map_pairs, chunk_size)):
            job_ref = execute_jobs[job_position]
            job_position += 1
            qnx.jobs.wait_for(job_ref)
            result_refs = qnx.jobs.results(job_ref, allow_incomplete=False)
            if len(result_refs) != len(pair_chunk):
                raise RuntimeError("El execute job no devolvio un resultado por circuito.")
            for pair_row, result_ref in zip(pair_chunk.itertuples(index=False), result_refs):
                result = result_ref.download_result()
                counts = backend_result_counts(result)
                observed_shots = int(sum(counts.values()))
                zero_key = "0" * len(FEATURES)
                zero_count = int(counts.get(zero_key, 0))
                rows.append(
                    {
                        "map": map_name,
                        "repeat": repeat,
                        "chunk": chunk_index,
                        "pair_index": int(pair_row.pair_index),
                        "i": int(pair_row.i),
                        "j": int(pair_row.j),
                        "source_index_i": int(pair_row.source_index_i),
                        "source_index_j": int(pair_row.source_index_j),
                        "job_id": str(getattr(job_ref, "id", "")),
                        "result_ref_id": str(getattr(result_ref, "id", "")),
                        "requested_shots": experiment.config.shots,
                        "observed_shots": observed_shots,
                        "zero_count": zero_count,
                        "zero_probability": zero_count / observed_shots,
                        "counts": json.dumps(counts, sort_keys=True),
                    }
                )
    frame = pd.DataFrame(rows)
    raw_path = experiment.output_dir / "nexus" / "raw_counts" / f"{map_name}.csv"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(raw_path, index=False)
    return frame


def nexus_load_execute_jobs(
    experiment: PreparedExperiment,
    map_name: str,
) -> list[Any]:
    """Load checkpointed execute-job references for a later collection session."""

    qnx, _, _ = nexus_context(experiment.config, {"collect"})
    tag = MAP_TAGS[map_name]
    ref_root = experiment.output_dir / "nexus" / "refs" / "execute_jobs" / tag
    pairs = experiment.config.subset_n * (experiment.config.subset_n - 1) // 2
    chunks = math.ceil(pairs / experiment.config.max_programs_per_job)
    jobs: list[Any] = []
    for repeat in range(experiment.config.repeats):
        for chunk_index in range(chunks):
            job_path = ref_root / f"repeat-{repeat:02d}-chunk-{chunk_index:03d}"
            if not job_path.exists():
                raise FileNotFoundError(f"Falta el checkpoint del execute job: {job_path}")
            jobs.append(_load_ref(qnx, job_path))
    return jobs


def reconstruct_measured_kernels(
    experiment: PreparedExperiment,
    map_name: str,
    raw_counts: pd.DataFrame,
) -> tuple[dict[int, np.ndarray], pd.DataFrame]:
    """Reconstruct one raw symmetric kernel per repeat; never project to PSD."""

    kernels: dict[int, np.ndarray] = {}
    summaries: list[pd.DataFrame] = []
    output_dir = experiment.output_dir / "nexus" / "kernels"
    output_dir.mkdir(parents=True, exist_ok=True)
    for repeat, group in raw_counts.groupby("repeat"):
        kernel = np.eye(experiment.config.subset_n, dtype=float)
        for row in group.itertuples(index=False):
            kernel[int(row.i), int(row.j)] = float(row.zero_probability)
            kernel[int(row.j), int(row.i)] = float(row.zero_probability)
        kernels[int(repeat)] = kernel
        np.save(output_dir / f"{map_name}_repeat{int(repeat):02d}_raw.npy", kernel)
        pd.DataFrame(kernel).to_csv(
            output_dir / f"{map_name}_repeat{int(repeat):02d}_raw.csv", index=False
        )
        _, _, summary = evaluate_precomputed_kernel(
            kernel,
            experiment.labels,
            experiment.folds,
            experiment.source_index,
            LOCKED_C[map_name],
            map_name,
            repeat=int(repeat),
            result_kind="nexus_raw",
        )
        for key, value in kernel_diagnostics(kernel).items():
            summary[key] = value
        summaries.append(summary)
    summary_frame = pd.concat(summaries, ignore_index=True)
    summary_frame.to_csv(
        experiment.output_dir / "nexus" / f"{map_name}_repeat_metrics.csv", index=False
    )
    return kernels, summary_frame


def aggregate_nexus_repeats(repeat_metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate every repeat; no best-run filtering is performed."""

    rows: list[dict[str, Any]] = []
    for map_name, group in repeat_metrics.groupby("map"):
        row: dict[str, Any] = {
            "map": map_name,
            "repeats": int(group["repeat"].nunique()),
            "C": float(group["C"].iloc[0]),
        }
        for metric in ("f1_mean", "balanced_accuracy_mean", "accuracy_mean", "mcc_mean"):
            row[f"{metric}_across_repeats_mean"] = float(group[metric].mean())
            row[f"{metric}_across_repeats_std"] = float(group[metric].std(ddof=1))
        row["minimum_eigenvalue_mean"] = float(group["minimum_eigenvalue"].mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        "f1_mean_across_repeats_mean", ascending=False
    ).reset_index(drop=True)


__all__ = [
    "ExperimentConfig",
    "FEATURES",
    "FEATURE_SET",
    "LOCKED_C",
    "LOCKED_CLASSICAL_C",
    "LOCKED_CLASSICAL_GAMMA",
    "LOCKED_MAPS",
    "NEXUS_ACKNOWLEDGEMENT",
    "PreparedExperiment",
    "aggregate_nexus_repeats",
    "backend_result_counts",
    "build_feature_circuit",
    "build_overlap_circuit",
    "discover_repo_root",
    "expected_workload",
    "nexus_collect_map_results",
    "nexus_compile_map_circuits",
    "nexus_estimate_map_cost",
    "nexus_load_execute_jobs",
    "nexus_submit_map_repeats",
    "nexus_upload_map_circuits",
    "prepare_experiment",
    "reconstruct_measured_kernels",
    "run_local_analysis",
]
