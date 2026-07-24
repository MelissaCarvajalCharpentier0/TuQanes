# TuQanes — Challenge 2: *Hacia el agua limpia para todos*

> Quantathon CR 2026 · Track 2 · Clasificación de potabilidad del agua con
> **SVM-RBF clásica** vs **QSVM con kernel cuántico de fidelidad** (Pytket / Quantinuum).
> Alineado con el **ODS 6 — Agua limpia y saneamiento**.

Este `README.md` es el **documento maestro** del repositorio: reúne el planteamiento del
problema, la línea base clásica, la implementación cuántica, los resultados con barras de
error, la comparación con la fase Nexus/H2 y una sección honesta de limitaciones. Está
pensado como base directa del informe técnico (PDF ≤ 8 páginas).

---

## 1. Resumen ejecutivo

Se estudia predecir si una muestra de agua es **potable (1)** o **no potable (0)** a partir de
mediciones fisicoquímicas, comparando de forma controlada dos familias de modelos:

1. **SVM clásica con kernel RBF** sobre el dataset completo y sobre un subconjunto cuántico
   de 80 filas.
2. **QSVM con kernel cuántico de fidelidad** construido con `pytket`, evaluado como kernel
   precomputado sobre los mismos folds congelados.

**Conclusión prudente (honesta):** con el protocolo actual **no hay evidencia de ventaja
cuántica**. El mejor mapa cuántico (`custom_water_domain_r1_robust`) alcanza en validación
cruzada de 80 filas **F1 = 0.6230 ± 0.1538**, ligeramente por encima de la SVM-RBF sobre las
mismas 80 filas (**F1 = 0.5773 ± 0.1335**), pero con **barras de error solapadas** y sobre un
subconjunto pequeño condicionado a la selección. En el holdout independiente clásico
(656 filas) la SVM-RBF obtiene **F1 = 0.5768**. Sí se entrega una **cadena reproducible**
completa y una fase Nexus/H2 empaquetada y modular.

Todas las cifras de este documento se **regeneran** con:

```bash
python main.py
```

y quedan consolidadas en [`artifacts/`](artifacts/).

---

## 2. Estructura del repositorio

```text
.
├── main.py                       # Punto de entrada único (reproduce cada cifra y figura)
├── requirements.txt              # Dependencias
├── pyproject.toml
├── README.md                     # Este documento
├── data/
│   └── water_potability.csv      # Dataset canónico (SHA-256 chem5_v01)
├── src/tuqanes/                  # Paquete del pipeline reproducible
│   ├── config.py                 # Rutas, contrato de datos, constantes
│   ├── io_utils.py               # Carga de dataset, kernels y folds + verificación de hash
│   ├── classical.py              # Baseline SVM-RBF (completo, holdout, subconjunto 80)
│   ├── quantum.py                # QSVM: cifras congeladas + verificación independiente
│   ├── metrics.py                # Métricas y evaluación de kernel precomputado
│   ├── plots.py                  # Figuras con barras de error
│   ├── nexus.py                  # Resumen de la fase Nexus (solo lectura, NO ejecuta)
│   └── report.py                 # Tabla maestra + resumen en Markdown
├── artifacts/                    # SALIDA unificada (se genera al correr main.py)
│   ├── classical/                # Métricas y predicciones clásicas
│   ├── quantum/                  # Ranking, geometría, ablaciones, recursos, verificación
│   ├── figures/                  # Figuras (F1 ranking, heatmaps, matriz de confusión, ...)
│   ├── nexus/                    # Estado del paquete + comparación exacta 16/64
│   ├── comparison/master_metrics.csv
│   └── RESULTS_SUMMARY.md
└── notebooks/                    # Trabajo original por integrante (NO se modifica)
    ├── Johnny/
    │   ├── previos_experiments/  # Exploración inicial (V02–V09, Gemini, Optuna, ...)
    │   ├── last_version/         # Generación vigente chem5_v01 (fuente de verdad)
    │   │   ├── artifacts_v1_2/   #   baseline clásico + subconjunto 80
    │   │   └── artifacts_v3_4/   #   kernels cuánticos + geometría + ablaciones
    │   └── nexus_reproducible/   # Fase Nexus/H2
    │       └── TuQanes_Package_Nexus/   # Paquete autocontenido (input/ + notebook + módulo)
    ├── Luis/                     # Estudio multiseed (N16/N32/N64, ZZ vs Custom, shots)
    └── Andrés/                   # (placeholder)
```

El pipeline **solo lee** los artefactos congelados de `notebooks/Johnny/last_version/` y **nunca
modifica** los notebooks. Toda la salida se escribe en `artifacts/`.

---

## 3. Cómo instalar y reproducir

### 3.1 Instalación

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

> El flujo principal (`main.py`) **solo** requiere el stack científico ligero
> (`numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`). Las dependencias
> cuánticas (`pytket`, `qiskit`, `guppylang`) se listan para reproducir los notebooks y el
> paquete Nexus, pero **no** son necesarias para regenerar las cifras del informe.

### 3.2 Ejecución (un solo comando)

```bash
python main.py                    # ejecuta TODO y consolida en artifacts/
```

Salida esperada (abreviada):

```text
[ok] Dataset canonico verificado (SHA-256 chem5_v01).
[ok] 10 kernels cuanticos exactos disponibles.
1/4  Baseline clasico  : CV F1=0.5573  Holdout(656) F1=0.5768
2/4  QSVM (80 filas)    : mejor custom_water_domain_r1_robust F1=0.6230 ± 0.1538 (C=10)
                          verificacion independiente: diff. max. F1 vs congelado = 0.0618
3/4  Figuras           : 13 figuras -> artifacts/figures
4/4  Fase Nexus        : paquete presente (NO ejecutado)
```

### 3.3 Ejecución por etapas

```bash
python main.py --stage classical   # baseline SVM-RBF (completo, holdout, subconjunto 80)
python main.py --stage quantum     # QSVM: cifras congeladas + verificación independiente
python main.py --stage figures     # regenera todas las figuras (con barras de error)
python main.py --stage report      # tabla maestra + RESULTS_SUMMARY.md
python main.py --stage nexus       # resumen de la fase Nexus/H2 (NO la ejecuta)
```

### 3.4 Verificación de integridad

`main.py` valida el **SHA-256** del dataset (`904004bd…`, generación `chem5_v01`) y la presencia
de los kernels exactos cacheados. Si el hash no coincide, avisa y usa los artefactos congelados
como cifras reportadas.

---

## 4. Contrato de datos y gobernanza

| Propiedad | Valor |
|---|---:|
| Filas | 3,276 |
| Variables predictoras | 9 |
| Clase 0 (no potable) | 1,998 (61.0 %) |
| Clase 1 (potable) | 1,278 (39.0 %) |
| SHA-256 | `904004bde729bfe3d2e195f46343bceead09e32a0eb95bb8184e7e20e029b2bf` |

**Espacio de variables vigente — `chem5_v01`** (5 variables → **5 qubits**):
`Sulfate, ph, Conductivity, Chloramines, Hardness`.

Reglas de gobernanza (evitan fuga de información):

1. El split train/holdout se hace **antes** de imputar, escalar o balancear.
2. Se conserva `source_index` para trazar cada fila al dataset original.
3. Imputadores/escaladores se ajustan **solo** con el training correspondiente.
4. El holdout **no** participa en selección de variables, hiperparámetros ni umbral.
5. Las 80 filas cuánticas se mantienen separadas del holdout; sus folds se **congelan**.
6. Los archivos `*_raw.csv` conservan los valores faltantes originales.

---

## 5. Decisiones de diseño y su justificación

Basado en los README internos de cada intento (`previos_experiments/`, `last_version/`,
`nexus_reproducible/`):

| Decisión | Justificación |
|---|---|
| **`chem5_v01`** (Sulfate, ph, Conductivity, Chloramines, Hardness) | Lectura fisicoquímica directa (ácido-base, contenido iónico, desinfectante residual, sulfatos, dureza) y **5 qubits**. Un comparador puramente estadístico (`chem5_prev`) daba F1 marginalmente mayor (~0.5666 vs ~0.5573) pero menos interpretable. Se prefiere trazabilidad de dominio. |
| **SVM-RBF con `C=10`, `gamma=0.01`** en el subconjunto de 80 | Mejor configuración de la grilla `C ∈ {0.1,1,10}`, `gamma ∈ {scale,auto,0.01}` por F1 medio de CV, congelada para comparación justa contra QSVM. |
| **Kernel de fidelidad** \|⟨φ(xᵢ)\|φ(xⱼ)⟩\|² vía circuito de solapamiento `U(xᵢ)U(xⱼ)†` | Permite QSVM como **kernel precomputado**, reutilizar folds y medir en hardware la probabilidad de `00000`. |
| **Escalado `robust_atan`** | Robusto a outliers; mapea a ángulos acotados `(2/π)·atan(z)` estables para los feature maps. |
| **3 mapas congelados para Nexus** (mejor por familia) | `custom_water_domain_r1_robust` (C=10), `zz_ring_r2_robust` (C=0.1), `pauli_z_zz_linear_r1_robust` (C=1) — uno por familia (custom / ZZFeatureMap / PauliFeatureMap). |
| **`H2-1LE` primero, `H2-Emulator` después** | `H2-1LE` (state-vector) aísla el **error de shots**; el emulador con ruido usa `run_id` distinto para no mezclar counts. |
| **≥ 3 repeticiones, sin escoger la mejor** | Se reporta media ± desviación estándar; ninguna repetición se descarta. |

---

## 6. Línea base clásica (SVM-RBF)

- División estratificada 80/20 → training 2,620, holdout 656.
- CV estratificada de 5 folds; balanceo por submuestreo **solo** en training.
- Evaluación final única sobre el holdout reservado.

**Resultados** (regenerados en [`artifacts/classical/`](artifacts/classical/)):

| Conjunto | n | F1 | Balanced acc. | MCC |
|---|---:|---:|---:|---:|
| CV completo (chem5_v01) | 2,620 | **0.5573 ± 0.0275** | 0.6270 | 0.2502 |
| Holdout independiente | 656 | **0.5768** | 0.6458 | 0.2878 |
| Subconjunto 80 (C=10, γ=0.01) | 80 | 0.5773 ± 0.1335 | 0.5375 | 0.0748 |

Figura: [`artifacts/figures/classical_holdout_confusion.png`](artifacts/figures/classical_holdout_confusion.png).

---

## 7. Implementación cuántica (QSVM)

**Pipeline:** cada feature map codifica las 5 variables en 5 qubits; el kernel de fidelidad
exacto (statevector, **sin shots ni ruido**) se calculó con `pytket` en `last_version`
(Partes 3–4) y quedó **congelado** en `artifacts_v3_4/kernels/` (matrices 80×80). El pipeline:

1. **Carga las cifras reportadas** (ranking por mapa, grilla completa de C, geometría del
   kernel, ablaciones, recursos de circuito, sensibilidad a shots/ruido) desde los artefactos
   congelados → [`artifacts/quantum/`](artifacts/quantum/). Son la **fuente de verdad**.
2. **Verificación independiente:** re-ajusta una QSVM con kernel precomputado sobre los mismos
   kernels y folds congelados (solo `numpy`/`sklearn`, sin `pytket`), y reporta la diferencia
   máxima de F1 frente a lo congelado → [`artifacts/quantum/qsvm_recompute_check.csv`](artifacts/quantum/qsvm_recompute_check.csv).
   **Diferencia máxima = 0.0618** (corrobora la conclusión sin depender de `pytket`; la
   diferencia proviene del post-procesamiento del kernel del pipeline histórico).

**Ranking por F1 (CV, 80 filas, cifras congeladas):**

| # | Mapa | Familia | C | F1 | Balanced acc. | MCC |
|--:|---|---|---:|---:|---:|---:|
| 1 | `custom_water_domain_r1_robust` | custom | 10.0 | **0.6230 ± 0.1538** | 0.625 | 0.2545 |
| 2 | `zz_ring_r2_robust` | ZZFeatureMap | 0.1 | 0.6072 ± 0.2059 | 0.625 | 0.2539 |
| 3 | `custom_water_domain_r2_reupload_robust` | custom | 1.0 | 0.6015 ± 0.1502 | 0.600 | 0.2028 |
| 4 | `pauli_z_zz_linear_r1_robust` | PauliFeatureMap | 1.0 | 0.5768 ± 0.1537 | 0.5875 | 0.1776 |
| 5 | `zz_linear_r1_robust` | ZZFeatureMap | 1.0 | 0.5768 ± 0.1537 | 0.5875 | 0.1776 |
| 6 | `pauli_z_zz_ring_r1_robust` | PauliFeatureMap | 1.0 | 0.5691 ± 0.1654 | 0.5875 | 0.1782 |
| 7 | `zz_ring_r1_robust` | ZZFeatureMap | 1.0 | 0.5691 ± 0.1654 | 0.5875 | 0.1782 |
| 8 | `zz_full_r1_robust` | ZZFeatureMap | 1.0 | 0.5689 ± 0.1750 | 0.5625 | 0.1297 |
| 9 | `pauli_xz_xxzz_linear_r1_robust` | PauliFeatureMap | 10.0 | 0.5588 ± 0.1175 | 0.550 | 0.0982 |
| 10 | `pauli_z_zz_ring_r1_minmax` | PauliFeatureMap | 1.0 | 0.4957 ± 0.2416 | 0.5125 | 0.0105 |

Figuras: [`map_f1_ranking.png`](artifacts/figures/map_f1_ranking.png),
[`classical_vs_quantum_f1.png`](artifacts/figures/classical_vs_quantum_f1.png) y un
heatmap por kernel (`kernel_heatmap_<mapa>.png`).

---

## 8. Fase Nexus / H2 (modular, **no ejecutada** por el pipeline)

La ejecución en hardware/emulador de Quantinuum vive **autocontenida** en
[`notebooks/Johnny/nexus_reproducible/TuQanes_Package_Nexus/`](notebooks/Johnny/nexus_reproducible/TuQanes_Package_Nexus/)
y **no** se corre desde `main.py` porque consume cuota (HQC). El paquete es portable: se copia
tal cual a Nexus Lab.

```text
TuQanes_Package_Nexus/
├── Nexus_QSVM_top3_reproducible.ipynb   # punto de entrada Nexus
├── nexus_qsvm.py                        # funciones, validaciones, checkpoints
├── input/                               # 7 archivos congelados (dataset + artifacts)
└── output/                              # se crea al ejecutar (por run_id)
```

Etapas seguras y deliberadas: `local_only` → `cost` (revisa HQC) → `submit` → `collect`.
El paquete anota su estado en [`artifacts/nexus/package_status.csv`](artifacts/nexus/package_status.csv)
(`present=True`, `executed_by_pipeline=False`).

**Estado de los últimos jobs (evidencia local):** las corridas históricas en `H2-1LE` quedaron
en estado `SUBMITTED` y **nunca se recolectaron counts**; la corrida de 64 muestras
(`pilot64-strict-v1`) solo **compiló 2 de 3 mapas** y **no ejecutó** ninguno. Por tanto **no
existen resultados medidos de H2** en local: la única referencia numérica de 64 muestras es la
**exacta local** (statevector), generada por este código.

---

## 9. Comparación: últimos artifacts vs. última ejecución Nexus (64 muestras)

> **Referencia:** únicamente artefactos generados por el último código
> ([`artifacts/nexus/local_exact_comparison_16_64.csv`](artifacts/nexus/local_exact_comparison_16_64.csv)
> y [`artifacts/comparison/master_metrics.csv`](artifacts/comparison/master_metrics.csv)).

Para el **mismo subconjunto de 64 filas** usado en la fase Nexus, comparando la referencia
exacta local (último código) contra lo que produjo la última ejecución en Nexus:

| Modelo / mapa | C | Exacto local (último código) F1 | Balanced acc. | MCC | Nexus H2 medido (64) |
|---|---:|---:|---:|---:|---|
| SVM-RBF clásica | 10.0 | 0.3741 ± 0.1401 | 0.4048 | −0.1976 | *no recolectado* |
| `custom_water_domain_r1_robust` | 10.0 | **0.5231 ± 0.3160** | 0.5833 | 0.1432 | *no recolectado* |
| `zz_ring_r2_robust` | 0.1 | 0.4850 ± 0.2739 | 0.5167 | 0.0188 | *no recolectado* |
| `pauli_z_zz_linear_r1_robust` | 1.0 | 0.4205 ± 0.2408 | 0.5024 | −0.0041 | *no recolectado* |

**Lectura honesta:**

- La última ejecución Nexus de 64 muestras **no produjo counts** (jobs `SUBMITTED`,
  2/3 mapas compilados, 0 ejecutados), por lo que la columna *Nexus H2 medido* está vacía y el
  delta exacto-vs-H2 **no puede calcularse todavía**.
- Sobre las 64 filas exactas, el mejor mapa es `custom_water_domain_r1_robust`
  (F1 = 0.5231), superando a la SVM-RBF sobre las mismas filas (F1 = 0.3741), pero con
  **barras de error muy grandes** (folds pequeños). Estas 64 filas son parte del pool de
  selección, así que es una comparación **condicionada**, no un holdout independiente.
- Para completar la comparación basta copiar el paquete a Nexus y correr `collect` sobre los
  jobs; el pipeline `collect` reconstruye los kernels medidos y `aggregate_metrics.csv`.

*Nota: las cifras de 64 filas (sección 9) usan un subconjunto distinto y folds distintos a la
CV de 80 filas (sección 7); por eso no coinciden numéricamente y no deben mezclarse.*

---

## 10. Tabla maestra consolidada

Generada en [`artifacts/comparison/master_metrics.csv`](artifacts/comparison/master_metrics.csv)
y resumida en [`artifacts/RESULTS_SUMMARY.md`](artifacts/RESULTS_SUMMARY.md):

| Etapa | Modelo | n | F1 | ± | Balanced acc. | MCC |
|---|---|---:|---:|---:|---:|---:|
| clásico | SVM-RBF chem5_v01 (dataset completo, CV) | 2,620 | 0.5573 | 0.0275 | 0.6270 | 0.2502 |
| clásico | SVM-RBF chem5_v01 (holdout 656) | 656 | 0.5768 | — | 0.6458 | 0.2878 |
| clásico | SVM-RBF (subconjunto 80, C=10 γ=0.01) | 80 | 0.5773 | 0.1335 | 0.5375 | 0.0748 |
| cuántico | QSVM `custom_water_domain_r1_robust` (C=10) | 80 | **0.6230** | 0.1538 | 0.625 | 0.2545 |
| cuántico | QSVM `zz_ring_r2_robust` (C=0.1) | 80 | 0.6072 | 0.2059 | 0.625 | 0.2539 |
| cuántico | QSVM `pauli_z_zz_linear_r1_robust` (C=1) | 80 | 0.5768 | 0.1537 | 0.5875 | 0.1776 |

*(La tabla completa con los 10 mapas está en el CSV.)*

---

## 11. Figuras generadas (con barras de error)

Todas en [`artifacts/figures/`](artifacts/figures/):

- `map_f1_ranking.png` — ranking de mapas por F1 con barras de error (± std CV).
- `classical_vs_quantum_f1.png` — clásico vs. mejor cuántico.
- `classical_holdout_confusion.png` — matriz de confusión del holdout clásico.
- `kernel_heatmap_<mapa>.png` — geometría de cada kernel de fidelidad (10 mapas).

---

## 12. Panorama de todas las pruebas realizadas (alto nivel)

Resumen general de los intentos (el detalle vive en el `README.md` de cada carpeta):

- **`notebooks/Johnny/previos_experiments/`** — Exploración inicial: barridos SVM-RBF
  (GridSearch + Optuna), primeros estudios de feature maps cuánticos y `PauliFeatureMap`,
  implementaciones de kernel cuántico **desde cero con `pytket`**, topología en anillo y una
  QSVM unificada (V02–V09), además de pruebas de artefactos corregidos. Estableció la mejor
  referencia clásica temprana (`max_f1_v3`, F1 ≈ 0.6297 en holdout).
- **`notebooks/Johnny/last_version/`** — **Generación vigente y fuente de verdad** (`chem5_v01`):
  Partes 1–2 (baseline SVM-RBF, holdout, subconjunto de 80) → `artifacts_v1_2/`; Partes 3–4
  (feature maps, kernels de fidelidad exactos, QSVM, geometría, recursos de circuito,
  sensibilidad a shots y a ruido proxy, ablaciones por familia/topología/escalado)
  → `artifacts_v3_4/`. **De aquí toma el pipeline todas las cifras.**
- **`notebooks/Johnny/nexus_reproducible/`** — Fase Nexus/H2: paquete autocontenido con 3 mapas
  congelados, protocolo por etapas y checkpoints. Estado local: jobs `SUBMITTED` sin recolectar
  (ver §8).
- **`notebooks/Luis/`** — Estudio **multiseed** de reproducibilidad: subconjuntos N16/N32/N64 con
  múltiples semillas, comparación **ZZ vs Custom**, estudios de shots (256/1024), estimación de
  recursos H2 y comparaciones pareadas en N64. Contexto complementario que refuerza la
  estabilidad de las conclusiones.
- **`notebooks/Andrés/`** — Placeholder, sin contenido sustantivo.

---

## 13. Limitaciones (sección honesta, obligatoria)

- **Sin ventaja cuántica demostrada.** La QSVM (F1 = 0.6230) supera marginalmente a la SVM-RBF
  sobre las mismas 80 filas (F1 = 0.5773), pero **las barras de error se solapan** y la CV está
  **condicionada a la selección** del subconjunto; no es una estimación insesgada de
  generalización.
- **Subconjunto cuántico pequeño (80 filas).** Los folds son reducidos → desviaciones estándar
  grandes (± 0.15–0.24). El piloto de 16 y la evaluación de 64 tienen folds aún más pequeños.
- **Dataset limitado.** Sin información geográfica/temporal/de procedencia; no es un sistema de
  certificación de agua. La **imputación por mediana por clase usa la etiqueta** para elegir la
  mediana, por lo que **no es desplegable** en producción (la clase real no se conoce al predecir).
- **Nexus/H2 no medido.** Los jobs quedaron `SUBMITTED` sin recolectar; la corrida de 64 solo
  compiló 2/3 mapas. No hay counts de hardware/emulador; el costo (HQC) del barrido completo de
  64 × 4096 shots × 3 repeticiones es **prohibitivo** en el emulador con ruido.
- **Dependencia del protocolo de kernel.** El recómputo independiente difiere hasta **0.0618** en
  F1 respecto a las cifras congeladas por diferencias de post-procesamiento del kernel; las
  cifras son reproducibles pero **dependientes del protocolo**.
- **Umbral orientado a métrica.** Los F1 clásicos altos se obtienen con umbrales que generan
  demasiados falsos positivos para certificar potabilidad.

---

## 14. Reproducibilidad y procedencia

- **Punto de entrada único:** `python main.py` regenera cada figura y cifra reportada.
- **Verificación de datos:** SHA-256 del dataset (`904004bd…`) validado en cada corrida.
- **Entradas de solo lectura:** el pipeline nunca modifica `notebooks/`; consume los artefactos
  congelados de `last_version/`.
- **Salida determinista:** folds y kernels congelados garantizan cifras estables entre corridas.
- **Fase Nexus separada:** empaquetada, modular y no ejecutada por el flujo principal.

---

## 15. Entregables (mapeo a los requisitos)

| Requisito | Dónde |
|---|---|
| Código (Pytket / Guppy) | `notebooks/` (originales) + `src/tuqanes/` (pipeline) + paquete Nexus |
| `requirements.txt` | [`requirements.txt`](requirements.txt) |
| Punto de entrada único que reproduce cada figura/cifra | [`main.py`](main.py) → [`artifacts/`](artifacts/) |
| `README.md` | este documento |
| Base para el informe técnico (PDF ≤ 8 pág.) | secciones 1, 6, 7, 9, 10, 13 |

---

*No se afirma ventaja cuántica. Este proyecto es un benchmark académico reproducible, no un
sistema de certificación de potabilidad del agua.*
