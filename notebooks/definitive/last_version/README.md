# Challenge 2 — Hacia el agua limpia para todos

> **README de la generación actual**  
> **Alcance:** documentar únicamente los notebooks, artefactos, resultados y decisiones vigentes de la nueva implementación.  
> **Última actualización:** 23 de julio de 2026 — adopción de `chem5_v01`.

---

## 1. Resumen

Este proyecto estudia la clasificación de muestras de agua como **potables** o **no potables** mediante:

1. una **SVM clásica con kernel RBF**;
2. una **QSVM con kernel cuántico de fidelidad**, construida con `pytket`;
3. una fase posterior, todavía no implementada, para ejecutar kernels medibles mediante **Quantinuum Nexus** y un emulador de la familia H2.

La generación actual reinicia el flujo experimental con una estructura reducida y auditable:

```text
water_potability.csv
        │
        ▼
Hacia_el_agua_limpia_partes_1_2_SVM_RBF.ipynb
        │
        ├── baseline clásico completo
        ├── holdout reservado
        ├── auditoría de threshold
        └── subconjunto QSVM de 80 muestras basado en chem5_v01
        │
        ▼
artifacts_v1/
        │
        ▼
Hacia_el_agua_limpia_partes_3_4_pytket_kernels.ipynb
        │
        ├── circuitos de feature maps
        ├── kernels de fidelidad exactos
        ├── QSVM con kernel precomputado
        ├── geometría del kernel
        ├── recursos de circuito
        └── sensibilidad a shots y ruido proxy
        │
        ▼
artifacts_v3_4/                 [pendiente de ejecución completa]
        │
        ▼
Quantinuum Nexus / H2          [fase de planeamiento]
```

Este README **no reconstruye la historia ni la estructura interna de `previous_experiment/`**. El material anterior se utiliza únicamente como fuente de algunas decisiones metodológicas y datos de referencia, declarados en la sección [10](#10-decisiones-heredadas-y-procedencia).

---

## 2. Estructura vigente

```text
Colab Notebooks/
├── README.md
├── water_potability.csv
│
├── Hacia_el_agua_limpia_partes_1_2_SVM_RBF.ipynb
├── artifacts_v1/
│   ├── data_contract.json
│   ├── train_holdout_split_report.csv
│   ├── feature_cv_summary.csv
│   ├── feature_cv_fold_metrics.csv
│   ├── feature_holdout_metrics.csv
│   ├── strict_svm_cv_fold_metrics.csv
│   ├── strict_svm_oof_predictions.csv
│   ├── strict_svm_holdout_predictions.csv
│   ├── strict_svm_holdout_confusion_matrix.png
│   ├── threshold_audit.csv
│   ├── quantum80_raw.csv
│   ├── quantum80_selected_features.csv
│   ├── quantum80_folds.csv
│   ├── quantum80_manifest.json
│   ├── quantum80_svm_cv_summary.csv
│   ├── quantum80_svm_grid_fold_metrics.csv
│   ├── quantum80_svm_best_fold_metrics.csv
│   ├── quantum80_svm_oof_predictions.csv
│   ├── quantum80_svm_oof_metrics.csv
│   └── quantum80_svm_oof_confusion_matrix.png
│
├── Hacia_el_agua_limpia_partes_3_4_pytket_kernels.ipynb
├── artifacts_v3_4/                         # se crea al ejecutar Partes 3 y 4
│   ├── circuits/
│   ├── figures/
│   ├── kernels/
│   ├── map_cv_fold_metrics.csv
│   ├── map_cv_predictions_all_C.csv
│   ├── map_cv_summary_all_C.csv
│   ├── map_cv_best_by_map.csv
│   ├── kernel_geometry_summary.csv
│   ├── circuit_resource_summary.csv
│   ├── shot_sensitivity_summary.csv
│   ├── noise_proxy_sensitivity_summary.csv
│   ├── ablation_by_family.csv
│   ├── ablation_by_topology.csv
│   ├── ablation_by_scaling.csv
│   └── parts_3_4_manifest.json
│
└── previous_experiment/                    # archivo de consulta; fuera del flujo vigente
```

### Fuentes de verdad

| Componente | Fuente vigente |
|---|---|
| Dataset | `water_potability.csv` |
| Metodología y resultados clásicos | `Hacia_el_agua_limpia_partes_1_2_SVM_RBF.ipynb` |
| Inputs congelados para QSVM | `artifacts_v1/quantum80_*` |
| Definición de circuitos y kernels | `Hacia_el_agua_limpia_partes_3_4_pytket_kernels.ipynb` |
| Resultados de Partes 3 y 4 | `artifacts_v3_4/`, cuando la ejecución se complete |
| Decisiones heredadas | `previous_experiment/README.md`, solo donde se cite expresamente |

No deben mezclarse métricas o artefactos de otras generaciones como si pertenecieran a este protocolo.

---

## 3. Dataset y contrato de datos

Archivo de entrada:

```text
water_potability.csv
```

### Características del snapshot utilizado

| Propiedad | Valor |
|---|---:|
| Filas | 3,276 |
| Variables predictoras | 9 |
| Clase `0`, no potable | 1,998 — 60.99% |
| Clase `1`, potable | 1,278 — 39.01% |
| SHA-256 | `904004bde729bfe3d2e195f46343bceead09e32a0eb95bb8184e7e20e029b2bf` |

Variables originales:

```text
ph
Hardness
Solids
Chloramines
Sulfate
Conductivity
Organic_carbon
Trihalomethanes
Turbidity
```

Valores faltantes:

| Variable | Faltantes | Proporción |
|---|---:|---:|
| `Sulfate` | 781 | 23.84% |
| `ph` | 491 | 14.99% |
| `Trihalomethanes` | 162 | 4.95% |

### Reglas del protocolo actual

- El split train/holdout se realiza antes de imputar, escalar o balancear.
- Se conserva `source_index` para rastrear cada fila hasta el dataset original.
- La imputación y el escalamiento se ajustan exclusivamente con el training correspondiente.
- El submuestreo de clases se aplica solamente al training.
- El holdout no participa en la selección de variables, hiperparámetros o threshold.
- Los folds del subconjunto QSVM se congelan y deben reutilizarse sin cambios.
- Los archivos `*_raw.csv` conservan los valores faltantes originales.

> **Limitación importante:** la imputación por mediana por clase usa la etiqueta para elegir la mediana correspondiente. Esto reproduce el planteamiento experimental, pero no es directamente desplegable en producción porque la clase real no se conoce al momento de predecir.

---

## 4. Partes 1 y 2 — Baseline SVM-RBF

Notebook:

```text
Hacia_el_agua_limpia_partes_1_2_SVM_RBF.ipynb
```

### 4.1 Protocolo

- División estratificada: 80% training y 20% holdout.
- Training: 2,620 muestras.
- Holdout: 656 muestras.
- Validación cruzada estratificada de cinco folds.
- Balanceo mediante submuestreo solamente dentro del training.
- Grilla evaluada:

```python
C = [0.1, 1.0, 10.0]
gamma = ["scale", "auto", 0.01]
```

- La grilla se conserva como auditoría estadística completa.
- Dentro de cada espacio de variables, `C` y `gamma` se eligen por F1 medio de CV, con `balanced_accuracy` y MCC como desempates.
- El espacio operativo se fija previamente en `chem5_v01` por criterio fisicoquímico.
- Evaluación final única sobre el holdout reservado.

### 4.2 Espacios de variables auditados

| Identificador | Variables | Uso |
|---|---|---|
| `raw9` | las nueve variables originales | referencia completa |
| `chem5_v01` | `Sulfate`, `ph`, `Conductivity`, `Chloramines`, `Hardness` | **espacio predefinido vigente** |
| `chem5_prev` | `Sulfate`, `ph`, `Solids`, `Chloramines`, `Hardness` | comparador estadístico |
| `chem4_prev` | `Sulfate`, `ph`, `Solids`, `Chloramines` | ablación |
| `preview4` | `ph`, `Solids`, `Conductivity`, `Turbidity` | ablación |

`chem5_prev` conserva una ventaja pequeña de F1 medio frente a `chem5_v01` —aproximadamente `0.5666` frente a `0.5573`—, pero no se adopta como base del flujo. Se prefiere `chem5_v01` porque sus cinco variables tienen una lectura fisicoquímica más directa para el objetivo experimental:

- `ph`: condición ácido-base;
- `Conductivity`: respuesta global al contenido iónico disuelto;
- `Chloramines`: desinfectante residual;
- `Sulfate`: contenido de sulfatos;
- `Hardness`: carga mineral asociada principalmente a calcio y magnesio.

`Solids` es una medida agregada de sólidos y, para esta versión del diseño, se considera menos específica que `Conductivity`. Esta es una decisión de dominio y trazabilidad, no una afirmación regulatoria ni una prueba de causalidad.

### 4.3 Configuración predefinida

```text
feature_set = chem5_v01
features    = Sulfate, ph, Conductivity, Chloramines, Hardness
C           = 10
gamma       = auto
```

Los hiperparámetros corresponden a la mejor configuración de `chem5_v01` dentro de la grilla. El uso de cinco variables mantiene el circuito esperado en cinco qubits.

### 4.4 Resultados de validación cruzada de `chem5_v01`

| Métrica | Media CV |
|---|---:|
| Accuracy | 0.6355 |
| Precision | 0.5302 |
| Recall | 0.5881 |
| F1 | 0.5573 |
| Specificity | 0.6658 |
| Balanced accuracy | 0.6270 |
| MCC | 0.2502 |

La diferencia frente al líder puramente estadístico es pequeña y se acepta como trade-off explícito por coherencia fisicoquímica.

### 4.5 Resultados en holdout

Se usa la regla estándar de `SVC.predict()`, equivalente a threshold `0` sobre `decision_function`.

| Métrica | Holdout |
|---|---:|
| Accuracy | 0.6555 |
| Precision | 0.5540 |
| Recall | 0.6016 |
| F1 | 0.5768 |
| Specificity | 0.6900 |
| Balanced accuracy | 0.6458 |
| MCC | 0.2878 |

Matriz de confusión:

| | Predicción no potable | Predicción potable |
|---|---:|---:|
| **Real no potable** | TN = 276 | FP = 124 |
| **Real potable** | FN = 102 | TP = 154 |

El resultado sigue siendo un baseline académico y no un sistema para certificar agua potable. Los 124 falsos positivos son muestras no potables declaradas potables.

### 4.6 Auditoría del threshold

El threshold elegido para maximizar F1 usando predicciones OOF del training fue:

```text
-0.736891
```

Su aplicación al holdout produce:

| Métrica | Threshold 0 | Threshold −0.736891 |
|---|---:|---:|
| F1 | 0.5768 | 0.5777 |
| Recall | 0.6016 | 0.8281 |
| Specificity | 0.6900 | 0.3350 |
| Balanced accuracy | 0.6458 | 0.5816 |
| MCC | 0.2878 | 0.1790 |
| Falsos positivos | 124 | 266 |

El cambio de threshold apenas modifica F1, pero empeora fuertemente especificidad, balanced accuracy, MCC y falsos positivos. Se conserva únicamente como auditoría académica.

---

## 5. Subconjunto congelado para QSVM

La salida de Partes 1 y 2 incluye una población experimental balanceada para que la QSVM y su baseline clásico utilicen exactamente las mismas filas, variables y particiones.

### Contrato del subconjunto

| Propiedad | Valor |
|---|---:|
| Muestras totales | 80 |
| No potables | 40 |
| Potables | 40 |
| Variables/qubits | 5 |
| Folds | 5 |
| Training por fold | 64 |
| Validación por fold | 16 — 8 por clase |
| Intersección con holdout de 656 | 0 |

Variables:

```text
Sulfate
ph
Conductivity
Chloramines
Hardness
```

La selección toma 40 muestras por clase mediante cuantiles uniformemente espaciados del `decision_score` de una SVM de ranking entrenada solo sobre el training original.

### Qué significa realmente el esquema 80 → 64/16

Las 80 filas **no se capturan porque 64 sean un training permanente y 16 un test permanente**. Las 80 forman el conjunto completo del experimento reducido. Después se aplica validación cruzada de cinco folds:

1. en cada fold, 64 filas entrenan el modelo;
2. las 16 restantes se usan como validación;
3. el grupo de 16 cambia en cada fold;
4. cada una de las 80 filas se valida exactamente una vez;
5. cada fila participa en training en los otros cuatro folds.

Las predicciones OOF unen las cinco validaciones y cubren las 80 filas sin evaluar cada fila con el modelo que la utilizó para entrenar. No existe dentro de estas 80 un test final independiente adicional. El holdout externo de 656 filas continúa separado, pero la QSVM actual todavía no se evalúa sobre él por el costo de construir ese kernel.

### Baseline SVM sobre las mismas 80 muestras

Mejor configuración:

```text
C = 10
gamma = 0.01
```

Promedios entre los cinco folds:

| Métrica | Media |
|---|---:|
| Accuracy | 0.5375 |
| Precision | 0.5357 |
| Recall | 0.6500 |
| F1 | 0.5773 |
| Specificity | 0.4250 |
| Balanced accuracy | 0.5375 |
| MCC | 0.0748 |

Predicciones OOF agregadas sobre las 80 filas:

| Métrica | Valor |
|---|---:|
| Accuracy | 0.5375 |
| Precision | 0.5306 |
| Recall | 0.6500 |
| F1 | 0.5843 |
| Specificity | 0.4250 |
| Balanced accuracy | 0.5375 |
| MCC | 0.0770 |
| TN | 17 |
| FP | 23 |
| FN | 14 |
| TP | 26 |

Este es el comparador clásico principal para la QSVM con `chem5_v01`. La caída respecto al subconjunto anterior confirma que cambiar variables y volver a seleccionar las 80 filas genera un experimento distinto; por eso no deben reutilizarse las métricas ni kernels basados en `chem5_prev`.

---

## 6. Partes 3 y 4 — Kernels cuánticos con `pytket`

Notebook:

```text
Hacia_el_agua_limpia_partes_3_4_pytket_kernels.ipynb
```

Entrada obligatoria:

```text
artifacts_v1/quantum80_raw.csv
artifacts_v1/quantum80_selected_features.csv
artifacts_v1/quantum80_folds.csv
artifacts_v1/quantum80_manifest.json
```

El notebook valida que el manifiesto contenga exactamente:

```text
feature_set = chem5_v01
features    = Sulfate, ph, Conductivity, Chloramines, Hardness
```

Si encuentra artefactos anteriores basados en `Solids`, detiene la ejecución para evitar mezclar generaciones.

### 6.1 Objetivo actual

- Construir feature maps de cinco qubits.
- Calcular kernels de fidelidad exactos mediante statevectors.
- Entrenar una SVM con `kernel="precomputed"`.
- Repetir imputación, escalamiento y construcción del kernel dentro de cada fold.
- Auditar la geometría del kernel.
- Contabilizar profundidad y puertas de dos qubits después de compilación.
- Simular sensibilidad a shots finitos.
- Aplicar un proxy numérico de ruido antes de la migración a Nexus/H2.

### 6.2 Preprocesamiento angular

El notebook soporta tres transformaciones:

| Escalado | Definición resumida |
|---|---|
| `robust_atan` | estandarización y compresión mediante arctan |
| `minmax` | mapeo del rango de training a `[-1, 1]` |
| `zscore_clip` | z-score dividido por 2.5 y recortado a `[-1, 1]` |

Para CV rigurosa, las medianas, el escalador y los rangos se ajustan dentro del training de cada fold. Los kernels globales se generan únicamente para inspección descriptiva y visualización.

### 6.3 Catálogo de mapas

| Mapa | Familia | Topología | Reps. | Escalado |
|---|---|---|---:|---|
| `zz_linear_r1_robust` | ZZFeatureMap | linear | 1 | robust_atan |
| `zz_ring_r1_robust` | ZZFeatureMap | ring | 1 | robust_atan |
| `zz_full_r1_robust` | ZZFeatureMap | full | 1 | robust_atan |
| `zz_ring_r2_robust` | ZZFeatureMap | ring | 2 | robust_atan |
| `pauli_z_zz_linear_r1_robust` | PauliFeatureMap | linear | 1 | robust_atan |
| `pauli_z_zz_ring_r1_robust` | PauliFeatureMap | ring | 1 | robust_atan |
| `pauli_xz_xxzz_linear_r1_robust` | PauliFeatureMap | linear | 1 | robust_atan |
| `pauli_z_zz_ring_r1_minmax` | PauliFeatureMap | ring | 1 | minmax |
| `custom_water_domain_r1_robust` | custom water-domain | domain | 1 | robust_atan |
| `custom_water_domain_r2_reupload_robust` | custom water-domain | domain | 2 | robust_atan |

### 6.4 Mapa primario de trabajo

```text
custom_water_domain_r2_reupload_robust
```

Se mantiene como hipótesis primaria porque permite evaluar directamente las interacciones ponderadas de `chem5_v01`. El cambio de variables invalida sus resultados anteriores: el mapa debe ejecutarse de nuevo antes de considerarlo ganador o compararlo con la SVM.

### 6.5 Mapa inspirado en dominio

Interacciones actualizadas:

| Variables | Peso |
|---|---:|
| `ph` — `Sulfate` | 1.10 |
| `Conductivity` — `Hardness` | 1.20 |
| `Chloramines` — `Sulfate` | 0.90 |
| `ph` — `Hardness` | 0.95 |
| `Chloramines` — `Conductivity` | 0.80 |

Los pesos están normalizados alrededor de `1` y expresan prioridad relativa dentro del circuito:

- `Conductivity`–`Hardness` recibe el peso mayor por su relación conceptual con la carga mineral e iónica;
- `ph`–`Sulfate` y `ph`–`Hardness` representan interacciones de química ácido-base y mineral;
- los enlaces con `Chloramines` se mantienen moderados para no asumir una causalidad fuerte no demostrada.

Son hiperparámetros heurísticos. No son límites de potabilidad, coeficientes químicos aprendidos ni evidencia causal. Deben auditarse mediante ablaciones.

### 6.6 Evaluaciones previstas

Para cada mapa se calculan o se preparan:

- accuracy;
- precision;
- recall;
- specificity;
- F1;
- balanced accuracy;
- MCC;
- alineamiento kernel-target;
- similitud media intra-clase e inter-clase;
- rango efectivo;
- dispersión de elementos off-diagonal;
- autovalores del kernel;
- número de puertas y profundidad;
- puertas y profundidad de dos qubits;
- sensibilidad a `256`, `1,024` y `4,096` shots;
- proxy de ruido en niveles `0.01`, `0.03` y `0.05`.

---

## 7. Estado de ejecución

### Completado

- Dataset validado y hasheado.
- Split 80/20 congelado.
- Auditoría de cinco espacios de variables.
- Adopción predefinida de `chem5_v01`.
- Selección de `C=10`, `gamma=auto` dentro de `chem5_v01`.
- Evaluación final clásica sobre holdout.
- Auditoría del threshold.
- Regeneración del subconjunto QSVM de 80 filas con `Conductivity`.
- Cinco folds balanceados: 64 training y 16 validación por fold.
- Baseline clásico OOF sobre las mismas 80 filas.
- Actualización de las interacciones de dominio en `pytket`.
- Validación del contrato de variables en Partes 3 y 4.

### Pendiente

- Reejecutar completamente Partes 3 y 4 con `chem5_v01`.
- Regenerar y validar `artifacts_v3_4/`; los resultados anteriores con `Solids` no son autoritativos.
- Seleccionar el mapa cuántico de la generación actual.
- Comparar la QSVM contra el nuevo baseline de 80 filas.
- Construir el protocolo de fidelidad basado en mediciones.
- Integrar Nexus.
- Estimar costos de ejecución.
- Ejecutar un piloto en emulador H2.

---

## 8. Reproducción

### 8.1 Partes 1 y 2

1. Colocar `water_potability.csv` en `MyDrive/Colab Notebooks/`.
2. Abrir `Hacia_el_agua_limpia_partes_1_2_SVM_RBF.ipynb`.
3. Ejecutar todas las celdas desde el inicio.
4. Confirmar que el hash del dataset coincide con el documentado.
5. Verificar la creación de `artifacts_v1/`.
6. No modificar las semillas, filas ni folds después de congelar el subconjunto QSVM.

Semillas vigentes:

```python
split              = 20260801
feature_cv         = 20260803
quantum_sample     = 20260805
outer_folds        = 20260806
undersample_base   = 20260820
```

### 8.2 Partes 3 y 4

1. Ejecutar y validar primero Partes 1 y 2.
2. Reiniciar el runtime si la celda de entorno reinstala NumPy o SciPy.
3. Abrir `Hacia_el_agua_limpia_partes_3_4_pytket_kernels.ipynb`.
4. Confirmar que `PART12_DIR` apunta a `artifacts_v1/`.
5. Ejecutar todas las celdas.
6. Verificar que se creen `artifacts_v3_4/circuits`, `figures` y `kernels`.
7. Considerar válidas las métricas solamente cuando exista `parts_3_4_manifest.json` y todos los resúmenes CSV esperados.

---

## 9. Planeamiento para Nexus y H2

La implementación actual calcula fidelidades mediante statevectors:

\[
K_{ij}=|\langle\psi(x_i)|\psi(x_j)\rangle|^2.
\]

Este método es apropiado para simulación exacta, pero no se traslada literalmente a hardware o a un emulador basado en shots. La fase Nexus debe reemplazarlo por circuitos medibles:

1. preparar \(U(x_i)\);
2. aplicar \(U(x_j)^\dagger\);
3. medir los cinco qubits;
4. estimar la fidelidad con la frecuencia del resultado `00000`.

\[
\widehat K_{ij}=\frac{N(00000)}{N_{shots}}.
\]

### Secuencia recomendada

1. Completar los kernels exactos de Partes 3 y 4.
2. Reducir los diez mapas a una lista corta de uno a tres candidatos.
3. Validar el circuito de solapamiento con 8–12 muestras.
4. Comparar fidelidad exacta y fidelidad estimada con shots.
5. Ejecutar un solo fold y un solo mapa.
6. Registrar circuitos, compilación, shots, IDs de jobs y kernels brutos.
7. Escalar a los cinco folds solamente si la geometría y las métricas se mantienen estables.

No debe iniciarse la ejecución completa en H2 antes de cerrar los resultados exactos, la selección del mapa y la estimación de recursos.

---

## 10. Decisiones heredadas y procedencia

Esta sección conserva únicamente decisiones y datos útiles del trabajo anterior. No reproduce su arquitectura, cronología ni catálogo de notebooks.

| Decisión o dato heredado | Uso en la generación actual | Fuente |
|---|---|---|
| No seleccionar un modelo únicamente por F1 | Se reportan especificidad, balanced accuracy, MCC y falsos positivos junto con F1 | `previous_experiment/README.md`; reiterado en el notebook actual de Partes 1 y 2 |
| No comparar poblaciones distintas | La QSVM de 80 filas se compara con una SVM evaluada sobre las mismas 80 filas y folds | `previous_experiment/README.md`; aplicado en `quantum80_*` |
| Cinco variables eran una hipótesis prometedora | Se auditan nuevamente; la generación actual adopta `chem5_v01` por coherencia fisicoquímica, aun con una pequeña penalización de CV | `previous_experiment/README.md`; decisión y validación nuevas en Partes 1 y 2 |
| El patrón Pauli `Z + ZZ` con topología ring era un candidato previo | Se conserva como comparador; el mapa primario de trabajo actual es el mapa de dominio actualizado | `previous_experiment/README.md`; definición nueva en Partes 3 y 4 |
| El dataset tiene 3,276 filas y desbalance 61/39 | Se vuelve a verificar directamente contra el CSV y se registra en `data_contract.json` | `water_potability.csv`; corroboración histórica en `previous_experiment/README.md` |
| El snapshot del dataset tiene SHA-256 `904004...b2bf` | Control de integridad y reproducibilidad | `water_potability.csv`; `data_contract.json`; referencia histórica en `previous_experiment/README.md` |

Cuando una decisión heredada entra en conflicto con los resultados de la generación actual, prevalecen los notebooks y artefactos actuales.

---

## 11. Criterios de comparación

La comparación SVM–QSVM debe respetar las siguientes condiciones:

- mismas 80 filas;
- mismas cinco variables;
- mismos cinco folds;
- mismo target;
- transformaciones ajustadas dentro de cada fold;
- selección de `C` sin observar el holdout completo;
- reporte de F1, specificity, balanced accuracy y MCC;
- conservación del kernel bruto antes de cualquier corrección numérica;
- separación clara entre simulación exacta, shots simulados, ruido proxy y emulación física.

La nueva generación no busca afirmar ventaja cuántica por construcción. El objetivo es determinar si alguna geometría cuántica ofrece una mejora estable y reproducible bajo una comparación controlada.

---

## 12. Limitaciones

- El dataset no incluye información geográfica, temporal o regulatoria suficiente para certificar potabilidad.
- El modelo es un benchmark académico, no un sistema de decisión sanitaria.
- La imputación por clase depende de la etiqueta verdadera y requiere rediseño para despliegue.
- El subconjunto de 80 filas fue seleccionado por cobertura del `decision_score`, no mediante muestreo poblacional aleatorio.
- Los 80 casos se usan en validación cruzada; no contienen un test final independiente adicional.
- Los pesos del mapa de dominio son heurísticos y no equivalen a coeficientes químicos o regulatorios.
- Las métricas sobre 80 filas tienen alta varianza y no representan por sí solas el desempeño sobre el dataset completo.
- El ruido proxy de Partes 3 y 4 no sustituye el modelo físico de un backend H2.
- Los kernels exactos basados en statevectors no representan todavía el costo ni la incertidumbre de una ejecución con shots.
- No existe todavía un resultado cuántico autoritativo de la generación actual.

---

## 13. Registro de fuentes

| ID | Fuente | Contenido utilizado |
|---|---|---|
| `S1` | `Hacia_el_agua_limpia_partes_1_2_SVM_RBF.ipynb` | contrato de datos, protocolo, métricas, threshold, subconjunto de 80 filas y artefactos |
| `S2` | `artifacts_v1/` | archivos reproducibles generados por Partes 1 y 2 |
| `S3` | `Hacia_el_agua_limpia_partes_3_4_pytket_kernels.ipynb` | mapas, circuitos, kernels, métricas previstas, recursos y manifiesto |
| `S4` | `water_potability.csv` | datos originales y hash del snapshot |
| `S5` | `previous_experiment/README.md` | únicamente decisiones heredadas y datos históricos citados en la sección 10 |


