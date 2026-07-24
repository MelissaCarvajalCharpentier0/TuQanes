# TuQanes - QSVM reproducible en Quantinuum Nexus

Esta carpeta implementa una fase nueva y aislada para ejecutar en Nexus los tres mapas de
caracteristicas seleccionados en la generacion `chem5_v01`. No modifica los notebooks ni los
artifacts de `../last_version/`.

## Candidatos congelados

| Familia | Mapa | C |
|---|---|---:|
| custom water-domain | `custom_water_domain_r1_robust` | 10.0 |
| ZZFeatureMap | `zz_ring_r2_robust` | 0.1 |
| PauliFeatureMap | `pauli_z_zz_linear_r1_robust` | 1.0 |

Los candidatos y sus hiperparametros se verifican contra
`../last_version/artifacts_v3_4/map_cv_best_by_map.csv`. Nexus no vuelve a seleccionarlos y
ninguna repeticion se descarta por tener peores metricas.

## Archivos

- `Nexus_QSVM_top3_reproducible.ipynb`: punto de entrada unico.
- `nexus_qsvm.py`: funciones reutilizables, validaciones y checkpoints de Nexus.
- `requirements-nexus.txt`: dependencias adicionales para un entorno externo. Nexus Lab ya
  incluye normalmente `qnexus` y autenticacion silenciosa.
- `RESULTADOS_LOCALES.md`: lectura corta de las comparaciones de 16 y 64 muestras.
- `artifacts_nexus/<run_id>/`: resultados creados por cada ejecucion.

## Protocolo corregido

1. Las 80 filas originales se conservan como pool de seleccion y nunca se mezclan con el
   holdout externo.
2. El piloto de 16 y el conjunto de evaluacion de 64 se seleccionan por clase y fold mediante
   una prioridad hash determinista. La seleccion no observa resultados de ningun modelo y el
   piloto queda contenido en el conjunto de 64.
3. El preprocesador por defecto se ajusta con las filas del development training que estan
   fuera de las 80 filas cuanticas. Usa mediana global, no necesita la etiqueta de evaluacion y
   no observa las filas de validacion.
4. La SVM-RBF clasica y los tres kernels cuanticos usan exactamente las mismas filas, features
   y folds. La grilla clasica completa se conserva como auditoria, pero la comparacion usa
   `C=10`, `gamma=0.01`, congelados desde `quantum80_svm_cv_summary.csv`. Los valores `C`
   cuanticos tambien estan congelados desde la seleccion anterior.
5. Cada fidelidad se mide con `U(x_i)` seguido por `U(x_j)^dagger`; la probabilidad de
   `00000` estima `|<phi(x_j)|phi(x_i)>|^2`. La diagonal se fija analiticamente en uno.
6. Los counts crudos se guardan antes de reconstruir o diagnosticar el kernel. Un kernel
   indefinido nunca se proyecta silenciosamente a PSD.
7. Se ejecutan al menos tres repeticiones y se reportan todas mediante media y desviacion
   estandar. No existe una operacion para escoger la mejor repeticion.

El modo alternativo `class_median_external_pool` reproduce la imputacion por clase solicitada
en el reto, pero necesita la etiqueta de cada muestra para imputarla. Se conserva solamente
como auditoria de sensibilidad y no es el valor por defecto.

## Ejecucion recomendada

### 1. Validacion local, sin cuota

Abra el notebook y mantenga:

```python
subset_n=16
nexus_stage="local_only"
quota_acknowledgement=""
```

Ejecute todas las celdas. Se generan kernels exactos, una previsualizacion binomial claramente
marcada como local, comparacion SVM/QSVM, QASM representativo, manifiestos y el inventario de
circuitos. Ningun dato se transmite a Nexus.

### 2. Costeo deliberado

En Nexus Lab cambie la configuracion a:

```python
nexus_stage="cost"
quota_acknowledgement="I_ACCEPT_NEXUS_QUOTA_USAGE"
```

Esta fase carga y compila circuitos y ejecuta los jobs de estimacion de costo. Revise
`artifacts_nexus/<run_id>/nexus/cost_estimates.csv` antes de continuar. El API de costeo tambien
consume cuota de costeo.

### 3. Envio y recoleccion

Solo despues de aprobar los costos, use `nexus_stage="submit"`. El notebook reutiliza los
circuitos compilados y envia tres repeticiones independientes. Cuando terminen, cambie a
`nexus_stage="collect"` para descargar counts, reconstruir kernels y producir las metricas
agregadas.

Los objetos Nexus se guardan mediante `qnx.filesystem.save`, de modo que una sesion posterior
puede reanudar compilaciones, ejecuciones o recoleccion sin reenviar trabajo completado.

## Escalado

| Muestras | Pares por mapa | Compile jobs por mapa | Execute jobs por mapa (3 repeticiones) |
|---:|---:|---:|---:|
| 16 | 120 | 1 | 3 |
| 64 | 2,016 | 7 | 21 |

El limite usado es 300 programas por job. Primero debe completarse el piloto de 16 en
`H2-1LE`. Para la evaluacion de 64, cree un `run_id` nuevo y cambie `subset_n=64`. Una extension
con ruido debe usar tambien un `run_id` distinto y `backend_name="H2-Emulator"`; no debe
sobrescribir ni mezclar los counts noiseless.

## Notas metodologicas

- El piloto de 16 valida el circuito y la tuberia. Sus metricas no se usan para reordenar mapas.
- Las 64 filas pertenecen al mismo pool de 80 usado en la seleccion historica de mapas. Por
  ello esta fase compara candidatos condicionados a esa seleccion y exacto-vs-Nexus, pero no
  constituye un holdout independiente ni una estimacion insesgada de generalizacion.
- `pauli_z_zz_linear_r1_robust` coincide exactamente con `zz_linear_r1_robust` en la ejecucion
  anterior. Se mantiene para cubrir la familia Pauli, no como evidencia de una geometria nueva.
- Los resultados `local_binomial_preview` no son resultados de H2 o Nexus.
- No se hacen afirmaciones de ventaja cuantica.
