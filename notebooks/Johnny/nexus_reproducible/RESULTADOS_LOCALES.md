# Resultados locales antes de Nexus

Estos resultados validan el pipeline reproducible y fijan la referencia exacta que luego debe
compararse contra los counts de Nexus. No son resultados de `H2-1LE`, `H2-Emulator` ni evidencia
de ventaja cuantica.

## Ganador historico capturado por familia

| Familia | Mapa congelado | C | F1 CV del artifact de seleccion |
|---|---|---:|---:|
| custom water-domain | `custom_water_domain_r1_robust` | 10.0 | 0.6230 |
| ZZFeatureMap | `zz_ring_r2_robust` | 0.1 | 0.6072 |
| PauliFeatureMap | `pauli_z_zz_linear_r1_robust` | 1.0 | 0.5768 |

La fuente es `last_version/artifacts_v3_4/map_cv_best_by_map.csv`. Esos valores solo explican
por que se congelaron los candidatos; no se mezclan con las metricas nuevas.

## Comparacion exacta con preprocesamiento corregido

Todos los modelos usan las mismas filas, cinco features, folds y preprocesador externo. La
SVM-RBF usa `C=10`, `gamma=0.01`, tambien congelados desde el artifact historico. La grilla
clasica completa se conserva como auditoria, sin reemplazar esa fila en la comparacion.

| n | Modelo/mapa | F1 medio | Balanced accuracy | MCC |
|---:|---|---:|---:|---:|
| 16 | SVM-RBF congelada | 0.6667 | 0.7500 | 0.5155 |
| 16 | `custom_water_domain_r1_robust` | 0.2667 | 0.4500 | -0.0845 |
| 16 | `zz_ring_r2_robust` | 0.6600 | 0.6000 | 0.2309 |
| 16 | `pauli_z_zz_linear_r1_robust` | 0.2667 | 0.4000 | -0.2000 |
| 64 | SVM-RBF congelada | 0.3741 | 0.4048 | -0.1976 |
| 64 | `custom_water_domain_r1_robust` | 0.5231 | 0.5833 | 0.1432 |
| 64 | `zz_ring_r2_robust` | 0.4850 | 0.5167 | 0.0188 |
| 64 | `pauli_z_zz_linear_r1_robust` | 0.4205 | 0.5024 | -0.0041 |

El perfil de 16 es un piloto de integracion y tiene folds muy pequenos; no debe usarse para
reordenar los mapas. En 64, el mejor resultado exacto entre los tres candidatos congelados es
`custom_water_domain_r1_robust`. Esta observacion tampoco cambia la preregistracion para Nexus:
se deben ejecutar y reportar los tres.

## Carga prevista

| n | Pares por mapa | Jobs execute totales (3 repeticiones) | Shots totales, tres mapas |
|---:|---:|---:|---:|
| 16 | 120 | 9 | 4,423,680 |
| 64 | 2,016 | 63 | 74,317,824 |

El costeo oficial debe ejecutarse y revisarse antes de enviar jobs. El notebook exige una frase
de reconocimiento explicita y usa checkpoints para evitar reenvios accidentales.

## Limite de interpretacion

Las 64 filas son parte del pool de 80 empleado para la seleccion historica. Por tanto, esta es
una comparacion condicionada a candidatos ya seleccionados y una referencia exacto-vs-Nexus;
no es un holdout independiente ni una estimacion insesgada de generalizacion. Los resultados
detallados y predicciones OOF estan en `artifacts_nexus/`.
