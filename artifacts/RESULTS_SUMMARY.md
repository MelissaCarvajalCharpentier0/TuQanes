# Resumen consolidado de resultados (TuQanes)

Generado por `main.py`. Todas las cifras se reproducen desde datos y kernels
congelados. La fase Nexus/H2 NO se ejecuta desde este flujo.

## Tabla maestra

| stage | model | n | f1_mean | f1_std | balanced_accuracy_mean | mcc_mean |
| --- | --- | --- | --- | --- | --- | --- |
| clasico | SVM-RBF chem5_v01 (dataset completo, CV) | 2620 | 0.5573 | 0.0275 | 0.627 | 0.2502 |
| clasico | SVM-RBF chem5_v01 (holdout 656) | 656 | 0.5768 | nan | 0.6458 | 0.2878 |
| clasico | SVM-RBF (subconjunto 80, C=10 gamma=0.01) | 80 | 0.5773 | 0.1335 | 0.5375 | 0.0748 |
| cuantico | QSVM custom_water_domain_r1_robust (C=10.0) | 80 | 0.623 | 0.1538 | 0.625 | 0.2545 |
| cuantico | QSVM zz_ring_r2_robust (C=0.1) | 80 | 0.6072 | 0.2059 | 0.625 | 0.2539 |
| cuantico | QSVM custom_water_domain_r2_reupload_robust (C=1.0) | 80 | 0.6015 | 0.1502 | 0.6 | 0.2028 |
| cuantico | QSVM pauli_z_zz_linear_r1_robust (C=1.0) | 80 | 0.5768 | 0.1537 | 0.5875 | 0.1776 |
| cuantico | QSVM zz_linear_r1_robust (C=1.0) | 80 | 0.5768 | 0.1537 | 0.5875 | 0.1776 |
| cuantico | QSVM pauli_z_zz_ring_r1_robust (C=1.0) | 80 | 0.5691 | 0.1654 | 0.5875 | 0.1782 |
| cuantico | QSVM zz_ring_r1_robust (C=1.0) | 80 | 0.5691 | 0.1654 | 0.5875 | 0.1782 |
| cuantico | QSVM zz_full_r1_robust (C=1.0) | 80 | 0.5689 | 0.175 | 0.5625 | 0.1297 |
| cuantico | QSVM pauli_xz_xxzz_linear_r1_robust (C=10.0) | 80 | 0.5588 | 0.1175 | 0.55 | 0.0982 |
| cuantico | QSVM pauli_z_zz_ring_r1_minmax (C=1.0) | 80 | 0.4957 | 0.2416 | 0.5125 | 0.0105 |

## Lectura

- Mejor mapa cuantico por F1 (CV, 80 filas): **custom_water_domain_r1_robust** con F1 = 0.6230 ± 0.1538 (C=10.0).
- El baseline clasico completo alcanza mayor F1/MCC en holdout independiente; la ventaja cuantica **no** esta demostrada.
- La fase Nexus queda empaquetada y modular en `notebooks/Johnny/nexus_reproducible/TuQanes_Package_Nexus` (no ejecutada aqui).
