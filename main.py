"""Punto de entrada unico del proyecto TuQanes - Challenge 2 (agua potable).

Reproduce, en un solo flujo, cada figura y cifra reportada:

    python main.py                 # ejecuta todo y consolida en artifacts/
    python main.py --stage quantum # solo la etapa cuantica
    python main.py --stage classical
    python main.py --stage nexus   # resumen del paquete Nexus (NO lo ejecuta)

Etapas:
  * classical : consolida el baseline SVM-RBF congelado (chem5_v01) y sus figuras.
  * quantum   : recomputa la QSVM (kernel precomputado) desde los kernels exactos
                cacheados, sobre los folds congelados, y reproduce el ranking,
                la geometria y las ablaciones.
  * figures   : regenera todas las figuras (con barras de error).
  * report    : tabla maestra clasico-vs-cuantico + resumen en Markdown.
  * nexus     : resume la fase Nexus/H2 (modular, no ejecutada).

La fase Nexus se mantiene autocontenida en
``notebooks/Johnny/nexus_reproducible/TuQanes_Package_Nexus`` y NO se ejecuta aqui
porque consume cuota de Quantinuum; solo se resume su evidencia local exacta.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Hace importable ``src/`` sin instalacion previa.
SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tuqanes import classical, config, io_utils, nexus, plots, quantum, report  # noqa: E402


def _banner(text: str) -> None:
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)


def _ensure_dirs() -> None:
    config.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    for directory in config.ALL_OUTPUT_DIRS:
        directory.mkdir(parents=True, exist_ok=True)


def check_inputs() -> None:
    """Verifica el dataset canonico y la presencia de los kernels cacheados."""
    status = io_utils.verify_dataset()
    if not status["present"]:
        print("[aviso] No se encontro data/water_potability.csv (opcional para este flujo).")
    elif not status["matches"]:
        print(f"[aviso] El hash del dataset ({status['hash'][:12]}...) no coincide con "
              "chem5_v01; las cifras clasicas se toman de los artefactos congelados.")
    else:
        print("[ok] Dataset canonico verificado (SHA-256 chem5_v01).")

    maps = io_utils.available_maps()
    if not maps:
        raise SystemExit(
            "No se encontraron kernels cacheados en "
            f"{config.KERNELS_DIR}. No se puede reproducir la etapa cuantica."
        )
    print(f"[ok] {len(maps)} kernels cuanticos exactos disponibles.")


def run_all() -> None:
    _ensure_dirs()
    check_inputs()

    _banner("1/4  Baseline clasico (SVM-RBF, chem5_v01)")
    classical_results = classical.run()
    cvb = classical_results["cv_best"].iloc[0]
    hold = classical_results["holdout"].iloc[0]
    print(f"  CV completo   : F1={cvb['f1_mean']:.4f}  bal_acc={cvb['balanced_accuracy_mean']:.4f}"
          f"  MCC={cvb['mcc_mean']:.4f}")
    print(f"  Holdout (656) : F1={hold['f1']:.4f}  bal_acc={hold['balanced_accuracy']:.4f}"
          f"  MCC={hold['mcc']:.4f}")

    _banner("2/4  QSVM cuantica (cifras reportadas + verificacion independiente)")
    quantum_results = quantum.run()
    print("  Ranking por F1 (CV, 80 filas) - cifras reportadas congeladas:")
    for _, r in quantum_results["best"].sort_values("f1_mean", ascending=False).iterrows():
        print(f"    {r['map']:<42} F1={r['f1_mean']:.4f} ± {r['f1_std']:.4f}  (C={r['C']})")
    print(f"  Verificacion independiente (kernel precomputado): diff. max. de F1 "
          f"vs congelado = {quantum_results['recompute_max_f1_diff']:.4f}")

    _banner("3/4  Figuras (con barras de error)")
    figures = plots.run(quantum_results, classical_results)
    print(f"  {len(figures)} figuras escritas en {config.ART_FIGURES}")

    _banner("4/4  Fase Nexus (modular, NO ejecutada) + consolidacion")
    nexus_results = nexus.run()
    print(f"  Paquete Nexus presente: {bool(nexus_results['status']['present'].iloc[0])} "
          "(no ejecutado por el pipeline)")
    report_results = report.run(classical_results, quantum_results, nexus_results)

    _banner("LISTO")
    print(f"Artefactos consolidados en: {config.ARTIFACTS}")
    print(f"Resumen: {config.ARTIFACTS / 'RESULTS_SUMMARY.md'}")
    print(f"Tabla maestra: {config.ART_COMPARISON / 'master_metrics.csv'}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Pipeline reproducible TuQanes - Challenge 2")
    parser.add_argument(
        "--stage",
        choices=["all", "classical", "quantum", "figures", "report", "nexus"],
        default="all",
        help="Etapa a ejecutar (por defecto: all).",
    )
    args = parser.parse_args(argv)

    if args.stage == "all":
        run_all()
        return

    _ensure_dirs()
    check_inputs()
    if args.stage == "classical":
        classical.run()
    elif args.stage == "quantum":
        quantum.run()
    elif args.stage == "nexus":
        nexus.run()
    elif args.stage in {"figures", "report"}:
        classical_results = classical.run()
        quantum_results = quantum.run()
        plots.run(quantum_results, classical_results)
        if args.stage == "report":
            report.run(classical_results, quantum_results, nexus.run())
    print(f"\nEtapa '{args.stage}' completada. Salida en {config.ARTIFACTS}")


if __name__ == "__main__":
    main()
