"""
=========================================================
MetaCog-C45  —  CLI Oficial del Framework
metacog_c45/cli.py
=========================================================

Interfaz de línea de comandos (CLI) oficial de MetaCog-C45.

Registrada en pyproject.toml como punto de entrada:
    [project.scripts]
    metacog = "metacog_c45.cli:main"

Uso tras instalación con pip:
    metacog --dataset datos.csv

Uso directo sin instalación:
    python -m metacog_c45 --dataset datos.csv

Compatibilidad con la interfaz anterior:
    python ejecutar_con_dataset.py   (sigue funcionando igual)

Esta CLI no reimplementa lógica: delega la ejecución
completa a ejecutar_con_dataset.py, que orquesta el
entrenamiento MetaCog-C45 y la generación del reporte HTML.
=========================================================
"""

import argparse
import os
import subprocess
import sys

from metacog_c45.version import __version__


# ── Directorio raíz del proyecto ──────────────────────
# Compatible con ejecución directa y con pip install
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)


def _build_parser() -> argparse.ArgumentParser:
    """Construye y retorna el parser de argumentos de la CLI."""
    parser = argparse.ArgumentParser(
        prog="metacog",
        description=(
            "MetaCog-C45 — Metacognitive Decision Tree Framework\n"
            "Entrena un árbol de decisión C4.5 con capa metacognitiva\n"
            "y genera un reporte HTML interactivo con todos los resultados."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ejemplos:\n"
            "  metacog --dataset datos.csv\n"
            "  metacog --dataset datos.csv --target clase\n"
            "  metacog --dataset datos.csv --target clase --max_depth 5\n"
            "  metacog --version"
        ),
    )

    parser.add_argument(
        "--dataset",
        metavar="RUTA",
        type=str,
        default=None,
        help=(
            "Ruta al archivo de dataset (.csv o .xlsx). "
            "Si no se especifica, usa el dataset de demostración "
            "(datasets/customer_churn.csv)."
        ),
    )

    parser.add_argument(
        "--target",
        metavar="COLUMNA",
        type=str,
        default=None,
        help=(
            "Nombre de la columna objetivo (variable dependiente). "
            "Si no se especifica, usa la última columna del dataset."
        ),
    )

    parser.add_argument(
        "--max_depth",
        metavar="N",
        type=int,
        default=None,
        help="Profundidad máxima del árbol de decisión (por defecto: sin límite).",
    )

    parser.add_argument(
        "--min_samples_split",
        metavar="N",
        type=int,
        default=None,
        help="Mínimo de muestras requeridas para dividir un nodo (por defecto: 2).",
    )

    parser.add_argument(
        "--no_browser",
        action="store_true",
        default=False,
        help="Genera el reporte HTML sin abrirlo automáticamente en el navegador.",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"MetaCog-C45 {__version__}",
        help="Muestra la versión del framework y termina.",
    )

    return parser


def main() -> None:
    """
    Punto de entrada oficial del CLI de MetaCog-C45.

    Registrado en pyproject.toml:
        [project.scripts]
        metacog = "metacog_c45.cli:main"

    También invocable como módulo:
        python -m metacog_c45 --dataset datos.csv
    """
    parser = _build_parser()
    args = parser.parse_args()

    # ── Construir los argumentos para el script de ejecución ──
    # ejecutar_con_dataset.py es la interfaz completa existente.
    # La CLI delega en él para evitar duplicación de lógica.
    script_path = os.path.join(_ROOT, "ejecutar_con_dataset.py")

    if not os.path.exists(script_path):
        print(
            f"[ERROR] No se encontró el script principal del framework:\n"
            f"        {script_path}\n\n"
            f"Asegúrese de que MetaCog-C45 esté correctamente instalado "
            f"o ejecute desde la raíz del repositorio."
        )
        sys.exit(1)

    # Construir comando con los argumentos recibidos
    cmd = [sys.executable, script_path]

    if args.dataset:
        cmd += ["--dataset", args.dataset]
    if args.target:
        cmd += ["--target", args.target]
    if args.max_depth is not None:
        cmd += ["--max_depth", str(args.max_depth)]
    if args.min_samples_split is not None:
        cmd += ["--min_samples_split", str(args.min_samples_split)]
    if args.no_browser:
        cmd += ["--no_browser"]

    # Delegar ejecución al script principal
    try:
        result = subprocess.run(cmd, check=False)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\n[INFO] Ejecución interrumpida por el usuario.")
        sys.exit(0)


if __name__ == "__main__":
    main()
