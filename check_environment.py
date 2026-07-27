"""
=========================================================
MetaCog-C45 Experimental Framework
check_environment.py — Environment Validation Script
=========================================================

USO:
    python check_environment.py

Verifica que el entorno de ejecución cumple todos los
requisitos del Protocolo Experimental v2.0.

ESTADO POSIBLE:
    [PASS] — Requerimiento cumplido
    [FAIL] — Requerimiento NO cumplido (experimento BLOQUEADO)
    [WARN] — Versión subóptima (el experimento puede continuar)
=========================================================
"""

import sys
import platform
import importlib
import importlib.metadata as meta

# -------------------------------------------------------
# Requisitos del Protocolo Experimental v2.0
# -------------------------------------------------------

REQUIRED = {
    # pip_name        : (import_name,  min_version,  critical)
    'numpy':           ('numpy',       '1.23',        True),
    'pandas':          ('pandas',      '1.5',         True),
    'scipy':           ('scipy',       '1.9',         True),
    'scikit-learn':    ('sklearn',     '1.2',         True),
    'matplotlib':      ('matplotlib',  '3.6',         True),
    'seaborn':         ('seaborn',     '0.12',        True),
    'openml':          ('openml',      '0.14',        True),
    'joblib':          ('joblib',      '1.2',         True),
    'tqdm':            ('tqdm',        '4.64',        False),
    'tabulate':        ('tabulate',    '0.9',         False),
    'fpdf2':           ('fpdf',        '2.7',         False),
    'python-pptx':     ('pptx',        '0.6',         False),
}

FRAMEWORK_MODULES = [
    'core.classifier',
    'core.tree_builder',
    'core.tree_utils',
    'core.decision_node',
    'core.metacognition',
    'experiments.pipeline.run_fold',
    'experiments.pipeline.run_dataset',
    'experiments.pipeline.metrics',
    'experiments.pipeline.statistics',
    'experiments.pipeline.report_generator',
    'experiments.pipeline.plot_generator',
]

MIN_PYTHON = (3, 10)

PASS  = '\033[92m[PASS]\033[0m'
FAIL  = '\033[91m[FAIL]\033[0m'
WARN  = '\033[93m[WARN]\033[0m'
INFO  = '\033[94m[INFO]\033[0m'


def version_tuple(v_str):
    """Convierte '1.23.4' en (1, 23, 4)."""
    try:
        return tuple(int(x) for x in str(v_str).split('.')[:3])
    except ValueError:
        return (0,)


def check_python():
    print("\n── Python Runtime ─────────────────────────────────")
    current = sys.version_info[:2]
    ok = current >= MIN_PYTHON
    status = PASS if ok else FAIL
    print(f"  {status} Python {sys.version.split()[0]}  "
          f"(requerido >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]})")
    print(f"  {INFO} Plataforma: {platform.platform()}")
    return ok


def check_dependencies():
    print("\n── Dependencias Externas ──────────────────────────")
    all_ok = True
    for pip_name, (import_name, min_ver, critical) in REQUIRED.items():
        # Comprobar instalación
        try:
            dist_key = pip_name.replace('-', '_')
            installed_ver = meta.version(dist_key)
        except meta.PackageNotFoundError:
            installed_ver = None

        if installed_ver is None:
            status = FAIL if critical else WARN
            print(f"  {status} {pip_name:<20} NOT INSTALLED  "
                  f"(requerido >= {min_ver})")
            if critical:
                all_ok = False
            continue

        # Comprobar versión mínima
        if version_tuple(installed_ver) >= version_tuple(min_ver):
            status = PASS
        else:
            status = FAIL if critical else WARN
            if critical:
                all_ok = False

        print(f"  {status} {pip_name:<20} {installed_ver:<12}  "
              f"(requerido >= {min_ver})")

    return all_ok


def check_framework_modules():
    print("\n── Módulos del Framework MetaCog-C45 ──────────────")
    all_ok = True
    for module_path in FRAMEWORK_MODULES:
        try:
            importlib.import_module(module_path)
            print(f"  {PASS} {module_path}")
        except ImportError as e:
            print(f"  {FAIL} {module_path}  — {e}")
            all_ok = False
    return all_ok


def check_data_directory():
    import os
    print("\n── Datos Experimentales ───────────────────────────")
    data_dir = os.path.join(os.path.dirname(__file__), 'experiments', 'data')
    if not os.path.exists(data_dir):
        print(f"  {WARN} experiments/data/ no existe.")
        print(f"        Ejecutar: python experiments/setup_datasets.py")
        return False
    pkls = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
    if len(pkls) == 0:
        print(f"  {WARN} No hay datasets preparados.")
        print(f"        Ejecutar: python experiments/setup_datasets.py")
        return False
    print(f"  {PASS} {len(pkls)} dataset(s) encontrados en experiments/data/")
    return True


def main():
    print("=" * 56)
    print("  MetaCog-C45 — Environment Verification")
    print("  Protocolo Experimental v2.0")
    print("=" * 56)

    r1 = check_python()
    r2 = check_dependencies()
    r3 = check_framework_modules()
    r4 = check_data_directory()

    print("\n" + "=" * 56)
    if r1 and r2 and r3:
        print("  \033[92mINSTALLATION VERIFIED\033[0m")
        if not r4:
            print("  \033[93mACCIÓN PENDIENTE: Ejecutar setup_datasets.py\033[0m")
    else:
        print("  \033[91mINSTALLATION FAILED — Revisar errores arriba\033[0m")
    print("=" * 56)


if __name__ == '__main__':
    main()
