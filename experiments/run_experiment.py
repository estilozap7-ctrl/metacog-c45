"""
=========================================================
MetaCog-C45 Experimental Framework
run_experiment.py — Punto de entrada principal (Optimizado)
=========================================================

USO:
    python run_experiment.py [--batch 1|2|3|all] [--n_jobs N] [--master_seed S]
                             [--theta_accept F] [--theta_reject F]
                             [--alpha_0 F] [--beta_0 F] [--gamma F]
                             [--lambda_1 F] [--lambda_2 F]

DESCRIPCIÓN:
    Orquesta la ejecución del protocolo 5×10-CV completo
    sobre los 30 datasets del Protocolo Experimental v2.0.
    Divide la ejecución en 3 lotes (batches) de 10 datasets.
    Genera backups automáticos tras cada lote completado.
    Permite checkpoints para reanudar si es interrumpido.
=========================================================
"""

import os
import sys
import shutil
import argparse
import hashlib
import pickle
import json
import datetime
import time
import pandas as pd

# --- Path setup ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from experiments.pipeline.run_dataset import run_dataset
from experiments.pipeline.statistics import run_statistical_suite
from experiments.pipeline.report_generator import generate_reports
from experiments.pipeline.plot_generator import generate_plots

DATA_DIR    = os.path.join(ROOT, 'experiments', 'data')
RESULTS_DIR = os.path.join(ROOT, 'experiments', 'results')

CHECKPOINT_PATH = os.path.join(RESULTS_DIR, 'checkpoint.json')
SUMMARY_PATH    = os.path.join(RESULTS_DIR, 'summary.csv')

# -------------------------------------------------------
# Utilidades de Checkpoint e Integridad
# -------------------------------------------------------

def list_datasets():
    """Retorna la lista ordenada de .pkl en DATA_DIR."""
    files = sorted(
        [f for f in os.listdir(DATA_DIR) if f.endswith('.pkl')]
    )
    if not files:
        print("ERROR: No hay datasets en experiments/data/. "
              "Ejecuta primero: python experiments/setup_datasets.py")
        sys.exit(1)
    return files


def compute_checksum(path):
    """SHA-256 del archivo para verificar integridad."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_batch(lote_dir):
    """Verifica integridad de los archivos de resultados."""
    required = ['raw_results.csv', 'raw_results.json', 'stats_results.csv']
    for fname in required:
        fpath = os.path.join(lote_dir, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Archivo faltante: {fpath}")
        size = os.path.getsize(fpath)
        if size == 0:
            raise ValueError(f"Archivo vacío: {fpath}")
    print(f"  [OK] Integridad verificada en {lote_dir}")


def backup_batch(lote_dir, lote_num):
    """Crea una copia de respaldo del lote."""
    backup_dir = lote_dir + '_backup'
    if os.path.exists(backup_dir):
        shutil.rmtree(backup_dir)
    shutil.copytree(lote_dir, backup_dir)
    print(f"  [OK] Backup creado: {backup_dir}")


def load_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        try:
            with open(CHECKPOINT_PATH, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {"completed_datasets": {}}


def save_checkpoint(checkpoint):
    with open(CHECKPOINT_PATH, 'w') as f:
        json.dump(checkpoint, f, indent=2)


def log_summary_incremental(dataset_name, lote_num, elapsed, checksum):
    timestamp = datetime.datetime.now().isoformat()
    row = pd.DataFrame([{
        'dataset': dataset_name,
        'lote': lote_num,
        'elapsed_seconds': round(elapsed, 2),
        'timestamp': timestamp,
        'checksum': checksum
    }])
    header = not os.path.exists(SUMMARY_PATH)
    row.to_csv(SUMMARY_PATH, mode='a', index=False, header=header)


# -------------------------------------------------------
# Ejecución de un lote
# -------------------------------------------------------

def run_lote(lote_num, dataset_files, n_jobs=-1, master_seed=42, metacog_params=None):
    print(f"\n{'='*60}")
    print(f"  LOTE {lote_num}  — {len(dataset_files)} datasets")
    print(f"{'='*60}")

    lote_dir    = os.path.join(RESULTS_DIR, f'lote{lote_num}')
    raw_csv     = os.path.join(lote_dir, 'raw_results.csv')
    plots_dir   = os.path.join(lote_dir, 'plots')
    os.makedirs(lote_dir,  exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    checkpoint = load_checkpoint()
    all_frames = []
    
    first_write = not os.path.exists(raw_csv)

    for idx, fname in enumerate(dataset_files, 1):
        path = os.path.join(DATA_DIR, fname)
        name = fname.replace('.pkl', '')
        checksum = compute_checksum(path)

        # Checkpoint: reanudar automáticamente
        if name in checkpoint.get("completed_datasets", {}):
            saved_chk = checkpoint["completed_datasets"][name]
            if saved_chk.get("checksum") == checksum and os.path.exists(raw_csv):
                df_existing = pd.read_csv(raw_csv)
                df_ds = df_existing[df_existing['dataset'] == name]
                if not df_ds.empty:
                    print(f"  [CHECKPOINT] {name:<20} cargado desde caché incremental.")
                    all_frames.append(df_ds)
                    first_write = False
                    continue

        t_start = time.time()
        df = run_dataset(
            path,
            n_jobs=n_jobs,
            master_seed=master_seed,
            lote_num=lote_num,
            total_lotes=3,
            dataset_idx=idx,
            total_datasets=len(dataset_files),
            metacog_params=metacog_params
        )
        elapsed = time.time() - t_start
        all_frames.append(df)

        # Escritura incremental del dataset completo
        df.to_csv(
            raw_csv,
            mode='a' if not first_write else 'w',
            index=False,
            header=first_write
        )
        first_write = False
        print(f"  [SAVED] {fname} escrito en raw_results.csv")

        # Guardar en checkpoint.json
        checkpoint["completed_datasets"][name] = {
            "timestamp": datetime.datetime.now().isoformat(),
            "checksum": checksum,
            "lote": lote_num,
            "elapsed_seconds": round(elapsed, 2)
        }
        save_checkpoint(checkpoint)

        # Registro incremental en summary.csv
        log_summary_incremental(name, lote_num, elapsed, checksum)

    df_lote = pd.concat(all_frames, ignore_index=True)

    # Estadísticas
    df_stats = run_statistical_suite(df_lote)

    # Reportes completos (JSON, MD, LaTeX + CSV ya escrito arriba)
    generate_reports(df_lote, df_stats, lote_dir)

    # Gráficos
    generate_plots(df_lote, plots_dir)

    # Verificación de integridad
    verify_batch(lote_dir)

    # Backup
    backup_batch(lote_dir, lote_num)

    print(f"\n  Lote {lote_num} COMPLETADO. Resultados en: {lote_dir}\n")
    return df_lote, df_stats


# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MetaCog-C45 — Campaña Experimental Oficial"
    )
    parser.add_argument(
        '--batch',
        choices=['1', '2', '3', 'all'],
        default='all',
        help="Lote a ejecutar: 1, 2, 3, o 'all' (default: all)"
    )
    parser.add_argument(
        '--n_jobs',
        type=int,
        default=max(1, os.cpu_count() - 1),
        help="Número de hilos/procesos en paralelo (default: CPU_count - 1)"
    )
    parser.add_argument(
        '--master_seed',
        type=int,
        default=42,
        help="Semilla maestra del protocolo (default: 42)"
    )
    parser.add_argument(
        '--theta_accept',
        type=float,
        default=0.5,
        help="Umbral de aceptación del SCS (default: 0.5)"
    )
    parser.add_argument(
        '--theta_reject',
        type=float,
        default=0.2,
        help="Umbral de rechazo/colapso del SCS (default: 0.2)"
    )
    parser.add_argument(
        '--alpha_0',
        type=float,
        default=1.0,
        help="Elasticidad base del Gain Ratio (default: 1.0)"
    )
    parser.add_argument(
        '--beta_0',
        type=float,
        default=1.0,
        help="Elasticidad base del Node Stability (default: 1.0)"
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.5,
        help="Exponente del factor competitivo (1-CI)^gamma (default: 0.5)"
    )
    parser.add_argument(
        '--lambda_1',
        type=float,
        default=1.0,
        help="Peso del Threshold Survival en el SCS (default: 1.0)"
    )
    parser.add_argument(
        '--lambda_2',
        type=float,
        default=1.0,
        help="Peso del Competitive Index en el SCS (default: 1.0)"
    )
    args = parser.parse_args()

    # ——— Construir diccionario de hiperparámetros metacognitivos ———
    metacog_params = {
        'theta_accept': args.theta_accept,
        'theta_reject': args.theta_reject,
        'alpha_0':      args.alpha_0,
        'beta_0':       args.beta_0,
        'gamma':        args.gamma,
        'lambda_1':     args.lambda_1,
        'lambda_2':     args.lambda_2,
    }

    print("="*60)
    print("  MetaCog-C45 — CAMPAÑA EXPERIMENTAL OFICIAL (Optimizado)")
    print(f"  Parámetros base: n_jobs={args.n_jobs} | master_seed={args.master_seed}")
    print("  Decision Core (metacog_params):")
    for k, v in metacog_params.items():
        print(f"    {k} = {v}")
    print("="*60)

    dataset_files = list_datasets()
    total = len(dataset_files)
    print(f"  Datasets encontrados: {total}")

    # Dividir en 3 lotes
    lote1 = dataset_files[0:10]
    lote2 = dataset_files[10:20]
    lote3 = dataset_files[20:30]

    all_lotes = {'1': lote1, '2': lote2, '3': lote3}

    os.makedirs(RESULTS_DIR, exist_ok=True)

    lotes_to_run = ['1', '2', '3'] if args.batch == 'all' else [args.batch]

    all_results = []
    all_stats   = []

    for lote_key in lotes_to_run:
        files = all_lotes[lote_key]
        if not files:
            print(f"  [SKIP] Lote {lote_key}: sin datasets.")
            continue
        df_lote, df_stats = run_lote(
            int(lote_key),
            files,
            n_jobs=args.n_jobs,
            master_seed=args.master_seed,
            metacog_params=metacog_params
        )
        all_results.append(df_lote)
        all_stats.append(df_stats)

    # Paquete final consolidado (solo si corrieron todos los lotes)
    if args.batch == 'all':
        print("\n" + "="*60)
        print("  GENERANDO EXPERIMENTAL RESULTS PACKAGE CONSOLIDADO")
        print("="*60)
        df_all   = pd.concat(all_results, ignore_index=True)
        df_stats_all = pd.concat(all_stats, ignore_index=True)
        package_dir = os.path.join(RESULTS_DIR, 'package')
        generate_reports(df_all, df_stats_all, package_dir)
        generate_plots(df_all, os.path.join(package_dir, 'plots'))
        print(f"\n  PACKAGE COMPLETO: {package_dir}")

    print("\n" + "="*60)
    print("  EJECUCIÓN FINALIZADA")
    print("="*60)


if __name__ == '__main__':
    main()
