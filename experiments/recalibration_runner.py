"""
=========================================================
MetaCog-C45 — Recalibración Experimental (Estrategia A)
recalibration_runner.py
=========================================================

Protocolo incremental de recalibración del Decision Core.
Ejecuta 5×10-CV sobre el panel de 4 datasets de validación
y genera un informe técnico por fase.

USO:
    python recalibration_runner.py --phase A
    python recalibration_runner.py --phase B
    python recalibration_runner.py --phase C
    python recalibration_runner.py --phase D
    python recalibration_runner.py --phase A --artifacts_dir /ruta/personalizada

Por defecto los informes se guardan en:
    <raíz_proyecto>/experiments/results/recalibration/

Criterio de aceptación del Comité:
    0.20 ≤ Nodes_MetaCog / Nodes_Classic ≤ 0.80
=========================================================
"""

import os
import sys
import argparse
import pickle
import datetime
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from experiments.pipeline.run_dataset import run_dataset
from experiments.pipeline.statistics import cliffs_delta

# -------------------------------------------------------
# Directorios
# -------------------------------------------------------
DATA_DIR = os.path.join(ROOT, 'experiments', 'data')

# Directorio de artefactos por defecto (portable, relativo al proyecto).
# Puede sobreescribirse vía --artifacts_dir en la línea de comandos.
DEFAULT_ARTIFACTS_DIR = os.path.join(ROOT, 'experiments', 'results', 'recalibration')

# -------------------------------------------------------
# Panel de validación
# -------------------------------------------------------
PANEL_DATASETS = ['iris', 'wine', 'breast_cancer', 'adult']

# -------------------------------------------------------
# Configuración Baseline (pre-recalibración)
# -------------------------------------------------------
BASELINE_PARAMS = {
    'theta_accept': 0.50,
    'theta_reject': 0.20,
    'alpha_0':      1.0,
    'beta_0':       1.0,
    'gamma':        0.50,
    'lambda_1':     1.0,
    'lambda_2':     1.0,
}

# -------------------------------------------------------
# Configuraciones por Fase (acumulativas)
# -------------------------------------------------------
PHASE_CONFIGS = {
    'A': {
        **BASELINE_PARAMS,
        'theta_accept': 0.40,
        'theta_reject': 0.05,
    },
    'B': {
        **BASELINE_PARAMS,
        'theta_accept': 0.40,
        'theta_reject': 0.05,
        'gamma':        0.20,
    },
    'C': {
        **BASELINE_PARAMS,
        'theta_accept': 0.40,
        'theta_reject': 0.05,
        'gamma':        0.20,
        'alpha_0':      1.5,
        'beta_0':       1.5,
    },
    'D': {
        **BASELINE_PARAMS,
        'theta_accept': 0.40,
        'theta_reject': 0.05,
        'gamma':        0.20,
        'alpha_0':      1.5,
        'beta_0':       1.5,
        'lambda_1':     1.2,
        'lambda_2':     0.8,
    },
}

ACCEPTANCE_MIN = 0.20
ACCEPTANCE_MAX = 0.80


# -------------------------------------------------------
# Utilidades
# -------------------------------------------------------

def get_dataset_path(name):
    """Retorna la ruta al .pkl de un dataset por nombre."""
    path = os.path.join(DATA_DIR, f'{name}.pkl')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset no encontrado: {path}")
    return path


def compute_node_ratio(df):
    """
    Calcula el ratio Nodes_MetaCog / Nodes_Classic por fold y dataset.
    Retorna un DataFrame con columnas [dataset, fold, ratio].
    """
    rows = []
    for ds in df['dataset'].unique():
        df_ds = df[df['dataset'] == ds]
        df_classic = df_ds[df_ds['mode'] == 'classic'].set_index('fold')
        df_metacog = df_ds[df_ds['mode'] == 'metacog'].set_index('fold')
        common_folds = df_classic.index.intersection(df_metacog.index)
        for fold in common_folds:
            n_classic = df_classic.loc[fold, 'nodes']
            n_metacog = df_metacog.loc[fold, 'nodes']
            ratio = n_metacog / n_classic if n_classic > 0 else np.nan
            rows.append({'dataset': ds, 'fold': fold, 'ratio': ratio})
    return pd.DataFrame(rows)


def run_stat_comparison(df_base, df_phase, metric='mcc'):
    """
    Wilcoxon + Cliff's Delta para metacog baseline vs metacog fase.
    Compara los valores de metacog en baseline vs los de metacog en la fase.
    """
    results = []
    for ds in PANEL_DATASETS:
        vals_base  = df_base[(df_base['dataset'] == ds) & (df_base['mode'] == 'metacog')][metric].dropna().values
        vals_phase = df_phase[(df_phase['dataset'] == ds) & (df_phase['mode'] == 'metacog')][metric].dropna().values

        if len(vals_base) == 0 or len(vals_phase) == 0:
            continue

        # Wilcoxon (requiere misma longitud o usa signed-rank sobre diferencias)
        min_len = min(len(vals_base), len(vals_phase))
        try:
            _, p_wilcoxon = wilcoxon(vals_phase[:min_len], vals_base[:min_len])
        except ValueError:
            p_wilcoxon = 1.0

        cd = cliffs_delta(vals_phase, vals_base)
        results.append({
            'dataset':       ds,
            'metric':        metric,
            'n_base':        len(vals_base),
            'n_phase':       len(vals_phase),
            'mean_base':     np.nanmean(vals_base),
            'mean_phase':    np.nanmean(vals_phase),
            'delta':         np.nanmean(vals_phase) - np.nanmean(vals_base),
            'cliffs_delta':  cd,
            'wilcoxon_p':    p_wilcoxon,
            'significant':   p_wilcoxon < 0.05,
        })
    return pd.DataFrame(results)


def summarize_phase(df_phase, df_ratios):
    """
    Genera tabla resumen por dataset con todas las métricas requeridas.
    """
    rows = []
    for ds in PANEL_DATASETS:
        df_c = df_phase[(df_phase['dataset'] == ds) & (df_phase['mode'] == 'classic')]
        df_m = df_phase[(df_phase['dataset'] == ds) & (df_phase['mode'] == 'metacog')]
        ratios_ds = df_ratios[df_ratios['dataset'] == ds]['ratio']

        rows.append({
            'Dataset':                ds,
            'Acc_Classic':            round(df_c['accuracy'].mean(), 4),
            'Acc_MetaCog':            round(df_m['accuracy'].mean(), 4),
            'MCC_Classic':            round(df_c['mcc'].mean(), 4),
            'MCC_MetaCog':            round(df_m['mcc'].mean(), 4),
            'PR_AUC_Classic':         round(df_c['pr_auc'].mean(), 4) if 'pr_auc' in df_c else np.nan,
            'PR_AUC_MetaCog':         round(df_m['pr_auc'].mean(), 4) if 'pr_auc' in df_m else np.nan,
            'Nodes_Classic':          round(df_c['nodes'].mean(), 1),
            'Nodes_MetaCog':          round(df_m['nodes'].mean(), 1),
            'Depth_Classic':          round(df_c['depth'].mean(), 1),
            'Depth_MetaCog':          round(df_m['depth'].mean(), 1),
            'Ratio_mean':             round(ratios_ds.mean(), 4),
            'Ratio_min':              round(ratios_ds.min(), 4),
            'Ratio_max':              round(ratios_ds.max(), 4),
            'Criterion_OK':           ACCEPTANCE_MIN <= ratios_ds.mean() <= ACCEPTANCE_MAX,
        })
    return pd.DataFrame(rows)


# -------------------------------------------------------
# Ejecución del panel
# -------------------------------------------------------

def run_panel(params, label, master_seed=42, n_jobs=4):
    """
    Ejecuta el protocolo 5×10-CV sobre los 4 datasets del panel.
    Retorna el DataFrame consolidado de resultados.
    """
    all_frames = []
    print(f"\n{'='*60}")
    print(f"  EJECUTANDO PANEL — Fase {label}")
    print(f"  Hiperparámetros activos:")
    for k, v in params.items():
        print(f"    {k} = {v}")
    print(f"{'='*60}")

    for idx, ds_name in enumerate(PANEL_DATASETS, 1):
        path = get_dataset_path(ds_name)
        print(f"\n  [{idx}/{len(PANEL_DATASETS)}] {ds_name}...")
        df = run_dataset(
            path,
            n_jobs=n_jobs,
            master_seed=master_seed,
            lote_num=0,
            total_lotes=1,
            dataset_idx=idx,
            total_datasets=len(PANEL_DATASETS),
            metacog_params=params
        )
        all_frames.append(df)

    return pd.concat(all_frames, ignore_index=True)


# -------------------------------------------------------
# Generador de informe por fase
# -------------------------------------------------------

def generate_phase_report(phase_label, params, df_phase, df_baseline,
                           df_summary, df_ratios, df_stats_mcc,
                           df_stats_acc, criterion_global):
    """
    Genera RecalibrationPhase{X}Report.md en el directorio de artefactos.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fname = os.path.join(ARTIFACTS_DIR, f'RecalibrationPhase{phase_label}Report.md')

    phase_names = {
        'A': 'Fase A — Ajuste de Umbrales (theta_accept, theta_reject)',
        'B': 'Fase B — Ajuste del Factor Competitivo (gamma)',
        'C': 'Fase C — Ajuste de Elasticidades Base (alpha_0, beta_0)',
        'D': 'Fase D — Ajuste de Pesos del SCS (lambda_1, lambda_2)',
    }

    changed_params = {k: v for k, v in params.items() if v != BASELINE_PARAMS.get(k)}

    lines = []
    lines.append(f"# Informe de Recalibración — {phase_names.get(phase_label, phase_label)}")
    lines.append(f"\n**Fecha:** {timestamp}  ")
    lines.append(f"**Panel:** iris · wine · breast_cancer · adult  ")
    lines.append(f"**Protocolo:** 5×10-CV  ")
    lines.append(f"**Criterio de aceptación:** 0.20 ≤ Nodes_MetaCog/Nodes_Classic ≤ 0.80")

    lines.append("\n---\n")
    lines.append("## 1. Configuración de Hiperparámetros\n")
    lines.append("| Parámetro | Baseline | Esta Fase | Modificado |")
    lines.append("|:----------|:--------:|:---------:|:----------:|")
    for k, v_base in BASELINE_PARAMS.items():
        v_phase = params.get(k, v_base)
        modified = "**✓**" if k in changed_params else "—"
        lines.append(f"| `{k}` | {v_base} | {v_phase} | {modified} |")

    lines.append("\n---\n")
    lines.append("## 2. Métricas de Rendimiento por Dataset\n")
    lines.append("| Dataset | Acc_C | Acc_M | MCC_C | MCC_M | PR-AUC_C | PR-AUC_M | Nodes_C | Nodes_M | Depth_C | Depth_M |")
    lines.append("|:--------|:-----:|:-----:|:-----:|:-----:|:--------:|:--------:|:-------:|:-------:|:-------:|:-------:|")
    for _, row in df_summary.iterrows():
        pr_c = f"{row['PR_AUC_Classic']:.4f}" if not np.isnan(row['PR_AUC_Classic']) else "N/A"
        pr_m = f"{row['PR_AUC_MetaCog']:.4f}" if not np.isnan(row['PR_AUC_MetaCog']) else "N/A"
        lines.append(
            f"| {row['Dataset']} | {row['Acc_Classic']:.4f} | {row['Acc_MetaCog']:.4f} | "
            f"{row['MCC_Classic']:.4f} | {row['MCC_MetaCog']:.4f} | {pr_c} | {pr_m} | "
            f"{row['Nodes_Classic']:.0f} | {row['Nodes_MetaCog']:.0f} | "
            f"{row['Depth_Classic']:.1f} | {row['Depth_MetaCog']:.1f} |"
        )

    lines.append("\n---\n")
    lines.append("## 3. Ratio Nodes_MetaCog / Nodes_Classic\n")
    lines.append(f"**Criterio:** 0.20 ≤ ratio ≤ 0.80  ")
    lines.append(f"**Resultado global:** {'✅ CUMPLE' if criterion_global else '❌ NO CUMPLE'}\n")
    lines.append("| Dataset | Ratio_mean | Ratio_min | Ratio_max | Criterio |")
    lines.append("|:--------|:----------:|:---------:|:---------:|:--------:|")
    for _, row in df_summary.iterrows():
        ok = "✅" if row['Criterion_OK'] else "❌"
        lines.append(
            f"| {row['Dataset']} | {row['Ratio_mean']:.4f} | {row['Ratio_min']:.4f} | "
            f"{row['Ratio_max']:.4f} | {ok} |"
        )

    lines.append("\n---\n")
    lines.append("## 4. Análisis Estadístico — MCC (MetaCog Fase vs. MetaCog Baseline)\n")
    lines.append("| Dataset | Mean_Base | Mean_Fase | Δ | Cliff's δ | Wilcoxon p | Significativo |")
    lines.append("|:--------|:---------:|:---------:|:--:|:---------:|:----------:|:-------------:|")
    for _, row in df_stats_mcc.iterrows():
        sig = "✅" if row['significant'] else "—"
        lines.append(
            f"| {row['dataset']} | {row['mean_base']:.4f} | {row['mean_phase']:.4f} | "
            f"{row['delta']:+.4f} | {row['cliffs_delta']:+.4f} | {row['wilcoxon_p']:.4f} | {sig} |"
        )

    lines.append("\n---\n")
    lines.append("## 5. Análisis Estadístico — Accuracy (MetaCog Fase vs. MetaCog Baseline)\n")
    lines.append("| Dataset | Mean_Base | Mean_Fase | Δ | Cliff's δ | Wilcoxon p | Significativo |")
    lines.append("|:--------|:---------:|:---------:|:--:|:---------:|:----------:|:-------------:|")
    for _, row in df_stats_acc.iterrows():
        sig = "✅" if row['significant'] else "—"
        lines.append(
            f"| {row['dataset']} | {row['mean_base']:.4f} | {row['mean_phase']:.4f} | "
            f"{row['delta']:+.4f} | {row['cliffs_delta']:+.4f} | {row['wilcoxon_p']:.4f} | {sig} |"
        )

    lines.append("\n---\n")
    lines.append("## 6. Veredicto de la Fase\n")
    n_ok = df_summary['Criterion_OK'].sum()
    n_total = len(df_summary)

    if criterion_global:
        lines.append(f"> ✅ **FASE {phase_label} APROBADA** — El criterio del Comité se cumple en {n_ok}/{n_total} datasets.")
        lines.append(f"> El árbol MetaCog-C45 ya no colapsa de forma generalizada con esta configuración.")
        lines.append(f"> **Se autoriza el avance a la Fase siguiente o la reejecución del Lote 1.**")
    else:
        lines.append(f"> ❌ **FASE {phase_label} RECHAZADA** — El criterio del Comité NO se cumple ({n_ok}/{n_total} datasets dentro del rango).")
        lines.append(f"> Se requiere ajuste de hiperparámetros antes de avanzar.")

    report_text = "\n".join(lines)

    with open(fname, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"\n  [REPORT] Informe guardado: {fname}")
    return fname, report_text


# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MetaCog-C45 — Recalibración Experimental (Estrategia A)"
    )
    parser.add_argument(
        '--phase',
        choices=['A', 'B', 'C', 'D'],
        required=True,
        help="Fase de recalibración a ejecutar: A, B, C o D"
    )
    parser.add_argument(
        '--n_jobs',
        type=int,
        default=4,
        help="Número de procesos en paralelo (default: 4)"
    )
    parser.add_argument(
        '--master_seed',
        type=int,
        default=42,
        help="Semilla maestra del protocolo (default: 42)"
    )
    parser.add_argument(
        '--artifacts_dir',
        type=str,
        default=None,
        help=(
            "Directorio donde se guardan los informes de recalibración. "
            f"Por defecto: {DEFAULT_ARTIFACTS_DIR}"
        )
    )
    args = parser.parse_args()

    # Resolver directorio de artefactos y crearlo si no existe
    global ARTIFACTS_DIR
    ARTIFACTS_DIR = args.artifacts_dir if args.artifacts_dir else DEFAULT_ARTIFACTS_DIR
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    phase = args.phase
    params = PHASE_CONFIGS[phase]

    # 1. Ejecutar baseline sobre el panel
    print("\n[PASO 1/3] Ejecutando panel BASELINE...")
    df_baseline = run_panel(BASELINE_PARAMS, label='BASELINE',
                            master_seed=args.master_seed, n_jobs=args.n_jobs)

    # 2. Ejecutar configuración de la fase
    print(f"\n[PASO 2/3] Ejecutando panel FASE {phase}...")
    df_phase = run_panel(params, label=phase,
                         master_seed=args.master_seed, n_jobs=args.n_jobs)

    # 3. Calcular métricas
    print("\n[PASO 3/3] Calculando estadísticas y generando informe...")
    df_ratios  = compute_node_ratio(df_phase)
    df_summary = summarize_phase(df_phase, df_ratios)

    df_stats_mcc = run_stat_comparison(df_baseline, df_phase, metric='mcc')
    df_stats_acc = run_stat_comparison(df_baseline, df_phase, metric='accuracy')

    # Criterio global: todos los datasets deben cumplir
    criterion_global = df_summary['Criterion_OK'].all()

    # Imprimir resumen en consola
    print("\n" + "="*60)
    print(f"  RESUMEN — FASE {phase}")
    print("="*60)
    print(df_summary[['Dataset', 'MCC_Classic', 'MCC_MetaCog',
                       'Nodes_Classic', 'Nodes_MetaCog',
                       'Ratio_mean', 'Criterion_OK']].to_string(index=False))
    print("="*60)
    print(f"  Criterio global: {'✅ CUMPLE' if criterion_global else '❌ NO CUMPLE'}")
    print("="*60)

    # Generar informe
    report_path, _ = generate_phase_report(
        phase, params, df_phase, df_baseline,
        df_summary, df_ratios, df_stats_mcc, df_stats_acc,
        criterion_global
    )

    print(f"\n  Informe generado: {report_path}")
    print(f"\n{'='*60}")
    print(f"  FASE {phase} FINALIZADA")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
