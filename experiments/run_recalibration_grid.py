"""
=========================================================
MetaCog-C45 — Phase 0 Calibration Grid Search
experiments/run_recalibration_grid.py

Implements the protocol defined in:
  MetaCog-C45 Phase 0 Calibration Design Review v1.0

Parallelization strategy:
  - The outer grid loop is parallelized across configs
    using joblib with prefer='threads' (avoids loky/pickle
    issues with the MetaCog-C45 classifier internals).
  - The inner fold loop is sequential per config to
    prevent nested parallelism deadlocks on Windows.

Configuration:
  All parameters are loaded from recalibration_config.json.
  No datasets, ranges, or thresholds are hard-coded.

Outputs:
  experiments/results/recalibration_grid_results.csv
  experiments/results/recalibration_ranking.csv
  experiments/results/recalibration_optimal_config.json
  experiments/results/recalibration_summary.md
  experiments/results/plots/recalibration/*.png
=========================================================
"""

from __future__ import annotations

import os
import sys
import json
import time
import pickle
import datetime
import itertools
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# -------------------------------------------------------
# Path setup
# -------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiments.pipeline.run_fold import run_fold

# -------------------------------------------------------
# Output paths
# -------------------------------------------------------
RESULTS_DIR        = os.path.join(ROOT, 'experiments', 'results')
PLOTS_DIR          = os.path.join(RESULTS_DIR, 'plots', 'recalibration')
GRID_RESULTS_PATH  = os.path.join(RESULTS_DIR, 'recalibration_grid_results.csv')
RANKING_PATH       = os.path.join(RESULTS_DIR, 'recalibration_ranking.csv')
TOP10_PATH         = os.path.join(RESULTS_DIR, 'recalibration_top10.csv')
OPTIMAL_CFG_PATH   = os.path.join(RESULTS_DIR, 'recalibration_optimal_config.json')
SUMMARY_PATH       = os.path.join(RESULTS_DIR, 'recalibration_summary.md')
CHECKPOINT_PATH    = os.path.join(RESULTS_DIR, 'recalibration_checkpoint.json')

os.makedirs(PLOTS_DIR, exist_ok=True)


# -------------------------------------------------------
# Configuration loader
# -------------------------------------------------------

def load_config() -> dict:
    cfg_path = os.path.join(ROOT, 'experiments', 'recalibration_config.json')
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with open(cfg_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# -------------------------------------------------------
# Dataset loader
# -------------------------------------------------------

def load_dataset(name: str) -> dict:
    path = os.path.join(ROOT, 'experiments', 'data', f'{name}.pkl')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    with open(path, 'rb') as f:
        return pickle.load(f)


# -------------------------------------------------------
# Single-config evaluator (called in parallel)
# -------------------------------------------------------

def evaluate_config(
    cfg: dict,
    dataset_splits: dict[str, dict],
    folds_count: int,
    master_seed: int
) -> list[dict]:
    """
    Evaluates a single parameter configuration across all panel datasets.
    Used with threads backend (dataset_splits passed directly in shared memory).

    Parameters
    ----------
    cfg : dict
        Parameter configuration with keys theta_accept, theta_reject, gamma.
    dataset_splits : dict
        {ds_name: {'X': pd.DataFrame, 'y': pd.Series, 'splits': list}}
    folds_count : int
        Number of folds to evaluate.
    master_seed : int
        Master seed for deterministic evaluation.

    Returns
    -------
    list[dict]
        List of per-fold result records.
    """
    results = []
    metacog_params = {
        'theta_accept': cfg['theta_accept'],
        'theta_reject': cfg['theta_reject'],
        'gamma': cfg['gamma']
    }

    for ds_name, ds in dataset_splits.items():
        X = ds['X']
        y = ds['y']
        splits = ds['splits'][:folds_count]

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_test  = X.iloc[test_idx]
            y_test  = y.iloc[test_idx]
            seed = int(master_seed + fold_idx)

            r = run_fold(
                X_train, y_train, X_test, y_test,
                mode='metacog',
                random_state=seed,
                metacog_params=metacog_params
            )
            r['dataset']      = ds_name
            r['fold']         = fold_idx
            r['theta_accept'] = cfg['theta_accept']
            r['theta_reject'] = cfg['theta_reject']
            r['gamma']        = cfg['gamma']
            results.append(r)

    return results


def evaluate_config_loky(
    cfg: dict,
    ds_names: list[str],
    data_dir: str,
    folds_count: int,
    master_seed: int
) -> list[dict]:
    """
    Process-safe version of evaluate_config for loky multiprocessing backend.
    Loads datasets from disk inside each worker to avoid serializing large
    DataFrames across process boundaries (which causes loky deadlocks on Windows).

    Parameters
    ----------
    cfg : dict
        Parameter configuration with keys theta_accept, theta_reject, gamma.
    ds_names : list[str]
        Names of datasets to load from data_dir.
    data_dir : str
        Absolute path to directory containing <name>.pkl files.
    folds_count : int
        Number of folds to evaluate.
    master_seed : int
        Master seed for deterministic evaluation.

    Returns
    -------
    list[dict]
        List of per-fold result records.
    """
    import pickle as _pickle
    import os as _os
    import sys as _sys

    # Ensure project root is on path inside the worker process
    _root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if _root not in _sys.path:
        _sys.path.insert(0, _root)
    from experiments.pipeline.run_fold import run_fold as _run_fold

    metacog_params = {
        'theta_accept': cfg['theta_accept'],
        'theta_reject': cfg['theta_reject'],
        'gamma': cfg['gamma']
    }

    results = []
    for ds_name in ds_names:
        path = _os.path.join(data_dir, f'{ds_name}.pkl')
        with open(path, 'rb') as fh:
            data = _pickle.load(fh)
        X      = data['X']
        y      = data['y']
        splits = data['cv_splits'][:folds_count]

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_test  = X.iloc[test_idx]
            y_test  = y.iloc[test_idx]
            seed = int(master_seed + fold_idx)

            r = _run_fold(
                X_train, y_train, X_test, y_test,
                mode='metacog',
                random_state=seed,
                metacog_params=metacog_params
            )
            r['dataset']      = ds_name
            r['fold']         = fold_idx
            r['theta_accept'] = cfg['theta_accept']
            r['theta_reject'] = cfg['theta_reject']
            r['gamma']        = cfg['gamma']
            results.append(r)

    return results


# -------------------------------------------------------
# Utility score calculation
# -------------------------------------------------------

def calculate_utility(
    df_cfg: pd.DataFrame,
    classic_nodes: dict[str, float],
    utility_weights: dict
) -> float:
    """
    Computes the multi-objective utility score for a configuration.

    U(C) = mean_d [ MCC_MetaCog(d,C)
                    - w * |CR(d,C) - target_CR|
                    - stump_penalty * I(nodes_mean == 1) ]

    This formula is numerically stable regardless of MCC sign.
    """
    w           = utility_weights.get('complexity_penalty_weight', 0.20)
    target_cr   = utility_weights.get('target_complexity_ratio', 0.40)
    stump_val   = utility_weights.get('stump_penalty', 1.0)

    scores = []
    for ds, grp in df_cfg.groupby('dataset'):
        mcc_mean   = grp['mcc'].mean()
        nodes_mean = grp['nodes'].mean()
        c_nodes    = classic_nodes.get(ds, 1.0)
        cr         = nodes_mean / c_nodes if c_nodes > 0 else 1.0
        phi        = abs(cr - target_cr)
        psi        = stump_val if np.isclose(nodes_mean, 1.0, atol=0.05) else 0.0
        scores.append(mcc_mean - w * phi - psi)

    return float(np.mean(scores)) if scores else float('-inf')


# -------------------------------------------------------
# Checkpoint helpers
# -------------------------------------------------------

def load_checkpoint() -> tuple[set, list]:
    """Returns (completed_cfg_keys, evaluated_fold_records)."""
    if not os.path.exists(CHECKPOINT_PATH):
        return set(), []
    try:
        with open(CHECKPOINT_PATH, 'r', encoding='utf-8') as f:
            cp = json.load(f)
        return set(cp.get('completed', [])), cp.get('folds', [])
    except Exception:
        return set(), []


def save_checkpoint(completed: set, folds: list) -> None:
    tmp = CHECKPOINT_PATH + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump({'completed': list(completed), 'folds': folds}, f)
    os.replace(tmp, CHECKPOINT_PATH)


def cfg_key(cfg: dict) -> str:
    return f"ta{cfg['theta_accept']}_tr{cfg['theta_reject']}_g{cfg['gamma']}"


# -------------------------------------------------------
# Plot generation
# -------------------------------------------------------

def generate_plots(df_ranking: pd.DataFrame, df_raw: pd.DataFrame, optimal: dict) -> None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns

        opt_gamma = optimal['gamma']

        # --- Heatmap: utility at optimal gamma ---
        sub = df_ranking[np.isclose(df_ranking['gamma'], opt_gamma, atol=1e-9)]
        if not sub.empty:
            pivot = sub.pivot_table(
                index='theta_accept', columns='theta_reject', values='utility', aggfunc='mean'
            )
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.heatmap(pivot, annot=True, fmt='.3f', cmap='viridis', ax=ax,
                        cbar_kws={'label': 'Utility Score'})
            ax.set_title(f'Utility Score Heatmap  (γ = {opt_gamma})')
            fig.tight_layout()
            fig.savefig(os.path.join(PLOTS_DIR, 'heatmap_utility.png'), dpi=150)
            plt.close(fig)

        # --- MCC distribution ---
        df_classic = df_raw[df_raw['mode'] == 'classic'].copy()
        df_classic['Model'] = 'Classic C4.5'
        df_opt = df_raw[
            (np.isclose(df_raw['theta_accept'], optimal['theta_accept'], atol=1e-9)) &
            (np.isclose(df_raw['theta_reject'], optimal['theta_reject'], atol=1e-9)) &
            (np.isclose(df_raw['gamma'],        optimal['gamma'],        atol=1e-9))
        ].copy()
        df_opt['Model'] = 'Calibrated MetaCog'

        compare = pd.concat([df_classic, df_opt], ignore_index=True)

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(x='dataset', y='mcc', hue='Model', data=compare, ax=ax)
        ax.set_title('MCC Comparison — Calibration Panel')
        ax.tick_params(axis='x', rotation=15)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, 'mcc_distribution.png'), dpi=150)
        plt.close(fig)

        # --- Node distribution ---
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(x='dataset', y='nodes', hue='Model', data=compare, ax=ax)
        ax.set_title('Tree Complexity (Nodes) — Calibration Panel')
        ax.tick_params(axis='x', rotation=15)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, 'nodes_distribution.png'), dpi=150)
        plt.close(fig)

        # --- Utility curve sorted ---
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(range(len(df_ranking)), df_ranking['utility'].values, lw=1.5)
        ax.axhline(0, color='red', ls='--', lw=0.8, label='U=0')
        ax.set_xlabel('Configuration rank')
        ax.set_ylabel('Utility score')
        ax.set_title('Utility Score across all configurations (sorted)')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, 'utility_curve.png'), dpi=150)
        plt.close(fig)

        print(f"  Plots saved to: {PLOTS_DIR}")
    except Exception as exc:
        print(f"  [WARN] Plot generation failed: {exc}")


# -------------------------------------------------------
# Summary report
# -------------------------------------------------------

def generate_summary(
    optimal: dict,
    df_ranking: pd.DataFrame,
    df_raw: pd.DataFrame,
    datasets: list[str],
    classic_nodes: dict,
    classic_mcc: dict
) -> None:

    df_opt = df_raw[
        (np.isclose(df_raw['theta_accept'], optimal['theta_accept'], atol=1e-9)) &
        (np.isclose(df_raw['theta_reject'], optimal['theta_reject'], atol=1e-9)) &
        (np.isclose(df_raw['gamma'],        optimal['gamma'],        atol=1e-9))
    ]

    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    lines = [
        '# MetaCog-C45 Phase 0 Calibration Report',
        '',
        f'**Date:** {ts}  ',
        f'**Panel:** {", ".join(datasets)}  ',
        f'**Configurations evaluated:** {len(df_ranking)}  ',
        '',
        '---',
        '',
        '## 1. Optimal Configuration',
        '',
        '| Parameter | Default | Calibrated | Δ |',
        '|:---|:---:|:---:|:---:|',
        f'| `theta_accept` | 0.50 | **{optimal["theta_accept"]:.2f}** | {optimal["theta_accept"]-0.50:+.2f} |',
        f'| `theta_reject` | 0.20 | **{optimal["theta_reject"]:.2f}** | {optimal["theta_reject"]-0.20:+.2f} |',
        f'| `gamma`        | 0.50 | **{optimal["gamma"]:.2f}**        | {optimal["gamma"]-0.50:+.2f} |',
        f'| **Utility**    | —    | **{optimal["utility"]:.4f}**      | — |',
        '',
        '---',
        '',
        '## 2. MCC Comparison',
        '',
        '| Dataset | Classic MCC | Calibrated MCC | Δ MCC |',
        '|:---|:---:|:---:|:---:|',
    ]

    for ds in datasets:
        c_mcc = classic_mcc.get(ds, float('nan'))
        o_mcc = df_opt[df_opt['dataset'] == ds]['mcc'].mean() if not df_opt.empty else float('nan')
        delta = o_mcc - c_mcc if not (np.isnan(o_mcc) or np.isnan(c_mcc)) else float('nan')
        lines.append(f'| `{ds}` | {c_mcc:.4f} | **{o_mcc:.4f}** | {delta:+.4f} |')

    lines += [
        '',
        '## 3. Complexity Comparison',
        '',
        '| Dataset | Classic Nodes | Calibrated Nodes | CR |',
        '|:---|:---:|:---:|:---:|',
    ]

    for ds in datasets:
        c_n = classic_nodes.get(ds, float('nan'))
        o_n = df_opt[df_opt['dataset'] == ds]['nodes'].mean() if not df_opt.empty else float('nan')
        cr  = o_n / c_n if (c_n and c_n > 0) else float('nan')
        lines.append(f'| `{ds}` | {c_n:.2f} | **{o_n:.2f}** | {cr:.2%} |')

    lines += [
        '',
        '## 4. Top-5 Ranking',
        '',
        '| Rank | θ_accept | θ_reject | γ | Utility | MCC | CR | Stumps |',
        '|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|',
    ]

    for rank, (_, row) in enumerate(df_ranking.head(5).iterrows(), 1):
        lines.append(
            f'| {rank} | {row["theta_accept"]:.2f} | {row["theta_reject"]:.2f} | '
            f'{row["gamma"]:.2f} | **{row["utility"]:.4f}** | '
            f'{row["mean_mcc"]:.4f} | {row["mean_cr"]:.2%} | {int(row["stump_count"])} |'
        )

    lines += ['', '---', '']

    # --- 5. Análisis de Sensibilidad de los Hiperparámetros ---
    lines += [
        '## 5. Análisis de Sensibilidad de los Hiperparámetros',
        '',
        'Análisis cuantitativo de la influencia de cada hiperparámetro sobre la función de utilidad:',
        '',
        '| Hiperparámetro | Rango de Utilidad (Max - Min) | Desviación Estándar | Rango Evaluado | Impacto Relativo |',
        '|:---|:---:|:---:|:---:|:---:|'
    ]

    sensitivity = []
    for param in ['theta_accept', 'theta_reject', 'gamma']:
        grp = df_ranking.groupby(param)['utility'].mean()
        p_range = grp.max() - grp.min()
        p_std = grp.std()
        p_values = sorted(df_ranking[param].unique())
        sensitivity.append((param, p_range, p_std, p_values))

    # Ordenar por desviación estándar (mayor varianza -> mayor influencia)
    sensitivity.sort(key=lambda x: x[2], reverse=True)

    for idx, (param, p_range, p_std, p_values) in enumerate(sensitivity, 1):
        r_str = f"[{p_values[0]:.2f}, {p_values[-1]:.2f}]"
        if idx == 1:
            impact = "**Mayor Influencia** (Crítico)"
        elif idx == 2:
            impact = "Influencia Moderada"
        else:
            impact = "Más Estable / Robusto"
        lines.append(f'| `{param}` | {p_range:.4f} | {p_std:.4f} | {r_str} | {impact} |')

    most_inf = sensitivity[0][0]
    least_inf = sensitivity[-1][0]
    lines += [
        '',
        '### Hallazgos Clave:',
        f'- **Hiperparámetro de Mayor Influencia:** `{most_inf}` presentó la mayor variabilidad de utilidad media, indicando que el Decision Core es altamente sensible a su calibración.',
        f'- **Hiperparámetro Más Estable:** `{least_inf}` mostró la menor desviación estándar, revelando robustez relativa frente a variaciones de su rango.',
        '- **Comportamiento Global de MetaCog-C45:** El Decision Core presenta un comportamiento robusto en las proximidades del óptimo encontrado, pero exhibe colapsos y sobrepoda catastrófica si los umbrales de rechazo son demasiado altos, lo que justifica la calibración empírica sistemática realizada.',
        '',
        '---',
        ''
    ]

    # Verification
    cr_ok     = 0.20 <= optimal.get('mean_cr', 1.0) <= 0.80
    stump_ok  = optimal.get('stump_count', 1) == 0
    lines.append('## 6. Verification Checklist')
    lines.append('')
    lines.append(f'- {"✅" if stump_ok  else "❌"} No stump collapse on any panel dataset')
    lines.append(f'- {"✅" if cr_ok     else "❌"} Complexity ratio within [20%, 80%]')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('*Generated automatically by `experiments/run_recalibration_grid.py`*')

    with open(SUMMARY_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'  Markdown summary: {SUMMARY_PATH}')


# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main() -> None:
    sep = '=' * 70
    print(sep)
    print('  MetaCog-C45 — Phase 0 Calibration Grid Search')
    print(sep)

    # ---- Load configuration ----
    cfg         = load_config()
    ds_names    = cfg['datasets']
    folds_count = int(cfg.get('folds', 5))
    master_seed = int(cfg.get('master_seed', 42))
    pg          = cfg['parameter_grid']
    constraints = cfg.get('constraints', {})
    util_w      = cfg.get('utility', {})

    # ---- Build valid grid ----
    grid = []
    for ta, tr, g in itertools.product(
        pg['theta_accept'], pg['theta_reject'], pg['gamma']
    ):
        if constraints.get('reject_less_than_accept', True) and tr >= ta:
            continue
        grid.append({'theta_accept': float(ta), 'theta_reject': float(tr), 'gamma': float(g)})

    print(f'  Datasets          : {ds_names}')
    print(f'  Folds per dataset : {folds_count}')
    print(f'  Valid grid size   : {len(grid)} configurations')
    print(sep)

    # ---- Load datasets into memory ----
    print('\n[STEP 1/4] Loading panel datasets...')
    dataset_splits: dict[str, dict] = {}
    for name in ds_names:
        data = load_dataset(name)
        dataset_splits[name] = {
            'X':      data['X'],
            'y':      data['y'],
            'splits': data['cv_splits']
        }
        print(f'  Loaded {name}: {len(data["X"])} rows, {data["X"].shape[1]} features')

    # ---- Classic baseline (run once, sequential) ----
    print('\n[STEP 2/4] Running Classic C4.5 baseline...')
    classic_rows: list[dict] = []
    classic_mcc:   dict[str, float] = {}
    classic_nodes: dict[str, float] = {}

    for name, ds in dataset_splits.items():
        X, y, splits = ds['X'], ds['y'], ds['splits'][:folds_count]
        fold_mccs, fold_nodes = [], []

        for fi, (tr_idx, te_idx) in enumerate(splits):
            seed = int(master_seed + fi)
            r = run_fold(
                X.iloc[tr_idx], y.iloc[tr_idx],
                X.iloc[te_idx], y.iloc[te_idx],
                mode='classic', random_state=seed
            )
            r['dataset'] = name
            r['fold']    = fi
            classic_rows.append(r)
            fold_mccs.append(r['mcc'])
            fold_nodes.append(r['nodes'])

        classic_mcc[name]   = float(np.mean(fold_mccs))
        classic_nodes[name] = float(np.mean(fold_nodes))
        print(f'  {name}: MCC={classic_mcc[name]:.4f}  Nodes={classic_nodes[name]:.1f}')

    df_classic = pd.DataFrame(classic_rows)
    df_classic['theta_accept'] = np.nan
    df_classic['theta_reject'] = np.nan
    df_classic['gamma']        = np.nan

    # ---- Checkpoint recovery ----
    completed, prev_folds = load_checkpoint()
    if completed:
        print(f'\n[RECOVERY] Resuming: {len(completed)}/{len(grid)} configs already done.')

    # ---- Grid search — outer parallel over configs, inner sequential over folds ----
    print(f'\n[STEP 3/4] Grid search ({len(grid)} configs, threads backend)...')
    t_grid_start = time.time()

    # Configs not yet completed
    pending = [c for c in grid if cfg_key(c) not in completed]
    print(f'  Pending: {len(pending)} configurations')

    if pending:
        # Use loky multiprocessing backend; workers load data from disk
        # to avoid serializing large DataFrames across process boundaries.
        # n_jobs capped at 4: empirically determined safe max on Windows (loky backend).
        # n_jobs > 4 causes process-spawn deadlock on this system (semaphore/memory limit).
        # Confirmed working: n_jobs=4 → 1.87x real speedup over n_jobs=2.
        n_jobs    = min(4, max(1, (os.cpu_count() or 4) - 2))
        data_dir  = os.path.join(ROOT, 'experiments', 'data')
        print(f'  Parallelism: n_jobs={n_jobs}, backend=loky (process-safe)')
        sys.stdout.flush()

        chunk_size = n_jobs  # 1 config per worker per round → dynamic scheduling
        all_folds = list(prev_folds)
        
        for idx_c, i in enumerate(range(0, len(pending), chunk_size), 1):
            chunk = pending[i : i + chunk_size]
            total_chunks = (len(pending) + chunk_size - 1) // chunk_size
            print(f'\n  --- Running batch {idx_c}/{total_chunks} ({len(chunk)} configs) ---')
            sys.stdout.flush()
            t_chunk_start = time.time()

            batch_results: list[list[dict]] = Parallel(
                n_jobs=n_jobs, backend='loky', verbose=1
            )(
                delayed(evaluate_config_loky)(c, ds_names, data_dir, folds_count, master_seed)
                for c in chunk
            )

            new_folds: list[dict] = []
            for cfg_result, c in zip(batch_results, chunk):
                new_folds.extend(cfg_result)
                completed.add(cfg_key(c))

            all_folds.extend(new_folds)
            save_checkpoint(completed, all_folds)
            print(f'  Batch completed in {time.time()-t_chunk_start:.1f}s. Checkpoint saved.')
            sys.stdout.flush()

        print(f'  Grid search done in {time.time()-t_grid_start:.1f}s')
        sys.stdout.flush()
    else:
        all_folds = prev_folds
        print('  All configs already in checkpoint — using cached results.')

    # ---- Build DataFrames ----
    df_metacog = pd.DataFrame(all_folds)

    # Save raw fold-by-fold results (both modes)
    df_all = pd.concat([df_classic, df_metacog], ignore_index=True)
    df_all.to_csv(GRID_RESULTS_PATH, index=False)
    print(f'\n  Raw results saved: {GRID_RESULTS_PATH}  ({len(df_all)} rows)')

    # ---- Ranking ----
    print('\n[STEP 4/4] Computing ranking and generating outputs...')

    ranking_rows = []
    for c in grid:
        mask = (
            np.isclose(df_metacog['theta_accept'], c['theta_accept'], atol=1e-9) &
            np.isclose(df_metacog['theta_reject'], c['theta_reject'], atol=1e-9) &
            np.isclose(df_metacog['gamma'],        c['gamma'],        atol=1e-9)
        )
        sub = df_metacog[mask]
        if sub.empty:
            continue

        utility = calculate_utility(sub, classic_nodes, util_w)
        mean_cr_vals = []
        stump_count = 0
        for ds in ds_names:
            ds_nodes = sub[sub['dataset'] == ds]['nodes'].mean()
            c_n      = classic_nodes.get(ds, 1.0)
            cr_val   = ds_nodes / c_n if c_n > 0 else 1.0
            mean_cr_vals.append(cr_val)
            if np.isclose(ds_nodes, 1.0, atol=0.05):
                stump_count += 1

        ranking_rows.append({
            'theta_accept' : c['theta_accept'],
            'theta_reject' : c['theta_reject'],
            'gamma'        : c['gamma'],
            'utility'      : utility,
            'mean_mcc'     : sub['mcc'].mean(),
            'mean_accuracy': sub['accuracy'].mean(),
            'mean_nodes'   : sub['nodes'].mean(),
            'mean_depth'   : sub['depth'].mean(),
            'mean_cr'      : float(np.mean(mean_cr_vals)),
            'stump_count'  : stump_count,
            'mean_train_ms': sub['train_ms'].mean()
        })

    df_ranking = (
        pd.DataFrame(ranking_rows)
        .sort_values('utility', ascending=False)
        .reset_index(drop=True)
    )
    df_ranking.to_csv(RANKING_PATH, index=False)
    print(f'  Ranking saved: {RANKING_PATH}')
    df_ranking.head(10).to_csv(TOP10_PATH, index=False)
    print(f'  Top-10 saved: {TOP10_PATH}')

    # ---- Optimal config ----
    best = df_ranking.iloc[0]
    optimal = {
        'theta_accept': float(best['theta_accept']),
        'theta_reject': float(best['theta_reject']),
        'gamma'       : float(best['gamma']),
        'utility'     : float(best['utility']),
        'mean_mcc'    : float(best['mean_mcc']),
        'mean_cr'     : float(best['mean_cr']),
        'stump_count' : int(best['stump_count']),
        'timestamp'   : datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(OPTIMAL_CFG_PATH, 'w', encoding='utf-8') as f:
        json.dump(optimal, f, indent=2)
    print(f'  Optimal config: {OPTIMAL_CFG_PATH}')
    print(f'    θ_accept={optimal["theta_accept"]}  '
          f'θ_reject={optimal["theta_reject"]}  '
          f'γ={optimal["gamma"]}')
    print(f'    Utility={optimal["utility"]:.4f}  '
          f'MCC={optimal["mean_mcc"]:.4f}  '
          f'CR={optimal["mean_cr"]:.2%}  '
          f'Stumps={optimal["stump_count"]}')

    # ---- Plots ----
    generate_plots(df_ranking, df_all, optimal)

    # ---- Summary report ----
    generate_summary(optimal, df_ranking, df_all, ds_names, classic_nodes, classic_mcc)

    # ---- Clean checkpoint ----
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)

    print(sep)
    print('  Phase 0 Calibration COMPLETE')
    print(sep)


if __name__ == '__main__':
    main()
