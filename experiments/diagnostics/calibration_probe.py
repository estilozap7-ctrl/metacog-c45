"""
=========================================================
MetaCog-C45 — Calibration Probe v1.0
calibration_probe.py

Diagnóstico cuantitativo del Decision Core con los
parámetros actuales de producción.

Intercepta DecisionValidator.validate() para capturar
la distribución empírica de:
  - Gain Ratio (GR)
  - Node Stability (NS)
  - Threshold Survival (TS)
  - Competitiveness Index (CI)
  - Split Confidence Score (SCS)
  - Elasticidades aplicadas (alpha, beta)
  - Veredictos (ACCEPT/REVIEW/DEFER/COLLAPSE)

Reporta:
  - Estadísticas por métrica (min, mean, p25, p50, p75, max)
  - % SCS < theta_reject
  - % theta_reject <= SCS < theta_accept
  - % SCS >= theta_accept
  - Distribución de veredictos
  - Descomposición por nivel de profundidad
  - Evolución de alpha/beta con profundidad

USO:
    python calibration_probe.py [--datasets NOMBRES] [--seed 42]
=========================================================
"""

from __future__ import annotations

import sys
import os
import json
import argparse
import pickle
import time
import numpy as np
import pandas as pd
from typing import List, Dict, Any

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

# -------------------------------------------------------
# Interceptor (Monkey-Patch) del DecisionValidator
# -------------------------------------------------------

_probe_records: List[Dict[str, Any]] = []
_verdict_records: List[Dict[str, Any]] = []

def _install_probe():
    """
    Instala el interceptor sobre DecisionValidator.validate()
    sin modificar el código fuente del framework.
    """
    from core.metacognition.decision_core import DecisionValidator

    original_validate = DecisionValidator.validate

    def probe_validate(self, X_u, y_u, top_k_candidates, depth,
                       max_depth, total_samples,
                       competitive_estimator, reflection_engine):

        outcome = original_validate(
            self, X_u, y_u, top_k_candidates, depth,
            max_depth, total_samples,
            competitive_estimator, reflection_engine
        )

        alpha = outcome.reproducibility_meta.get("alpha_applied", None)
        beta  = outcome.reproducibility_meta.get("beta_applied", None)
        n_samples = len(y_u)

        # --- Registrar cada candidato evaluado ---
        for cand in outcome.top_k_candidates:
            _probe_records.append({
                "depth"       : depth,
                "n_samples"   : n_samples,
                "feature"     : cand.get("feature", ""),
                "gain_ratio"  : float(cand.get("gain_ratio", 0.0)),
                "gr_normalized": float(cand.get("gr_normalized", 0.0)),
                "ns"          : float(cand.get("ns", 0.0)),
                "ts"          : float(cand.get("ts", 0.0)),
                "ci"          : float(cand.get("competitiveness_index", 0.0)),
                "scs"         : float(cand.get("scs", 0.0)),
                "alpha"       : float(alpha) if alpha is not None else None,
                "beta"        : float(beta)  if beta  is not None else None,
            })

        # --- Registrar el veredicto del nodo ---
        _verdict_records.append({
            "depth"      : depth,
            "n_samples"  : n_samples,
            "verdict"    : outcome.verdict.value,
            "scs_best"   : float(outcome.scs),
            "alpha"      : float(alpha) if alpha is not None else None,
            "beta"       : float(beta)  if beta  is not None else None,
        })

        return outcome

    DecisionValidator.validate = probe_validate
    return original_validate


# -------------------------------------------------------
# Utilidades de Reporte
# -------------------------------------------------------

SEP  = "=" * 70
SSEP = "-" * 70


def _stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {k: float("nan") for k in ["n", "min", "p25", "p50", "p75", "max", "mean", "std"]}
    a = np.array(values, dtype=float)
    return {
        "n"   : len(a),
        "min" : float(np.min(a)),
        "p25" : float(np.percentile(a, 25)),
        "p50" : float(np.percentile(a, 50)),
        "p75" : float(np.percentile(a, 75)),
        "max" : float(np.max(a)),
        "mean": float(np.mean(a)),
        "std" : float(np.std(a)),
    }


def _print_metric(name: str, values: List[float], indent: str = "  ") -> None:
    s = _stats(values)
    print(f"{indent}{name:<20} n={int(s['n'])}  "
          f"min={s['min']:.4f}  p25={s['p25']:.4f}  "
          f"p50={s['p50']:.4f}  p75={s['p75']:.4f}  "
          f"max={s['max']:.4f}  mean={s['mean']:.4f}  std={s['std']:.4f}")


def _pct(count: int, total: int) -> str:
    return f"{count:4d} / {total:4d}  ({100 * count / max(1, total):5.1f}%)"


# -------------------------------------------------------
# Carga de datasets
# -------------------------------------------------------

DATA_DIR = os.path.join(ROOT, "experiments", "data")

DEFAULT_DATASETS = ["iris", "breast-w", "diabetes", "balance-scale", "breast_cancer"]


def load_dataset(name: str):
    path = os.path.join(DATA_DIR, f"{name}.pkl")
    if not os.path.exists(path):
        print(f"  [WARN] Dataset no encontrado: {path}")
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


# -------------------------------------------------------
# Ejecución del Probe
# -------------------------------------------------------

def run_probe(dataset_names: List[str], seed: int, n_folds: int = 5) -> None:
    """
    Entrena MetaCog-C45 sobre los primeros n_folds folds de cada dataset
    y captura todas las métricas internas del Decision Core.
    """
    from core.classifier import C45Classifier

    print(f"\n{SEP}")
    print("  MetaCog-C45 — CALIBRATION PROBE v1.0")
    print(f"  Parámetros actuales de producción:")
    print(f"    theta_accept  = 0.50  |  theta_reject = 0.20")
    print(f"    alpha_0=1.0  beta_0=1.0  gamma=0.50  lambda_1=1.0  lambda_2=1.0")
    print(f"    B=50  |  seed={seed}")
    print(f"  Datasets: {dataset_names}  |  folds por dataset: {n_folds}")
    print(SEP)

    for ds_name in dataset_names:
        data = load_dataset(ds_name)
        if data is None:
            continue

        name   = data["name"]
        X      = data["X"]
        y      = data["y"]
        splits = data["cv_splits"]

        print(f"\n  Procesando [{name}] — {len(splits)} folds disponibles, usando {n_folds}...")
        t0 = time.time()
        for fold_idx, (train_idx, test_idx) in enumerate(splits[:n_folds]):
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            clf = C45Classifier(mode="metacog", random_state=seed + fold_idx)
            clf.fit(X_train, y_train)
        elapsed = time.time() - t0
        print(f"    Completado en {elapsed:.1f}s  |  registros capturados: {len(_probe_records)}")


# -------------------------------------------------------
# Reporte de Distribuciones
# -------------------------------------------------------

def report(theta_accept: float = 0.50, theta_reject: float = 0.20) -> None:

    if not _probe_records:
        print("\n[ERROR] No se capturaron registros. Verifica que el probe se instaló correctamente.")
        return

    df   = pd.DataFrame(_probe_records)
    dfv  = pd.DataFrame(_verdict_records)

    total_cands   = len(df)
    total_verdicts = len(dfv)

    # -------------------------------------------------------
    print(f"\n{SEP}")
    print("  SECCIÓN 1 — DISTRIBUCIONES GLOBALES DE MÉTRICAS")
    print(SEP)

    for metric in ["gain_ratio", "gr_normalized", "ns", "ts", "ci", "scs"]:
        _print_metric(metric, df[metric].tolist())

    if df["alpha"].notna().any():
        _print_metric("alpha (elasticidad)", df["alpha"].dropna().tolist())
        _print_metric("beta  (elasticidad)", df["beta"].dropna().tolist())

    # -------------------------------------------------------
    print(f"\n{SEP}")
    print("  SECCIÓN 2 — DISTRIBUCIÓN DEL SCS RESPECTO A UMBRALES")
    print(f"  theta_reject={theta_reject}  |  theta_accept={theta_accept}")
    print(SEP)

    scs_vals = df["scs"].values
    below_reject  = int(np.sum(scs_vals < theta_reject))
    in_defer_zone = int(np.sum((scs_vals >= theta_reject) & (scs_vals < theta_accept)))
    above_accept  = int(np.sum(scs_vals >= theta_accept))

    print(f"  SCS < theta_reject   ({theta_reject}) : {_pct(below_reject,  total_cands)}  → Zona COLLAPSE")
    print(f"  theta_reject ≤ SCS < theta_accept    : {_pct(in_defer_zone, total_cands)}  → Zona DEFER")
    print(f"  SCS ≥ theta_accept   ({theta_accept}) : {_pct(above_accept,  total_cands)}  → Zona ACCEPT")

    # -------------------------------------------------------
    print(f"\n{SEP}")
    print("  SECCIÓN 3 — DISTRIBUCIÓN DE VEREDICTOS")
    print(SEP)

    for verdict in ["ACCEPT", "REVIEW", "DEFER", "COLLAPSE"]:
        cnt = int((dfv["verdict"] == verdict).sum())
        print(f"  {verdict:<10}: {_pct(cnt, total_verdicts)}")

    # -------------------------------------------------------
    print(f"\n{SEP}")
    print("  SECCIÓN 4 — ANÁLISIS POR NIVEL DE PROFUNDIDAD")
    print(SSEP)
    print(f"  {'Depth':<8} {'N_nodos':<9} {'SCS_mean':<12} {'SCS_p50':<12} "
          f"{'alpha_mean':<12} {'% COLLAPSE':<12}")
    print(SSEP)

    for depth_val in sorted(dfv["depth"].unique()):
        sub_v   = dfv[dfv["depth"] == depth_val]
        sub_c   = df[df["depth"] == depth_val]
        n_nodes = len(sub_v)
        scs_m   = sub_c["scs"].mean() if not sub_c.empty else float("nan")
        scs_p50 = sub_c["scs"].median() if not sub_c.empty else float("nan")
        alph_m  = sub_c["alpha"].mean() if sub_c["alpha"].notna().any() else float("nan")
        col_pct = 100.0 * (sub_v["verdict"] == "COLLAPSE").sum() / max(1, n_nodes)
        print(f"  {depth_val:<8} {n_nodes:<9} {scs_m:<12.4f} {scs_p50:<12.4f} "
              f"{alph_m:<12.4f} {col_pct:<12.1f}%")

    # -------------------------------------------------------
    print(f"\n{SEP}")
    print("  SECCIÓN 5 — DIAGNÓSTICO DE SOBREPENALIZACIÓN POR PROFUNDIDAD")
    print(f"  (Ilustración: NS=0.7, TS=0.9, CI=0.3, GR_norm=1.0 — candidato ideal)")
    print(SSEP)

    ns_ref, ts_ref, ci_ref, gr_ref = 0.70, 0.90, 0.30, 1.0
    gamma_ref = 0.5
    max_d     = 20
    total_s   = 1000

    print(f"  {'depth':<8} {'rho':>6} {'factor':>8} {'alpha':>8} {'beta':>8}  "
          f"{'NS^a':>8} {'TS^b':>8} {'(1-CI)^g':>10} {'SCS':>8}  {'Zona'}")
    print(SSEP)

    for d in [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]:
        n_s = max(5, int(total_s * (0.9 ** d)))
        rho = n_s / total_s
        factor = 1.0 + 1.0 * (d / max(1, max_d)) + 1.0 * (1.0 - rho)
        a = 1.0 * factor
        b = 1.0 * factor
        ns_term  = ns_ref  ** a
        ts_term  = ts_ref  ** b
        ci_term  = (1.0 - ci_ref) ** gamma_ref
        scs_val  = gr_ref * ns_term * ts_term * ci_term
        zona = ("COLLAPSE" if scs_val < theta_reject
                else "DEFER"   if scs_val < theta_accept
                else "ACCEPT")
        print(f"  {d:<8} {rho:>6.3f} {factor:>8.3f} {a:>8.3f} {b:>8.3f}  "
              f"{ns_term:>8.4f} {ts_term:>8.4f} {ci_term:>10.4f} {scs_val:>8.4f}  {zona}")

    # -------------------------------------------------------
    print(f"\n{SEP}")
    print("  SECCIÓN 6 — CONCLUSIÓN DEL DIAGNÓSTICO")
    print(SEP)

    collapse_pct = 100.0 * below_reject / max(1, total_cands)
    collapse_verdict_pct = 100.0 * (dfv["verdict"] == "COLLAPSE").sum() / max(1, total_verdicts)

    print(f"\n  > {collapse_pct:.1f}% de los candidatos tienen SCS < theta_reject={theta_reject}")
    print(f"  > {collapse_verdict_pct:.1f}% de los nodos reciben veredicto COLLAPSE")

    if collapse_pct > 70:
        print(f"\n  [DIAGNÓSTICO] Sobrepenalización severa confirmada.")
        print(f"  La configuración actual induce colapso en la mayoría de nodos.")
        print(f"  Recomendación: Proceder con Fase 1 (ajuste de theta_reject).")
    elif collapse_pct > 40:
        print(f"\n  [DIAGNÓSTICO] Sobrepenalización moderada.")
        print(f"  Proceder con Fase 1 (ajuste de thresholds) y monitorear.")
    else:
        print(f"\n  [DIAGNÓSTICO] La política actual no induce colapso sistémico.")
        print(f"  Revisar otros factores (datos, imbalance).")

    print(f"\n{SEP}\n")


# -------------------------------------------------------
# Exportación de Resultados
# -------------------------------------------------------

def export_results(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    if _probe_records:
        df = pd.DataFrame(_probe_records)
        df.to_csv(os.path.join(output_dir, "probe_candidates.csv"), index=False)

    if _verdict_records:
        dfv = pd.DataFrame(_verdict_records)
        dfv.to_csv(os.path.join(output_dir, "probe_verdicts.csv"), index=False)

    # Guardar resumen en JSON
    summary = {
        "config": {
            "theta_accept": 0.50,
            "theta_reject": 0.20,
            "alpha_0": 1.0, "beta_0": 1.0, "gamma": 0.5,
            "lambda_1": 1.0, "lambda_2": 1.0, "B": 50
        },
        "total_candidates_evaluated": len(_probe_records),
        "total_nodes_evaluated": len(_verdict_records),
    }
    if _probe_records:
        scs_vals = [r["scs"] for r in _probe_records]
        summary["scs_pct_below_reject"] = float(100 * np.mean(np.array(scs_vals) < 0.20))
        summary["scs_pct_defer_zone"]   = float(100 * np.mean((np.array(scs_vals) >= 0.20) & (np.array(scs_vals) < 0.50)))
        summary["scs_pct_above_accept"] = float(100 * np.mean(np.array(scs_vals) >= 0.50))
    if _verdict_records:
        verdicts = [r["verdict"] for r in _verdict_records]
        for v in ["ACCEPT", "REVIEW", "DEFER", "COLLAPSE"]:
            summary[f"verdict_{v.lower()}_pct"] = float(100 * verdicts.count(v) / max(1, len(verdicts)))

    with open(os.path.join(output_dir, "probe_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"  Resultados exportados a: {output_dir}")


# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MetaCog-C45 — Calibration Probe: diagnóstico cuantitativo del Decision Core"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help=f"Nombres de datasets (sin .pkl). Default: {DEFAULT_DATASETS}"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla maestra (default: 42)"
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Número de folds por dataset (default: 5, reduce para mayor velocidad)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(ROOT, "experiments", "results", "calibration_probe"),
        help="Directorio de salida para los CSVs y JSON de resultados"
    )
    args = parser.parse_args()

    # Instalar el interceptor ANTES de importar/instanciar cualquier clasificador
    _install_probe()

    # Ejecutar probe
    run_probe(args.datasets, seed=args.seed, n_folds=args.folds)

    # Generar reporte
    report(theta_accept=0.50, theta_reject=0.20)

    # Exportar datos crudos
    export_results(args.output)


if __name__ == "__main__":
    main()
