"""
=========================================================
MetaCog-C45 — Pipeline Benchmark (Fase 8)
benchmark_pipeline.py

Mide el Speedup y la Eficiencia del pipeline paralelo
versus el pipeline serial sobre el dataset Iris.
=========================================================
"""

import os
import sys
import time
import psutil
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from experiments.pipeline.run_dataset import run_dataset

iris_path = os.path.join(ROOT, "experiments", "data", "iris.pkl")


def measure(n_jobs, master_seed=42, label=""):
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)  # MB

    t0 = time.time()
    df = run_dataset(iris_path, n_jobs=n_jobs, master_seed=master_seed)
    elapsed = time.time() - t0

    mem_after = process.memory_info().rss / (1024 * 1024)  # MB
    mem_delta = mem_after - mem_before

    print(f"\n  [{label:<15}] n_jobs={n_jobs} | Tiempo: {elapsed:.2f}s | RAM delta: +{mem_delta:.1f} MB")
    return elapsed


def main():
    print("=" * 60)
    print("  MetaCog-C45 — BENCHMARK DE RENDIMIENTO (Iris, 50 Folds)")
    print("=" * 60)

    cpu_count = os.cpu_count()
    print(f"  CPU threads disponibles: {cpu_count}")
    n_jobs_to_test = [1, 2, 4, max(1, cpu_count - 1)]

    times = {}
    for nj in n_jobs_to_test:
        label = "serial" if nj == 1 else f"parallel-{nj}"
        elapsed = measure(nj, label=label)
        times[nj] = elapsed

    # Resultados
    t_serial = times[1]
    print("\n" + "=" * 60)
    print("  TABLA DE RESULTADOS")
    print("=" * 60)
    print(f"  {'n_jobs':<10} {'Tiempo (s)':<15} {'Speedup':<12} {'Eficiencia'}")
    print(f"  {'-'*50}")
    for nj in n_jobs_to_test:
        speedup = t_serial / times[nj]
        efficiency = speedup / nj * 100
        label = "Serial" if nj == 1 else f"Paralelo x{nj}"
        print(f"  {label:<10} {times[nj]:<15.2f} {speedup:<12.2f} {efficiency:.1f}%")
    print("=" * 60)

    speedup_max = t_serial / times[max(1, cpu_count - 1)]
    print(f"\n  Speedup máximo alcanzado: {speedup_max:.2f}x con n_jobs={max(1, cpu_count - 1)}")
    print(f"  Tiempo serial estimado para 30 datasets: {t_serial * 30 / 60:.1f} min")
    print(f"  Tiempo paralelo estimado para 30 datasets: {times[max(1, cpu_count - 1)] * 30 / 60:.1f} min")


if __name__ == '__main__':
    main()
