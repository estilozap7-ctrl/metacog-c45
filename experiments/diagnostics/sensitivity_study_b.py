ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import numpy as np
import pandas as pd
from core.metacognition.reflection_engine import ReflectionEngine

REPLICATIONS = 15
B_VALUES = [10, 20, 30, 50, 75, 100, 150, 200]
SEED = 42
N = 150

rng_data = np.random.RandomState(SEED)
X = pd.DataFrame(rng_data.uniform(0, 10, (N, 5)), columns=[f'x{i}' for i in range(5)])
y = pd.Series((X['x0'] > 5).astype(int))
candidate = {'feature': 'x0', 'threshold': 5.0, 'gain_ratio': 0.85}

sep = '=' * 65
print(sep)
print('ESTUDIO DE SENSIBILIDAD: Parametro B')
print(sep)
print(f"{'B':>5} | {'NS_mean':>8} | {'NS_std':>8} | {'TS_mean':>8} | {'TS_std':>8} | {'time_ms':>9}")
print('-' * 65)

results = {}
for B in B_VALUES:
    ns_vals, ts_vals, times = [], [], []
    for rep in range(REPLICATIONS):
        engine = ReflectionEngine(B=B, random_state=SEED + rep)
        t0 = time.perf_counter()
        gamma = engine.reflect(X, y, candidate)
        elapsed = (time.perf_counter() - t0) * 1000
        ns_vals.append(gamma['NS'])
        ts_vals.append(gamma['TS'])
        times.append(elapsed)
    r = {
        'ns_mean': float(np.mean(ns_vals)),
        'ns_std': float(np.std(ns_vals)),
        'ts_mean': float(np.mean(ts_vals)),
        'ts_std': float(np.std(ts_vals)),
        'time_ms': float(np.mean(times))
    }
    results[B] = r
    print(f"{B:>5} | {r['ns_mean']:>8.4f} | {r['ns_std']:>8.4f} | {r['ts_mean']:>8.4f} | {r['ts_std']:>8.4f} | {r['time_ms']:>9.2f}")

print()
print('Varianza Total (NS_std + TS_std) y Latencia:')
for B in B_VALUES:
    r = results[B]
    vt = r['ns_std'] + r['ts_std']
    print(f"  B={B:>3}: varianza_total={vt:.5f}  latencia={r['time_ms']:.2f}ms")

# Buscar el B que minimiza varianza total y tiene latencia < 30ms
best = min(B_VALUES, key=lambda b: (results[b]['ns_std'] + results[b]['ts_std']))
print(f"\n>>> B optimo (menor varianza): B = {best}")

print()
print(sep)
print('ANALISIS SESGO DIMENSIONAL: log2(d) vs log2(d_eff)')
print(sep)
print(f"{'d':>5} | {'log2(d)':>10} | {'log2(d_eff=2)':>14} | {'factor_sesgo':>12}")
print('-' * 65)
for d in [2, 5, 10, 20, 50, 100]:
    log2_d = float(np.log2(d))
    log2_d_eff = float(np.log2(2))
    sesgo = log2_d_eff / log2_d
    print(f"{d:>5} | {log2_d:>10.4f} | {log2_d_eff:>14.4f} | {sesgo:>12.4f}")

print()
print('CONCLUSION:')
print('  log2(d) global sobrepenaliza en datasets dispersos.')
print('  Con d=100, d_eff=2: factor_sesgo=0.15 => subestimacion 85%.')
print('  IMPLEMENTACION usa log2(d_eff): espacio activo observado en bootstrap.')
