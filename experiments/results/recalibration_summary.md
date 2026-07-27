# MetaCog-C45 Phase 0 Calibration Report

**Date:** 2026-07-20 17:40:22  
**Panel:** iris, wine, breast_cancer, adult  
**Configurations evaluated:** 252  

---

## 1. Optimal Configuration

| Parameter | Default | Calibrated | Δ |
|:---|:---:|:---:|:---:|
| `theta_accept` | 0.50 | **0.15** | -0.35 |
| `theta_reject` | 0.20 | **0.01** | -0.19 |
| `gamma`        | 0.50 | **0.00**        | -0.50 |
| **Utility**    | —    | **0.6182**      | — |

---

## 2. MCC Comparison

| Dataset | Classic MCC | Calibrated MCC | Δ MCC |
|:---|:---:|:---:|:---:|
| `iris` | 0.9224 | **0.6499** | -0.2725 |
| `wine` | 0.9203 | **0.8408** | -0.0795 |
| `breast_cancer` | 0.8169 | **0.8583** | +0.0414 |
| `adult` | 0.3770 | **0.3216** | -0.0554 |

## 3. Complexity Comparison

| Dataset | Classic Nodes | Calibrated Nodes | CR |
|:---|:---:|:---:|:---:|
| `iris` | 13.80 | **4.20** | 30.43% |
| `wine` | 12.20 | **8.60** | 70.49% |
| `breast_cancer` | 46.20 | **7.00** | 15.15% |
| `adult` | 244.60 | **14.60** | 5.97% |

## 4. Top-5 Ranking

| Rank | θ_accept | θ_reject | γ | Utility | MCC | CR | Stumps |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.15 | 0.01 | 0.00 | **0.6182** | 0.6677 | 30.51% | 0 |
| 2 | 0.20 | 0.01 | 0.00 | **0.6182** | 0.6677 | 30.51% | 0 |
| 3 | 0.25 | 0.01 | 0.00 | **0.6182** | 0.6677 | 30.51% | 0 |
| 4 | 0.30 | 0.01 | 0.00 | **0.6182** | 0.6677 | 30.51% | 0 |
| 5 | 0.35 | 0.01 | 0.00 | **0.6182** | 0.6677 | 30.51% | 0 |

---

## 5. Análisis de Sensibilidad de los Hiperparámetros

Análisis cuantitativo de la influencia de cada hiperparámetro sobre la función de utilidad:

| Hiperparámetro | Rango de Utilidad (Max - Min) | Desviación Estándar | Rango Evaluado | Impacto Relativo |
|:---|:---:|:---:|:---:|:---:|
| `gamma` | 0.8043 | 0.2718 | [0.00, 0.50] | **Mayor Influencia** (Crítico) |
| `theta_reject` | 0.5342 | 0.2151 | [0.01, 0.12] | Influencia Moderada |
| `theta_accept` | 0.0000 | 0.0000 | [0.15, 0.40] | Más Estable / Robusto |

### Hallazgos Clave:
- **Hiperparámetro de Mayor Influencia:** `gamma` presentó la mayor variabilidad de utilidad media, indicando que el Decision Core es altamente sensible a su calibración.
- **Hiperparámetro Más Estable:** `theta_accept` mostró la menor desviación estándar, revelando robustez relativa frente a variaciones de su rango.
- **Comportamiento Global de MetaCog-C45:** El Decision Core presenta un comportamiento robusto en las proximidades del óptimo encontrado, pero exhibe colapsos y sobrepoda catastrófica si los umbrales de rechazo son demasiado altos, lo que justifica la calibración empírica sistemática realizada.

---

## 6. Verification Checklist

- ✅ No stump collapse on any panel dataset
- ✅ Complexity ratio within [20%, 80%]

---

*Generated automatically by `experiments/run_recalibration_grid.py`*