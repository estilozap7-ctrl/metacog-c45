# Release Notes - MetaCog-C45 v1.0.0

We are proud to release the official production-ready core of **MetaCog-C45** (v1.0.0), a self-reflective, metacognitive decision tree framework.

## 1. Scientific & Engineering Context

Traditional greedy decision tree algorithms (C4.5, CART) suffer from myopic split selection and high threshold variance under noisy data. MetaCog-C45 addresses these vulnerabilities by introducing a dual-system cognitive architecture.

*   **System 1 (Greedy search):** Performs fast, information-gain split selection.
*   **System 2 (Metacognitive verification):** Executes perturbation simulations (bootstrap) to calculate stability indices ($NS$, $TS$), computes local competitiveness ($CI$), unifies them in a Cobb-Douglas Utility function ($SCS$), and audits the decision lifecycle to build a persistent memory of experience.

---

## 2. Traceability and Scientific Validation Mapping

To ensure strict compliance with Q1 journal requirements, the framework provides end-to-end traceability of the implemented modules:

| Research Gap | Hypothesis | Mathematical Model | Component | Verification Tests |
|---|---|---|---|---|
| **Myopic Split Selection:** Greedy algorithms fail under local sampling noise. | Local bootstrap perturbation can estimate the asymptotic variance of the optimal split choice. | Node Stability: <br> $NS(u) = p_* \cdot (1 - \frac{H(\mathbf{p})}{\log_2(d_{\text{eff}})})$ | `ReflectionEngine` & `NodeStabilityModule` | `test_node_stability_d_eff_normalization`<br>`test_b_convergence` |
| **Threshold Sensitivity:** Continuous thresholds overfit local variance. | Normalizing threshold variance by local feature variance maps boundary survival. | Threshold Survival: <br> $TS(u) = \exp(- \frac{\sigma_\theta^2}{\text{Var}_u(X^*) \cdot (1+\epsilon) + \epsilon})$ | `ThresholdSurvivalModule` | `test_ts_zero_variance_stability`<br>`test_edge_case_pure_node` |
| **Feature Redundancy & Ambiguity:** Multi-attribute colinearity dilutes confidence. | Relativizing the top-2 candidates' gain ratio maps local ambiguity. | Competitiveness Index: <br> $CI(u) = 1 - \frac{GR(X^*) - GR(X')}{GR(X^*) + \epsilon}$ | `CompetitiveEstimator` | `test_high_competition`<br>`test_absolute_dominance` |
| **Lack of Decision Auditing:** Trees lack explainability of discarded paths. | A unified Cobb-Douglas utility function can veto unstable splits and record justification. | Split Confidence Score: <br> $SCS = \overline{GR} \cdot NS^{\alpha(u)} \cdot TS^{\beta(u)} \cdot (1-CI)^{\gamma}$ | `SplitConfidenceScore` & `DecisionValidator` | `test_scs_veto_effect`<br>`test_verdict_accept`<br>`test_verdict_collapse_low_samples` |
| **Experience Preservation:** Tree builders cannot query past split contexts. | Indexing node profiles using Relative Size and L2 Class Distances allows fast similarity search. | Weighted Distance: <br> $d_{\text{total}} = w_1 d_{\text{size}} + w_2 d_{\text{class}}$ | `MemoryIndex` & `MetaMemory` | `test_knn_similarity`<br>`test_memory_indexing` |

---

## 3. Quick Start

Ensure python 3.11+ is installed.

```python
import pandas as pd
from core.metacognition import (
    ReflectionEngine,
    CompetitiveEstimator,
    DecisionValidator,
    DecisionAudit,
    MetaMemory
)

# 1. Initialize components
engine = ReflectionEngine(B=50, random_state=42)
estimator = CompetitiveEstimator()
validator = DecisionValidator(theta_accept=0.5, theta_reject=0.2)
memory = MetaMemory()

# 2. Local Node Subspace
X_u = pd.DataFrame({'x1': [1,2,3,4,5,6,7,8], 'x2': [10,20,10,20,10,20,10,20]})
y_u = pd.Series([0,0,0,0,1,1,1,1])

# Top-K candidate list from System 1
candidates = [
    {"feature": "x1", "threshold": 4.5, "gain_ratio": 0.85},
    {"feature": "x2", "threshold": 15.0, "gain_ratio": 0.15}
]

# 3. Validation & Auditing
outcome = validator.validate(
    X_u=X_u, y_u=y_u, top_k_candidates=candidates,
    depth=0, max_depth=5, total_samples=100,
    competitive_estimator=estimator, reflection_engine=engine
)

trace = DecisionAudit.audit_decision("node_0", X_u, y_u, depth=0, outcome=outcome)
memory.store(trace)

print(f"Verdict: {outcome.verdict.value}")
print(f"XAI: {outcome.justification}")
```

---

## ⚡ Infrastructure Notice (Phase III)
The parallelization implemented in the execution pipeline (`run_experiment.py` via `joblib.Parallel`) is strictly an infrastructure optimization designed to accelerate experimental execution on multi-core CPUs. **It does not modify the mathematical equations, algorithms, metacognitive weights, or decision-making behavior of the MetaCog-C45 algorithm.**

