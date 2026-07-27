# Changelog - MetaCog-C45

All notable changes to the MetaCog-C45 scientific framework will be documented in this file.

The project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.1] - 2026-07-12

### Added
- **Parallel Fold Execution:** Parallel validation cross-validation ($5\times 10$-CV) in `run_dataset.py` using `joblib.Parallel` and `joblib.delayed`.
- **Fault-Tolerant Checkpoint Manager:** `checkpoint.json` and incremental `summary.csv` to resume execution without repeating completed datasets.
- **Real-Time Progress Tracking:** Customized `tqdm_parallel` progress bar with Lote, Dataset, Fold progress and dynamic ETA estimation.
- **Configurable Parallel Jobs:** Added `--n_jobs` parameter defaulting to CPU threads minus 1.
- **Master Seed Derivation:** Deterministic and reproducible seed derivation per fold: `fold_seed = master_seed + fold_idx`.

### Note
- **Scientific Equivalence:** This optimization is strictly at the infrastructure/pipeline execution level and does not alter the mathematical equations or behaviour of the MetaCog-C45 algorithm.

---

## [1.1.0] - 2026-07-12

### Added
- **Dual-mode induction via `mode` parameter in `C45Classifier` and `TreeBuilder`:**
  - `mode='classic'` (default): 100% identical behavior to the original PyC45 C4.5 algorithm. Full backward compatibility guaranteed.
  - `mode='metacog'`: Activates the complete MetaCog-C45 Decision Core. Exposes `memory_` and `experience_stats_` post-`fit`.
- **`BestSplitFinder.find_top_k_splits(X, y, k)`:** New method returning the Top-K best candidates and the per-feature GR map consumed by `DecisionValidator`.
- **`DecisionNode.trace_id`:** New optional field linking each structural node to its audit trace ID in `MetaMemory`.
- **`DecisionValidator` now stores `_competitive_estimator` and `_reflection_engine`** as injected attributes for use by `TreeBuilder` without repeated constructor arguments.
- **`examples/use_metacog_classifier.py`:** End-to-end demo of the unified MetaCog-C45 flow via `C45Classifier(mode='metacog')`.

### Changed
- **`tree_builder.py`:** Fully refactored to support dual-mode induction (`classic`/`metacog`) via Dependency Injection of `DecisionValidator` and `MetaMemory`. Shared stop conditions and node factories extracted to private helpers `_make_leaf` / `_make_split_node`.
- **`classifier.py`:** Extended to accept `mode`, `B`, `random_state`, `top_k`, `theta_accept`, `theta_reject`, `theta_comp`. `summary()` now displays metacognitive metrics in `mode='metacog'`.

### Removed
- **`core/prediction.py`:** Eliminated redundant file (exact duplicate of `classifier.py`). No external API changes.

---

## [1.0.0] - 2026-07-12

### Added
- **Metacognitive Decision Core (System 2):**
  - `SplitConfidenceScore`: Multiplicative Cobb-Douglas utility function integrating Gain Ratio, Node Stability ($NS$), Threshold Survival ($TS$), and Competitiveness Index ($CI$) with node-dependent elasticities $\alpha(u), \beta(u)$ based on depth and sample size.
  - `DecisionPolicy`: State machine transition policy (`ACCEPT`, `REVIEW`, `DEFER`, `COLLAPSE`).
  - `DecisionValidator`: Top-K candidate validation and verification coordinator.
  - `DecisionOutcome`: Structured audit outcome encapsulating XAI justification, uncertainty, and metadata.
  - `DecisionAudit`: Factory class responsible for compiling reproducible `DecisionTrace` records.
- **Metacognitive Memory Subsystem:**
  - `DecisionTrace`: Enhanced dataclass containing decision-level stats, bootstrap selections, thresholds, and reproducibility parameters.
  - `MetaMemory`: Intelligent repository to store and index traces.
  - `MemoryIndex`: Feature and depth indexes, plus a K-NN profile similarity query tool.
  - `ExperienceStatistics`: Consolidator class computing the **Metacognitive Feature Importance (MFI)** metric:
    $$MFI(X_j) = \sum n_u \cdot NS(u) \cdot TS(u)$$
- **Reflection Engine & Perturbation Analysis:**
  - `ReflectionEngine`: Local bootstrap simulation coordinator with stratified sampling for class imbalance.
  - `NodeStabilityModule`: Active subspace normalization using $d_{\text{eff}}$ to avoid dimensionality bias:
    $$NS(u) = p_* \cdot \left(1 - \frac{H(\mathbf{p})}{\log_2(d_{\text{eff}})}\right)$$
  - `ThresholdSurvivalModule`: Exponential decay of continuous cut-points relative to local variance:
    $$TS(u) = \exp\left(-\frac{\sigma_\theta^2}{\text{Var}_u(X^*) \cdot (1+\epsilon) + \epsilon}\right)$$
- **Testing and Verification:**
  - `tests/test_reflection_engine.py`: 10 high-rigor unit/integration tests (reproducibility, edge cases, B convergence).
  - `tests/test_memory_subsystem.py`: 11 memory unit/integration tests.
  - `tests/test_decision_core.py`: 6 decision core unit/integration tests.
  - Comprehensive demo scripts in `examples/use_metacognitive_memory.py` and `examples/use_decision_core.py`.
