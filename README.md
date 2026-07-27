# MetaCog-C45: A Metacognitive Decision Tree Framework

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/estilozap7-ctrl/metacog-c45/releases)
[![Python](https://img.shields.io/badge/python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-57%20passed-success.svg)](tests/)
[![Reproducible](https://img.shields.io/badge/reproducible-5×10--CV-orange.svg)](experiments/)
[![Status](https://img.shields.io/badge/status-Release%201.0-brightgreen.svg)](RELEASE_NOTES.md)

**MetaCog-C45** is a scientific machine learning framework that extends the classical **C4.5 decision tree algorithm** with a **dual-process metacognitive layer (System 2)** inspired by cognitive science. It provides statistically validated split decisions, decision traceability, explainability, and reproducible experimental evaluation protocols.

---

## Academic Context

MetaCog-C45 is the technological artifact developed as part of a **Master's thesis in Artificial Intelligence and Knowledge Engineering**. The framework constitutes the experimental platform for validating the research hypotheses, generating reproducible empirical evidence, and supporting the scientific publications derived from the investigation.

The framework is designed to fulfill three complementary roles:

1. **Research Instrument**: Implements the mathematical models proposed in the thesis — Split Confidence Score (SCS), Node Stability (NS), Threshold Survival (TS), and Competitiveness Index (CI) — and provides the experimental pipeline for their systematic evaluation.
2. **Scientific Software**: Adheres to the reproducibility standards required by peer-reviewed research venues, including deterministic cross-validation ($5 \times 10$-CV), controlled random seeds, and automated equivalence verification.
3. **Open-Source Framework**: Designed to be adopted by the research community as a reusable, documented, and independently verifiable tool.

---

## Key Features

| Feature | Description |
|---|---|
| **C4.5 Algorithm** | Complete implementation: Information Gain, Gain Ratio, Reduced Error Pruning (REP) |
| **Decision Core (System 2)** | Metacognitive validation layer that coordinates Node Stability, Threshold Survival, Competitiveness Index, and Split Confidence Score before each split |
| **Split Confidence Score (SCS)** | Multiplicative Cobb-Douglas utility function integrating stability, survival, and competitiveness metrics |
| **Reflection Engine** | Local bootstrap perturbation for estimating the variance of split decisions across data perturbations |
| **MetaMemory** | Persistent, indexed repository of decision traces with K-NN similarity retrieval |
| **Explainable AI (XAI)** | Structured justifications for every ACCEPT, REVIEW, DEFER, and COLLAPSE verdict |
| **Interactive HTML Report** | Single-file dashboard with classification metrics, decision tree, ROC curve, and metacognitive audit trace |
| **Experimental Pipeline** | Parallelized $5 \times 10$-CV protocol with fault-tolerant checkpoints and incremental results |
| **Dual-Mode Classifier** | `mode='classic'` for full C4.5 backward compatibility; `mode='metacog'` for the metacognitive layer |

---

## Scientific Foundation

MetaCog-C45 is grounded in four theoretical foundations:

**C4.5 Algorithm (Quinlan, 1993)**
The baseline classifier applies information-theoretic criteria — specifically Gain Ratio — to select the optimal split attribute at each node, with post-pruning via Reduced Error Pruning (REP) to control overfitting.

**Dual Process Theory**
The framework models inductive learning as a two-system process: System 1 performs fast, greedy split selection (C4.5); System 2 executes metacognitive verification of each candidate split, analogous to slow, deliberate reasoning in dual-process models of cognition.

**Metacognition in Machine Learning**
System 2 computes three complementary metrics per node — Node Stability (NS), Threshold Survival (TS), and Competitiveness Index (CI) — unified in the Split Confidence Score (SCS):

$$SCS = \overline{GR} \cdot NS^{\alpha(u)} \cdot TS^{\beta(u)} \cdot (1 - CI)^{\gamma}$$

where $\overline{GR}$ is the average Gain Ratio across bootstrap replications, and $\alpha(u)$, $\beta(u)$ are node-adaptive elasticities determined by tree depth and local sample size.

**Explainability (XAI)**
Every split decision is audited and stored as a structured `DecisionTrace`, providing full traceability from input features to final prediction.

The complete mathematical formulation and empirical validation are detailed in the associated thesis and forthcoming publications.

---

## Architecture

```
User / Researcher
       │
       ▼
CLI ─────────────────────────────────────────
       │  ejecutar_con_dataset.py
       │  python -m metacog_c45
       ▼
MetaCog-C45 Framework
│
├── core/                     C4.5 Engine (System 1)
│   ├── classifier.py         Dual-mode classifier entry point
│   ├── tree_builder.py       Recursive tree construction
│   ├── best_split.py         Top-K candidate selection
│   └── metacognition/        System 2 — Decision Core
│       ├── reflection_engine.py   Bootstrap perturbation (Reflection Engine)
│       ├── decision_core.py       SCS, DecisionValidator, DecisionPolicy
│       └── meta_memory.py         MetaMemory — experience repository
│
├── metrics/                  Evaluation (Accuracy, MCC, ROC-AUC, PR-AUC)
├── visualization/            Tree diagrams, ROC curves
├── metacog_c45/              Public API and official CLI (PyPI-ready)
└── web_verification/         Interactive HTML report engine
       │
       ▼
web_verification/index.html   Dashboard: Metrics · Tree · ROC · XAI Trace
experiments/                  Reproducible experimental pipeline (5×10-CV)
tests/                        Automated test suite (57 tests, pytest)
```

---

## Installation

### Requirements

| Requirement | Minimum Version |
|---|---|
| Python | 3.10 |
| pip | 21.0 |

### Quick Setup

```bash
# 1. Clone the repository
git clone https://github.com/estilozap7-ctrl/metacog-c45.git
cd metacog-c45

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify the environment
python check_environment.py
```

Expected output: `INSTALLATION VERIFIED`

For detailed installation instructions, virtual environment configuration, and optional pip package installation, see [INSTALL.md](INSTALL.md).

---

## Quick Start

### Interactive mode

```bash
python ejecutar_con_dataset.py
```

This command runs the full MetaCog-C45 pipeline on any CSV or Excel dataset and generates the HTML report in `web_verification/index.html`.

### Direct dataset argument

```bash
python ejecutar_con_dataset.py --dataset datasets/customer_churn.csv --no_browser
```

### Python API

```python
import pandas as pd
from core.classifier import C45Classifier

# Load your dataset
df = pd.read_csv("your_dataset.csv")
X = df.drop(columns=["target"])
y = df["target"]

# --- Classic C4.5 (System 1 only) ---
clf = C45Classifier(mode='classic', max_depth=5)
clf.fit(X, y)
predictions = clf.predict(X)

# --- MetaCog-C45 (System 1 + System 2) ---
clf_meta = C45Classifier(
    mode='metacog',
    B=50,                # Bootstrap replications for Reflection Engine
    theta_accept=0.5,    # SCS acceptance threshold
    theta_reject=0.05,   # SCS collapse threshold
    gamma=0.1            # Competitiveness penalty exponent
)
clf_meta.fit(X, y)
clf_meta.summary()       # Displays metacognitive metrics post-fit
```

For complete API usage, see [`examples/`](examples/) and [`docs/user_guide.md`](docs/user_guide.md).

---

## Project Structure

```text
metacog-c45/
├── core/                        # C4.5 engine and System 2 metacognitive layer
│   └── metacognition/           # Decision Core, Reflection Engine, MetaMemory
├── metacog_c45/                 # Public package layer and official CLI
├── metrics/                     # Evaluation metrics
├── visualization/               # Tree diagrams and ROC curves
├── web_verification/            # Interactive HTML report engine
├── experiments/                 # Experimental pipeline (5×10-CV)
│   ├── pipeline/                # Core pipeline modules
│   ├── benchmarks/              # Performance benchmarks
│   ├── diagnostics/             # Decision Core diagnostic tools
│   └── data/                    # Pre-processed experimental datasets
├── tests/                       # Automated test suite (pytest, 57 tests)
├── examples/                    # Usage examples and API demonstrations
├── datasets/                    # Sample datasets
├── docs/                        # Technical reports and supplementary documentation
├── check_environment.py         # Environment verification script
├── ejecutar_con_dataset.py      # Main user-facing CLI script
├── pyproject.toml               # PEP 517/518 packaging configuration
└── requirements.txt             # Runtime dependencies
```

---

## Reproducibility

MetaCog-C45 is designed to satisfy the reproducibility requirements of scientific publications:

- **Automated Test Suite**: 57 unit and integration tests covering all framework components (`pytest tests/`).
- **Equivalence Verification**: `tests/test_equivalence.py` verifies numerical identity between serial and parallel execution modes across all metrics, tree topology, and SCS values.
- **Experimental Protocol**: $5 \times 10$-CV with deterministic seed derivation (`fold_seed = master_seed + fold_idx`), fault-tolerant checkpoints, and incremental CSV results.
- **Decision Audit**: Every split decision is stored as a structured `DecisionTrace` — including feature, threshold, SCS value, verdict, and XAI justification — and is retrievable via K-NN similarity from MetaMemory.

```bash
# Run the full test suite
python -m pytest tests/

# Run the experimental campaign (requires dataset setup)
python experiments/setup_datasets.py
python experiments/run_experiment.py --batch all
```

---

## Documentation

| Document | Description |
|---|---|
| [INSTALL.md](INSTALL.md) | Full installation guide (virtual environment, pip, verification) |
| [QUICKSTART.md](QUICKSTART.md) | Step-by-step guide for running the experimental campaign |
| [CHANGELOG.md](CHANGELOG.md) | Complete version history and API changes |
| [RELEASE_NOTES.md](RELEASE_NOTES.md) | Scientific context, traceability map, and infrastructure notes |
| [docs/user_guide.md](docs/user_guide.md) | Comprehensive API usage guide and hyperparameter reference |
| [docs/reports/](docs/reports/) | Generated scientific reports (sensitivity analysis, recalibration) |
| [examples/](examples/) | Annotated usage examples |

---

## Publications

This section will be updated as research outputs become available.

| Type | Reference |
|---|---|
| Master's Thesis | *(in preparation)* |
| Journal Article | *(in preparation)* |
| DOI | *(to be assigned upon publication)* |

---

## How to Cite

If you use MetaCog-C45 in your research, please use the following reference:

```bibtex
@software{metacog_c45_2026,
  title        = {{MetaCog-C45}: A Metacognitive Decision Tree Framework},
  author       = {Buelvas Cogollo, Luis Alberto},
  year         = {2026},
  version      = {1.0.0},
  url          = {https://github.com/estilozap7-ctrl/metacog-c45},
  note         = {Developed as part of a Master's thesis in Artificial Intelligence
                  and Knowledge Engineering. Software artifact supporting
                  experimental validation and scientific publications.},
  license      = {MIT}
}
```

---

## License

This project is distributed under the **MIT License**. See the [LICENSE](LICENSE) file for full terms.

---

<div align="center">

**MetaCog-C45** · Release 1.0.0 · Master's Research in Artificial Intelligence and Knowledge Engineering

[GitHub](https://github.com/estilozap7-ctrl/metacog-c45) · [INSTALL.md](INSTALL.md) · [QUICKSTART.md](QUICKSTART.md) · [CHANGELOG.md](CHANGELOG.md)

</div>
