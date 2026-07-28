# User Guide — MetaCog-C45 v1.0.0

---

### Institutional Presentation

> **MetaCog-C45 Framework (Versión 1.0)**  
> Framework científico para árboles de decisión metacognitivos basado en C4.5, diseñado para integrar procesos de razonamiento dual (Sistema 1 + Sistema 2), Decision Core, Reflection Engine y MetaMemory.
> 
> **Autor:** Luis Alberto Buelvas Cogollo  
> **Afiliación:** Universidad de Córdoba, Montería, Córdoba, Colombia  
> **Programa:** Maestría en Inteligencia Artificial e Ingeniería del Conocimiento  
> **Líneas de investigación:** Inteligencia Artificial · Machine Learning · Explainable Artificial Intelligence (XAI)

---

This guide provides comprehensive step-by-step instructions for applying MetaCog-C45 to any binary classification dataset.

For installation instructions, see [INSTALL.md](../INSTALL.md).  
For a quick-start run of the experimental campaign, see [QUICKSTART.md](../QUICKSTART.md).

---

## 1. Dataset Requirements

- **Binary Classification**: The target column must contain exactly two unique values (e.g., `0` and `1`, `True/False`, `yes/no`).
- **Predictor Variables**: Numeric features (continuous or discrete) or numerically encoded categoricals.
- **No Missing Values (NaN)**: Clean or impute missing records before passing data to the classifier.

---

## 2. Python API — Complete Flow

### Step A: Import Framework Modules

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from core.classifier import C45Classifier
from core.pruning import DecisionTreePruner
from metrics.confusion_matrix import ConfusionMatrix
from metrics.classification_metrics import ClassificationMetrics
from metrics.roc_auc import ROCAUCCalculator
from visualization.feature_importance import FeatureImportance
from visualization.tree_plot import TreePlotter
```

### Step B: Load and Prepare Data

```python
# Load dataset (CSV or Excel)
df = pd.read_csv("your_file.csv")  # or pd.read_excel("your_file.xls")

# Remove identifier columns (e.g., ID, names)
if "ID" in df.columns:
    df = df.drop("ID", axis=1)

target = "target_column_name"
X = df.drop(target, axis=1)
y = df[target]
```

### Step C: Train/Validation Split

```python
np.random.seed(42)
shuffled_indices = np.random.permutation(len(df))
split_idx = int(len(df) * 0.70)  # 70% train, 30% validation

X_train = X.iloc[shuffled_indices[:split_idx]]
y_train = y.iloc[shuffled_indices[:split_idx]]
X_val   = X.iloc[shuffled_indices[split_idx:]]
y_val   = y.iloc[shuffled_indices[split_idx:]]
```

> **Note**: For datasets exceeding 5,000 records with continuous features, consider subsampling (e.g., `df = df.sample(n=1500, random_state=42)`) to reduce Gain Ratio computation time.

### Step D: Training and Pruning

```python
# Classic C4.5
clf = C45Classifier(mode='classic', max_depth=5, min_samples_split=10)
clf.fit(X_train, y_train)

# Reduce Error Pruning (REP)
DecisionTreePruner.prune(clf, X_val, y_val)

clf.print_tree()
clf.summary()
```

### Step E: MetaCog-C45 Mode (System 2)

```python
clf_meta = C45Classifier(
    mode='metacog',
    B=50,
    theta_accept=0.5,
    theta_reject=0.05,
    alpha_0=0.5,
    beta_0=0.5,
    gamma=0.1,
    lambda_1=0.3,
    lambda_2=0.3
)
clf_meta.fit(X_train, y_train)
clf_meta.summary()

# Access metacognitive experience statistics
print(clf_meta.experience_stats_)
```

### Step F: Evaluation Metrics

```python
preds = clf.predict(X_val)

# Confusion Matrix
cm = ConfusionMatrix.from_predictions(y_val, preds)
cm.summary()

# Classification metrics (Accuracy, Precision, Recall, F1, MCC)
metrics = ClassificationMetrics(cm)
metrics.summary()

# ROC AUC
probs    = clf.predict_proba(X_val)[:, 1]
roc_data = ROCAUCCalculator.calculate(y_val, probs)
print(f"ROC AUC: {roc_data['auc']:.4f}")
```

### Step G: Feature Importance and Plots

```python
# Feature importance ranking
fi = FeatureImportance().fit(clf.root)
print(fi.dataframe())

# Save decision tree plot
plotter = TreePlotter()
original_show = plt.show
plt.show = lambda: None
plotter.plot(clf.root, figsize=(14, 8))
plt.savefig("tree_result.png", dpi=150)
plt.close('all')
plt.show = original_show
```

---

## 3. Interactive HTML Report

The fastest way to get a full analysis report is to use the main CLI script:

```bash
python ejecutar_con_dataset.py --dataset datasets/customer_churn.csv --no_browser
```

This generates:
- `web_verification/report_data.js` — serialized results (metrics, tree, ROC, XAI traces)
- `web_verification/index.html` — interactive dashboard (open in any browser)

To view via local HTTP server:
```bash
python -m http.server 8080 --directory web_verification
# Open http://localhost:8080 in your browser
```

---

## 4. Hyperparameter Reference

| Parameter | Default | Description |
|---|---|---|
| `mode` | `'classic'` | `'classic'` (C4.5 only) or `'metacog'` (System 1 + System 2) |
| `max_depth` | `None` | Maximum tree depth (unlimited if None) |
| `min_samples_split` | `2` | Minimum samples required to split a node |
| `B` | `50` | Bootstrap replications for Reflection Engine |
| `theta_accept` | `0.5` | SCS threshold above which a split is directly accepted |
| `theta_reject` | `0.05` | SCS threshold below which the node collapses to a leaf |
| `alpha_0` | `0.5` | Base elasticity for Node Stability (NS) |
| `beta_0` | `0.5` | Base elasticity for Threshold Survival (TS) |
| `gamma` | `0.1` | Competitiveness penalty exponent $(1-CI)^\gamma$ |
| `lambda_1` | `0.3` | Depth amplification coefficient for $\alpha(u)$ |
| `lambda_2` | `0.3` | Sample scarcity amplification coefficient for $\beta(u)$ |

> **Note**: Default values correspond to the recalibrated post-sensitivity-study configuration that satisfies the scientific committee acceptance criterion $0.20 \le R_{\text{nodes}} \le 0.80$.

---

## Scientific Contribution

MetaCog-C45 introduces a novel metacognitive architecture for decision trees, proposing nine foundational scientific mechanisms:

- **Reflection Engine:** Local bootstrap perturbation module estimating split stability.
- **Decision Core:** System 2 decision-making engine evaluating node split candidates.
- **MetaMemory:** Persistent, indexed repository storing structural decision traces.
- **Split Confidence Score (SCS):** Multiplicative Cobb-Douglas utility metric integrating Gain Ratio, stability, survival, and competitiveness.
- **Node Stability (NS):** Normalized metric measuring feature selection variance across bootstrap samples.
- **Threshold Survival (TS):** Exponential decay function evaluating continuous cut-point variance.
- **Competitiveness Index (CI):** Margin ratio between winning split attribute and top alternative candidates.
- **DecisionTrace:** Immutable dataclass encapsulating execution context, feature distributions, SCS metrics, and XAI justifications.
- **Metacognitive Feature Importance (MFI):** Global feature ranking metric weighted by node sample size, NS, and TS.

---

## About the Author

- **Nombre:** Luis Alberto Buelvas Cogollo
- **Universidad:** Universidad de Córdoba (Montería, Córdoba, Colombia)
- **Programa de Maestría:** Maestría en Inteligencia Artificial e Ingeniería del Conocimiento
- **Área de investigación:** Inteligencia Artificial / Machine Learning / Explainable AI (XAI) / Metacognición Computacional
- **Líneas de trabajo:** Machine Learning · Explainable AI · Árboles de Decisión · Metacognición Computacional

---

## Citation & Copyright

### Citation
> Luis Alberto Buelvas Cogollo.  
> **MetaCog-C45: A Metacognitive Decision Tree Framework.**  
> Version 1.0.  
> Universidad de Córdoba. 2026.

### Copyright
Copyright © 2026 **Luis Alberto Buelvas Cogollo** (Universidad de Córdoba).  
Todos los derechos según la licencia seleccionada (**MIT License**).

