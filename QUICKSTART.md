# Quick Start — MetaCog-C45 v1.1.0

## 1. Verify Installation
```bash
python check_environment.py
```

## 2. Prepare Datasets
```bash
python experiments/setup_datasets.py
```

## 3. Run the Experimental Campaign

**Full campaign (all 30 datasets, 3 batches):**
```bash
python experiments/run_experiment.py --batch all
```

**Single batch:**
```bash
python experiments/run_experiment.py --batch 1   # Datasets 1–10
python experiments/run_experiment.py --batch 2   # Datasets 11–20
python experiments/run_experiment.py --batch 3   # Datasets 21–30
```

## 4. Find Results
Results are saved to:
```
experiments/results/
├── lote1/
│   ├── raw_results.csv
│   ├── raw_results.json
│   ├── stats_results.csv
│   ├── summary.md
│   ├── summary.tex
│   └── plots/
├── lote2/  ...
├── lote3/  ...
└── package/          ← Consolidated Experimental Results Package
    ├── raw_results.csv
    ├── raw_results.json
    ├── stats_results.csv
    ├── summary.md
    ├── summary.tex
    └── plots/
```

## 5. Quick API Usage

```python
import pandas as pd
from core.classifier import C45Classifier

# Load your data
X_train, X_test = ...   # pd.DataFrame
y_train, y_test = ...   # pd.Series

# Classic C4.5
clf = C45Classifier(mode='classic')
clf.fit(X_train, y_train)
print(clf.predict(X_test))

# MetaCog-C45
clf_m = C45Classifier(mode='metacog', B=50)
clf_m.fit(X_train, y_train)
clf_m.summary()
```
