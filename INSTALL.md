# Installation Guide — MetaCog-C45 v1.0 / v1.1.0

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

## Requirements

| Requirement | Minimum Version |
|---|---|
| Python | 3.10 |
| pip | 21.0 |

---

## 1. Clone the Repository

```bash
git clone https://github.com/estilozap7-ctrl/metacog-c45.git
cd metacog-c45
```

---

## 2. Create a Clean Virtual Environment (Recommended)

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Verify the Installation

```bash
python check_environment.py
```

Expected final output:
```
INSTALLATION VERIFIED
```

Any `[FAIL]` line must be resolved before running experiments.

---

## 5. Prepare Experimental Datasets

```bash
python experiments/setup_datasets.py
```

This will download and preprocess the 30 datasets of the Official Protocol v2.0.  
Data is cached locally in `experiments/data/`; subsequent runs skip completed downloads.

---

## 6. (Optional) Install as Editable Package

```bash
pip install -e .
```

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'seaborn'`
```bash
pip install seaborn
```

### `ModuleNotFoundError: No module named 'openml'`
```bash
pip install openml
```

### Python version error
Ensure you are using Python 3.10 or newer:
```bash
python --version
```

### PATH warning for scripts (Windows)
If pip warns that installed scripts are not on PATH, add the scripts directory to your system PATH or always invoke tools using `python -m <module>`.

---

## ⚡ Parallelization Notice (Phase III)
The parallelization implemented in `run_experiment.py` (via `joblib.Parallel`) corresponds exclusively to an execution pipeline optimization and **does not alter the mathematical formulation or behavior of the MetaCog-C45 algorithm**.

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


