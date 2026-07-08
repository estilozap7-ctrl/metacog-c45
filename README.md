# PyC45: Framework de Clasificación C4.5 en Python

**PyC45** es una implementación limpia, modular y orientada a objetos del clásico algoritmo de árboles de decisión **C4.5** para aprendizaje supervisado en tareas de clasificación binaria. Está diseñado con fines educativos y prácticos, ofreciendo una estructura completamente documentada en español y utilidades de evaluación y visualización de modelos de machine learning.

---

## 📂 Estructura del Proyecto

El repositorio está organizado de la siguiente manera:

```text
PyC45/
├── datasets/
│   ├── customer_churn.csv              # Dataset de ejemplo (Fuga de clientes)
│   └── default_of_credit_card_clients.xls # Dataset de prueba Excel
├── core/
│   ├── __init__.py
│   ├── decision_node.py                # Estructura de datos del nodo del árbol
│   ├── classifier.py                   # Clase principal del clasificador C4.5
│   ├── prediction.py                   # Alias/Módulo de predicción alternativo
│   ├── tree_builder.py                 # Construcción recursiva del árbol
│   ├── tree_utils.py                   # Utilidades para manipulación y visualización en consola
│   ├── entropy.py                      # Calculadora de entropía de Shannon y pureza
│   ├── information_gain.py             # Ganancia de información y Split Info
│   ├── gain_ratio.py                   # Cálculo del Gain Ratio (criterio C4.5)
│   ├── best_split.py                   # Búsqueda del mejor punto de corte/variable
│   └── pruning.py                      # Algoritmo de Poda por Error Reducido (REP)
├── metrics/
│   ├── __init__.py
│   ├── confusion_matrix.py             # Clase para modelar la matriz de confusión
│   ├── classification_metrics.py       # Cálculo de exactitud, precisión, f1, mcc, etc.
│   └── roc_auc.py                      # Curva ROC y cálculo del área bajo la curva (AUC)
├── visualization/
│   ├── feature_importance.py           # Análisis de importancia de variables predictoras
│   └── tree_plot.py                    # Generador de gráficos del árbol con Matplotlib
├── examples/
│   ├── churn_prediction.py             # Script de demostración con datos de fuga de clientes
│   └── credit_card_demo.py             # Script de demostración con tarjetas de crédito
├── tests/
│   └── test_c45.py                     # Suite completa de pruebas unitarias
├── web_verification/
│   ├── index.html                      # Página del informe técnico web
│   ├── styles.css                      # Estilos del informe web
│   ├── report.js                       # Renderizado dinámico del informe
│   ├── generate_report.py              # Ejecución del pipeline y exportación JSON
│   ├── generate_pdf.py                 # Generador del manual en formato PDF
│   └── manual_de_usuario.pdf           # Manual descargable generado
├── requirements.txt                    # Dependencias del proyecto
├── manual_de_usuario.md                # Manual paso a paso (Markdown)
├── manual_de_usuario.pdf               # Manual en formato PDF descargable (Copia Raíz)
└── README.md                           # Documentación y Manual de Usuario (este archivo)
```

---

## 🛠️ Requisitos e Instalación

El framework requiere las siguientes librerías de Python:
* **NumPy** (operaciones numéricas y vectoriales)
* **Pandas** (manipulación de datos estructurados)
* **Matplotlib** (generación de gráficos y curvas ROC)
* **xlrd** (lectura de hojas de cálculo de Excel `.xls`)
* **fpdf2** (generación de informes en formato PDF)

### Instalación de dependencias:
Puedes instalar los requerimientos ejecutando el siguiente comando en la raíz del proyecto:
```bash
pip install -r requirements.txt
```

---

## 📖 Manual de Usuario Paso a Paso (Nuevos Datasets)

Este manual práctico te guía para aplicar el clasificador C4.5 a cualquier nuevo conjunto de datos de clasificación binaria (por ejemplo, abandono de clientes, aprobación de créditos, diagnósticos médicos, etc.).

### 1. Requisitos del Dataset a Utilizar
* **Clasificación Binaria:** La columna objetivo (target) debe contener únicamente valores `0` (clase negativa) y `1` (clase positiva).
* **Variables Predictoras:** Pueden ser numéricas (continuas o discretas) o categóricas codificadas numéricamente.
* **Sin Valores Nulos (NaN):** Limpia o imputa registros vacíos antes de pasar los datos al clasificador.

### 2. Implementación en Código — Flujo Completo

#### Paso A: Importar los Módulos del Framework
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

#### Paso B: Carga y Preparación de Datos
```python
# Cargar el dataset (soporta CSV y Excel)
df = pd.read_csv("tu_archivo.csv")  # o pd.read_excel("tu_archivo.xls")

# Eliminar variables identificadoras (ej. ID, nombres)
if "ID" in df.columns:
    df = df.drop("ID", axis=1)

target = "nombre_columna_objetivo"
X = df.drop(target, axis=1)
y = df[target]
```

#### Paso C: Partición de Entrenamiento y Validación
```python
np.random.seed(42)
shuffled_indices = np.random.permutation(len(df))
split_idx = int(len(df) * 0.70) # 70% train, 30% validation

X_train = X.iloc[shuffled_indices[:split_idx]]
y_train = y.iloc[shuffled_indices[:split_idx]]
X_val = X.iloc[shuffled_indices[split_idx:]]
y_val = y.iloc[shuffled_indices[split_idx:]]
```
> **Nota:** Si tu dataset supera los 5,000 registros con variables decimales (continuas), se recomienda tomar una muestra representativa (ej. `df = df.sample(n=1500, random_state=42)`) para agilizar el cálculo del Gain Ratio.

#### Paso D: Entrenamiento y Poda (Pruning)
```python
# Entrenar C4.5
clf = C45Classifier(max_depth=5, min_samples_split=10, min_gain_ratio=0.01)
clf.fit(X_train, y_train)

# Podar el árbol de decisión usando los datos de validación
DecisionTreePruner.prune(clf, X_val, y_val)

# Mostrar el árbol final en consola
clf.print_tree()
clf.summary()
```

#### Paso E: Evaluación con Métricas Avanzadas
```python
# Predecir sobre validación
preds = clf.predict(X_val)

# Matriz de Confusión
cm = ConfusionMatrix.from_predictions(y_val, preds)
cm.summary()

# Métricas clásicas (Accuracy, Precision, Recall, F1, MCC, etc.)
metrics = ClassificationMetrics(cm)
metrics.summary()

# Área Bajo la Curva ROC (AUC)
probs = clf.predict_proba(X_val)[:, 1]
roc_data = ROCAUCCalculator.calculate(y_val, probs)
print(f"Área Bajo la Curva ROC (AUC): {roc_data['auc']:.4f}")
```

#### Paso F: Importancia de Variables y Gráficos
```python
# 1. Ranking de Importancia
fi = FeatureImportance().fit(clf.root)
print(fi.dataframe())

# 2. Guardar el gráfico del árbol
plotter = TreePlotter()
original_show = plt.show
plt.show = lambda: None  # Evita bloquear la ejecución
plotter.plot(clf.root, figsize=(14, 8))
plt.savefig("arbol_resultado.png", dpi=150)
plt.close('all')
plt.show = original_show
```

---

## 📋 3. Plantilla Genérica Integrada (Script Completo)

Puedes crear un script llamado `ejecutar_analisis.py`, pegar el código de abajo y ejecutarlo inmediatamente en cualquier dataset:

```python
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Asegurar importación del core
sys.path.append(os.path.abspath('.'))

from core.classifier import C45Classifier
from core.pruning import DecisionTreePruner
from metrics.confusion_matrix import ConfusionMatrix
from metrics.classification_metrics import ClassificationMetrics
from metrics.roc_auc import ROCAUCCalculator
from visualization.feature_importance import FeatureImportance
from visualization.tree_plot import TreePlotter

# ==========================================
# CONFIGURACIÓN DEL DATASET
# ==========================================
RUTA_DATASET = "datasets/customer_churn.csv"  # Ruta del archivo
COLUMNA_TARGET = "churn"                       # Variable objetivo
COLUMNAS_ELIMINAR = []                         # Columnas irrelevantes (ej. ["ID"])
MAX_SAMPLES = 1500                             # Submuestreo de seguridad
# ==========================================

def main():
    print(f"Cargando dataset: {RUTA_DATASET}")
    ext = os.path.splitext(RUTA_DATASET)[1].lower()
    df = pd.read_excel(RUTA_DATASET) if ext in ['.xls', '.xlsx'] else pd.read_csv(RUTA_DATASET)

    if COLUMNAS_ELIMINAR:
        df = df.drop(columns=COLUMNAS_ELIMINAR, errors='ignore')

    if len(df) > MAX_SAMPLES:
        df = df.sample(n=MAX_SAMPLES, random_state=42).reset_index(drop=True)

    X = df.drop(columns=[COLUMNA_TARGET])
    y = df[COLUMNA_TARGET]

    np.random.seed(42)
    sh = np.random.permutation(len(df))
    sp = int(len(df) * 0.70)
    X_train, y_train = X.iloc[sh[:sp]], y.iloc[sh[:sp]]
    X_val, y_val = X.iloc[sh[sp:]], y.iloc[sh[sp:]]

    print("Entrenando clasificador C4.5...")
    clf = C45Classifier(max_depth=5, min_samples_split=10, min_gain_ratio=0.01)
    clf.fit(X_train, y_train)

    print("\n--- Ejecutando Poda REP ---")
    DecisionTreePruner.prune(clf, X_val, y_val)
    clf.print_tree()

    print("\n--- Evaluación ---")
    preds = clf.predict(X_val)
    cm = ConfusionMatrix.from_predictions(y_val, preds)
    cm.summary()
    ClassificationMetrics(cm).summary()

    probs = clf.predict_proba(X_val)[:, 1]
    roc_data = ROCAUCCalculator.calculate(y_val, probs)
    print(f"ROC AUC: {roc_data['auc']:.4f}")

    print("\nImportancia de Variables:")
    print(FeatureImportance().fit(clf.root).dataframe())

if __name__ == "__main__":
    main()
```

---

## 🧪 Pruebas y Validación de la Instalación

### Ejecutar Pruebas Unitarias
Verifica el correcto funcionamiento del framework completo:
```bash
python -m unittest discover -s tests
```

### Ejecutar Informe Técnico Web
Genera los datos del informe y visualízalo en tu navegador:
```bash
# 1. Ejecutar el análisis y guardar resultados
python web_verification/generate_report.py
```
Una vez generados los datos, tienes dos opciones para abrir el informe:
* **Opción A (Abrir directamente):** Haz doble clic en el archivo [index.html](file:///c:/Users/gemcrak/Desktop/PyC45/web_verification/index.html) dentro de la carpeta `web_verification/` para abrirlo directamente en tu navegador. Esto es posible y no produce errores de CORS porque los datos están empaquetados en un archivo de script [report_data.js](file:///c:/Users/gemcrak/Desktop/PyC45/web_verification/report_data.js) cargado mediante una etiqueta `<script>`.
* **Opción B (Servidor Local):** Inicia un servidor web local mediante HTTP si deseas simular un entorno de producción o si tu navegador bloquea por defecto la ejecución de scripts locales:
  ```bash
  python -m http.server 8080 --directory web_verification
  ```
  Luego abre en tu navegador: [http://localhost:8080](http://localhost:8080).


---

## 🚀 Guía de Ejecución Paso a Paso

Sigue estas instrucciones paso a paso para configurar tu entorno y ejecutar los distintos scripts del proyecto:

### Paso 1: Crear y Activar un Entorno Virtual (Opcional pero recomendado)
Para mantener limpias tus dependencias, puedes aislar la instalación utilizando un entorno virtual:

* **En Windows:**
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
* **En Linux/macOS:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### Paso 2: Instalar las Dependencias del Proyecto
Asegúrate de instalar todas las dependencias indicadas en `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Paso 3: Ejecutar la Simulación Principal (main.py)
Este script genera un dataset sintético, entrena el clasificador C4.5, realiza la poda (pruning), muestra la estructura del árbol y genera métricas de evaluación detalladas junto con la importancia de variables:
```bash
python main.py
```

### Paso 4: Ejecutar los Ejemplos y Selección de Datasets Reales
El framework contiene scripts de demostración y un selector interactivo para correr todo el pipeline sobre conjuntos de datos reales:

* **Selector Interactivo de Datasets (Recomendado):**
  Te permite elegir interactivamente cualquier archivo `.csv` o `.xls/.xlsx` dentro de la carpeta `datasets/`, seleccionar automáticamente la columna objetivo, manejar valores nulos/categóricos y ejecutar todo el framework:
  ```bash
  python ejecutar_con_dataset.py
  ```
* **Demostración de Fuga de Clientes (Customer Churn):**
  ```bash
  python examples/churn_prediction.py
  ```
* **Demostración de Aprobación de Tarjetas de Crédito:**
  ```bash
  python examples/credit_card_demo.py
  ```

### Paso 5: Ejecutar la Suite de Pruebas Unitarias
Para comprobar que todas las partes del clasificador y las métricas funcionan correctamente y pasan los tests de validación:
```bash
python -m unittest discover -s tests
```

### Paso 6: Generar y Lanzar el Informe Técnico Web
Si deseas visualizar el informe dinámico en tu navegador local con gráficos interactivos y métricas avanzadas:

1. **Genera los datos del informe:**
   ```bash
   python web_verification/generate_report.py
   ```
2. **Visualiza el informe:** Puedes elegir uno de los siguientes métodos para visualizarlo:
   * **Opción A (Abrir directamente sin servidor - Recomendado por simplicidad):** Abre el archivo [index.html](file:///c:/Users/gemcrak/Desktop/PyC45/web_verification/index.html) directamente en tu navegador (haciendo doble clic). Esto es totalmente funcional y no produce errores de CORS porque los datos del reporte se exportan dinámicamente como código JS ejecutable en [report_data.js](file:///c:/Users/gemcrak/Desktop/PyC45/web_verification/report_data.js).
   * **Opción B (Lanzar servidor local):** Si prefieres simular un entorno web real o tu navegador restringe la ejecución de scripts en archivos locales:
     ```bash
     python -m http.server 8080 --directory web_verification
     ```
     E ingresa a [http://localhost:8080](http://localhost:8080).

### Paso 7: Generar el Manual de Usuario en PDF (Opcional)
Si realizaste modificaciones a la documentación en markdown y necesitas reconstruir el manual en PDF:
```bash
python web_verification/generate_pdf.py
```
