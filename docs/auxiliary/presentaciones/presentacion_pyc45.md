# MetaCog-C45: Framework de Árboles de Decisión Metacognitivos
### Funcionamiento, Características y Aportes a la Analítica de Datos

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

# 🤖 Introducción a MetaCog-C45

### \u00bfQu\u00e9 es MetaCog-C45?
**PyC45** es una implementación limpia, modular y orientada a objetos en Python puro del algoritmo de árboles de decisión **C4.5** (diseñado originalmente por Ross Quinlan en 1993).

### Objetivo Principal
- **Educativo:** Proporcionar una base de código transparente y en español para entender la construcción recursiva de árboles de decisión y la poda sin la complejidad de optimizaciones de bajo nivel (como Scikit-Learn en Cython).
- **Práctico:** Ofrecer un pipeline completo para la clasificación binaria supervisada, permitiendo entrenar, podar, evaluar y visualizar modelos de machine learning con facilidad.

---

# 📂 Estructura Modular

El proyecto está diseñado bajo principios de modularidad y responsabilidad única:

*   **`core/`**: El núcleo algorítmico.
    *   `classifier.py`: Interfaz principal compatible con la api clásica de ajuste/predicción (`fit`/`predict`).
    *   `tree_builder.py`: Constructor recursivo del árbol (enfoque Top-Down).
    *   `entropy.py` & `information_gain.py` & `gain_ratio.py`: Cálculo de métricas de partición y pureza.
    *   `best_split.py`: Selector de variables y umbrales óptimos para cortes numéricos.
    *   `pruning.py`: Poda por error reducido (REP).
*   **`metrics/`**: Evaluación del modelo.
    *   `confusion_matrix.py`: Modelado y resumen de matriz de confusión.
    *   `classification_metrics.py`: Exactitud, Precisión, Recall, F1-score y coeficiente MCC.
    *   `roc_auc.py`: Curva ROC y cálculo del Área Bajo la Curva (AUC).
*   **`visualization/`**: Utilidades gráficas.
    *   `tree_plot.py`: Generador de diagramas del árbol mediante Matplotlib.
    *   `feature_importance.py`: Importancia de variables mediante Gain Ratio acumulado.

---

# ⚙️ Funcionamiento Paso a Paso (El Pipeline)

El ciclo de entrenamiento y evaluación en **MetaCog-C45** consta de 7 etapas:

```text
[ Carga de Datos ] ──► [ Preprocesamiento ] ──► [ División Train/Val ] ──► [ Entrenamiento C4.5 ]
                                                                                   │
[ Visualización ]  ◄── [ Evaluación (ROC/AUC) ] ◄── [ Poda por Error (REP) ] ◄─────┘
```

1.  **Carga de Datos:** Lectura nativa de archivos CSV y Excel (.xls) usando Pandas.
2.  **Preprocesamiento:** Eliminación de variables no predictivas e imputación.
3.  **Partición de Datos:** División de conjuntos (70% Entrenamiento, 30% Validación para Poda y Test).
4.  **Entrenamiento:** Construcción recursiva del árbol mediante el criterio de Gain Ratio.
5.  **Poda (Pruning):** Simplificación bottom-up usando el conjunto de validación.
6.  **Evaluación:** Generación de matrices de confusión y métricas robustas ante desbalances.
7.  **Visualización:** Gráfico visual del árbol, ranking de variables e informes web interactivos.

---

# 📐 El Núcleo Matemático: Criterios de Selección

El algoritmo selecciona la variable que mejor divide las clases calculando:

### 1. Entropía de Shannon (Impureza de un nodo)
$$H(S) = - \sum_{i} p_i \log_2(p_i)$$
Mide el desorden del conjunto. Es 0 para nodos puros y 1 para clases equipartidas.

### 2. Ganancia de Información (Information Gain)
$$IG(S, A) = H(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} H(S_v)$$
Mide la reducción de la entropía tras dividir por el atributo $A$. Sesga hacia atributos con muchos valores.

### 3. Normalización con Gain Ratio (Criterio C4.5)
$$SplitInfo(S, A) = - \sum_{v} \frac{|S_v|}{|S|} \log_2\left(\frac{|S_v|}{|S|}\right)$$
$$GainRatio(S, A) = \frac{IG(S, A)}{SplitInfo(S, A)}$$
Normaliza la ganancia dividiéndola por la entropía inherente de la partición (SplitInfo), penalizando divisiones atomizadas.

---

# ✂️ Simplificación del Árbol: Poda REP

La poda evita el **sobreajuste (overfitting)** simplificando el árbol posterior al entrenamiento:

### Poda por Error Reducido (REP - Reduced Error Pruning)
- **Enfoque Bottom-Up:** Comienza en las ramas más profundas cuyos hijos directos son hojas.
- **Evaluación con Validación:** Compara el rendimiento (Exactitud/Accuracy) en el conjunto de validación de dos escenarios:
    1.  Mantener el subárbol actual (con sus nodos de decisión y hojas correspondientes).
    2.  Colapsar el subárbol en una única hoja que prediga la clase mayoritaria en ese punto.
- **Criterio de Poda:** Si colapsar el nodo mantiene o mejora la exactitud global en el conjunto de validación, el subárbol es eliminado y reemplazado por la hoja.
- **Resultado:** Árboles más pequeños, generales, legibles y computacionalmente rápidos.

---

# ⚖️ PyC45 vs. Algoritmo C4.5 Clásico

| Dimensión | C4.5 Clásico (Quinlan, 1993) | Implementaci\u00f3n MetaCog-C45 |
| :--- | :--- | :--- |
| **Lenguaje** | Código en C monolítico y de difícil lectura | Python 3 orientado a objetos, limpio y modular |
| **Integración** | Archivos planos propietarios | Ecosistema moderno de datos (Pandas, NumPy) |
| **Puntos de Corte** | Cálculo interno secuencial en C | Búsqueda vectorizada y paralelizable de umbrales continuos |
| **Evaluación** | Tablas de error básicas | Matriz de confusión extendida (MCC, F1) y curva ROC-AUC |
| **Poda** | Poda pesimista basada en cotas de error teórico | Poda por Error Reducido (REP) con datos de validación |
| **Explicabilidad** | Reglas de texto plano | Diagramación gráfica (Matplotlib), ranking de variables e informe web |

---

# 🚀 Aporte a la Analítica de Datos

La implementación de **MetaCog-C45** proporciona tres aportes fundamentales al análisis moderno de datos:

### 1. Explicabilidad Transparente (XAI)
En sectores regulados (como finanzas para scoring crediticio o medicina para diagnóstico), los modelos de "caja negra" (redes neuronales, XGBoost) son difíciles de auditar. PyC45 genera árboles legibles cuyas reglas de negocio son 100% explícitas y visualizables.

### 2. Ecosistema de Métricas Riguroso
El framework no se limita a calcular la Exactitud (que puede mentir ante clases desbalanceadas). Integra el **Coeficiente de Correlación de Matthews (MCC)** y el **Área Bajo la Curva (ROC-AUC)**, garantizando que el analista evalúe de manera realista la capacidad discriminatoria del modelo.

### 3. Reducción de Barreras Educativas
Facilita el estudio detallado del algoritmo al separar claramente cada cálculo (entropía, ganancia, poda, búsqueda de cortes). Permite depurar paso a paso (debugging) la construcción de un árbol sobre datos reales.

---

# 🏁 Conclusiones del Proyecto

- **Efectividad y Control:** PyC45 demuestra que se puede construir un clasificador robusto y preciso sin depender de grandes librerías monolíticas, dando control total al analista sobre las decisiones del algoritmo.
- **Interpretabilidad:** El árbol final y la importancia de variables calculada por Gain Ratio acumulado permiten entender de inmediato qué atributos (como el historial de pagos `PAY_0`) lideran la toma de decisiones.
- **Próximos Pasos en Producción:** Para maximizar su rendimiento comercial, se recomienda acoplarlo con técnicas de sobremuestreo (SMOTE) para mitigar el desbalance de clases y optimizar la búsqueda de umbrales numéricos.

---

# 📚 Citación y Derechos de Autor

### Cómo Citar
> Luis Alberto Buelvas Cogollo.  
> **MetaCog-C45: A Metacognitive Decision Tree Framework.**  
> Version 1.0.  
> Universidad de Córdoba. 2026.

### Copyright
Copyright © 2026 **Luis Alberto Buelvas Cogollo** (Universidad de Córdoba).  
Todos los derechos según la licencia seleccionada (**MIT License**).

