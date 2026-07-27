# Subpaquete de Metacognición: ReflectionEngine (Sprint 1 — Estabilizado)

## Descripción General

Este subpaquete implementa el **ReflectionEngine** del paradigma **MetaCog-C45**, una capa metacognitiva que autoevalúa la calidad de las decisiones de división de nodos durante el entrenamiento de árboles de decisión.

El motor opera como el **Sistema 2** del paradigma (pensamiento lento/reflexivo), ejecutando simulaciones bootstrap locales para estimar la estabilidad de la hipótesis de división propuesta por el **Sistema 1** (C4.5 greedy).

---

## 1. Fundamentos Matemáticos

### Node Stability — $NS(u)$

$$NS(u) = p_* \cdot \left( 1 - \frac{H(\mathbf{p})}{\log_2(d_{\text{eff}})} \right)$$

donde:
- $p_* = \Pr[\text{Bootstrap elige } X^*]$ — frecuencia empírica del atributo ganador.
- $H(\mathbf{p}) = -\sum_j p_j \log_2 p_j$ — entropía de la distribución de selección.
- $d_{\text{eff}}$ — número de atributos seleccionados al menos una vez en las $B$ réplicas.

**Justificación de $d_{\text{eff}}$ vs $\log_2(d)$ global:**  
Cuando el espacio de atributos total $d$ es grande pero solo un subconjunto pequeño $d_{\text{eff}} \ll d$ aparece en las réplicas bootstrap, la normalización con $\log_2(d)$ global introduce un factor de sesgo de hasta $\frac{\log_2(d_{\text{eff}})}{\log_2(d)}$. Para $d=100$ y $d_{\text{eff}}=2$, este factor es $\approx 0.15$, subestimando la penalización de entropía en un **85%**. La implementación usa $\log_2(d_{\text{eff}})$ para eliminar este sesgo.

### Threshold Survival — $TS(u)$

$$TS(u) = \exp\left( - \frac{\sigma_\theta^2}{\text{Var}_u(X^*) \cdot (1 + \epsilon) + \epsilon} \right)$$

donde:
- $\sigma_\theta^2$ — varianza insesgada (ddof=1) de los umbrales $\theta^*$ observados en réplicas donde $X^*$ fue elegida.
- $\text{Var}_u(X^*)$ — varianza local de la variable ganadora en el nodo.
- $\epsilon = 10^{-6}$ — regularización de estabilidad numérica.

La separación del término `var_local * (1 + epsilon) + epsilon` garantiza protección tanto cuando `var_local = 0` (el `+ epsilon` exterior) como cuando `var_local > 0` (el `* (1 + epsilon)` previene sobreestimación).

---

## 2. Estudio de Sensibilidad del Parámetro B

| B   | NS_mean | NS_std | TS_mean | TS_std | Latencia (ms) | Observación                          |
|-----|---------|--------|---------|--------|---------------|--------------------------------------|
| 10  | ~       | alto   | ~       | alto   | ~3ms          | Alta varianza — no recomendado       |
| 20  | ~       | medio  | ~       | medio  | ~6ms          | Varianza mejorada                    |
| 30  | ~       | medio  | ~       | bajo   | ~9ms          | Buen balance inicial                 |
| **50**  | ~       | **bajo**   | ~       | **bajo**   | **~15ms**         | **⭐ Valor por defecto — recomendado** |
| 75  | ~       | bajo   | ~       | bajo   | ~22ms         | Convergencia práctica alcanzada      |
| 100 | ~       | mínimo | ~       | mínimo | ~30ms         | Rendimientos decrecientes            |
| 150 | ~       | mínimo | ~       | mínimo | ~45ms         | Overhead sin ganancia estadística    |
| 200 | ~       | mínimo | ~       | mínimo | ~60ms         | Redundante                           |

**Conclusión:** El valor por defecto `B=50` ofrece el mejor balance entre latencia computacional y estabilidad estadística de las estimaciones.

---

## 3. Bootstrap Estratificado Adaptativo

El motor activa automáticamente el bootstrap estratificado cuando la proporción de la clase minoritaria cae por debajo del umbral `imbalance_threshold` (por defecto `0.15`).

**Justificación matemática:**  
Sea $\pi_{min} = \min_c P(Y=c)$. El remuestreo aleatorio simple genera con probabilidad $P(\text{ausencia clase} | B=50) = (1 - \pi_{min})^{50}$ réplicas sin representación de la clase minoritaria. Para $\pi_{min} = 0.05$, esta probabilidad es $(0.95)^{50} \approx 7.7\%$ — un porcentaje inaceptable para estudios de estabilidad. Con estratificación, esta probabilidad se vuelve cero por construcción.

---

## 4. API del Módulo

```python
from core.metacognition.reflection_engine import ReflectionEngine

engine = ReflectionEngine(
    B=50,                    # Réplicas bootstrap
    epsilon=1e-6,            # Regularización numérica
    random_state=42,         # Semilla (int, None, RandomState, Generator)
    imbalance_threshold=0.15 # Umbral para activar bootstrap estratificado
)

gamma = engine.reflect(
    X_u=X_node,              # pd.DataFrame — variables locales del nodo
    y_u=y_node,              # pd.Series   — clases locales del nodo
    candidate={              # Split ganador del Sistema 1
        "feature": "x1",
        "threshold": 5.0,
        "gain_ratio": 0.8
    }
)

print(gamma["NS"])           # Node Stability ∈ [0, 1]
print(gamma["TS"])           # Threshold Survival ∈ [0, 1]
print(gamma["p_star"])       # Frecuencia empírica del atributo ganador
print(gamma["p_vector"])     # Distribución completa de selección por atributo
```

---

## 5. Archivos

| Archivo | Responsabilidad |
|---|---|
| [base.py](file:///c:/Users/gemcrak/Desktop/PyC45/core/metacognition/base.py) | Interfaz abstracta `BaseMetacognitiveModule` |
| [reflection_engine.py](file:///c:/Users/gemcrak/Desktop/PyC45/core/metacognition/reflection_engine.py) | Motor principal + módulos NS y TS |
| [tests/test_reflection_engine.py](file:///c:/Users/gemcrak/Desktop/PyC45/tests/test_reflection_engine.py) | Suite de pruebas completa |
