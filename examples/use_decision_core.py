"""
=========================================================
MetaCog-C45 Examples
Demonstration of the Metacognitive Decision Core
=========================================================

Autor      : Luis Alberto Buelvas Cogollo / Antigravity
Proyecto   : MetaCog-C45
Descripción:
Este script ilustra de forma didáctica la ejecución integrada del
Decision Core de MetaCog-C45.
Muestra el cálculo de elasticidades dinámicas, la normalización local,
la simulación bootstrap y la aplicación de políticas formales
para determinar y auditar el veredicto final.
=========================================================
"""

import sys
import os
import pandas as pd
import numpy as np

# Agregar la raíz del proyecto al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.metacognition import (
    ReflectionEngine,
    CompetitiveEstimator,
    MetaMemory,
    DecisionValidator,
    DecisionAudit,
    SplitConfidenceScore
)


def run_decision_core_demo():
    print("=" * 80)
    print("MetaCog-C45 - DEMOSTRACIÓN DEL DECISION CORE (NÚCLEO DE DECISIÓN)")
    print("=" * 80)

    # 1. Simulación de datos
    rng = np.random.RandomState(101)
    N = 120
    df = pd.DataFrame({
        'edad': rng.randint(18, 65, N),
        'score_riesgo': rng.uniform(0.0, 1.0, N),
        'antiguedad': rng.uniform(0, 10, N)
    })
    # Definir clase separable por score_riesgo > 0.4
    df['default'] = (df['score_riesgo'] > 0.4).astype(int)

    X = df[['edad', 'score_riesgo', 'antiguedad']]
    y = df['default']

    # 2. Inicializar los componentes del framework
    engine = ReflectionEngine(B=30, random_state=101)
    estimator = CompetitiveEstimator()
    validator = DecisionValidator(theta_accept=0.5, theta_reject=0.2, theta_comp=0.75)
    memory = MetaMemory()

    # 3. Simular candidatos Top-K del C4.5 Split Finder
    print("\n[Paso 1] Candidatos Top-K propuestos por System 1:")
    top_k_candidates = [
        {"feature": "score_riesgo", "threshold": 0.4, "gain_ratio": 0.81},
        {"feature": "antiguedad", "threshold": 5.0, "gain_ratio": 0.35},
        {"feature": "edad", "threshold": 45.0, "gain_ratio": 0.12}
    ]
    for idx, c in enumerate(top_k_candidates):
        print(f"  {idx+1}. Variable: {c['feature']:<12} | Threshold: {c['threshold']:<5} | Gain Ratio: {c['gain_ratio']:.4f}")

    # 4. Validar el nodo a diferentes profundidades para ver elasticidad dinámica
    # Caso A: Nodo Raíz (poca profundidad, muchos datos)
    print("\n[Paso 2] Ejecutando validación en el Nodo Raíz (depth=0, n=120):")
    outcome_root = validator.validate(
        X_u=X, y_u=y, top_k_candidates=top_k_candidates,
        depth=0, max_depth=5, total_samples=120,
        competitive_estimator=estimator, reflection_engine=engine
    )

    print(f"  Veredicto final: {outcome_root.verdict.value}")
    print(f"  Justificación: {outcome_root.justification}")
    print(f"  SCS obtenido: {outcome_root.scs:.4f}")
    print(f"  Elasticidades aplicadas: alpha={outcome_root.reproducibility_meta['alpha_applied']:.2f}, beta={outcome_root.reproducibility_meta['beta_applied']:.2f}")

    # Caso B: Nodo Profundo (alta profundidad, pocas muestras)
    print("\n[Paso 3] Ejecutando validación en Nodo Profundo (depth=4, n=15) con los mismos candidatos:")
    # Tomamos un subset pequeño de datos
    X_deep = X.iloc[:15]
    y_deep = y.iloc[:15]
    
    outcome_deep = validator.validate(
        X_u=X_deep, y_u=y_deep, top_k_candidates=top_k_candidates,
        depth=4, max_depth=5, total_samples=120,
        competitive_estimator=estimator, reflection_engine=engine
    )

    print(f"  Veredicto final: {outcome_deep.verdict.value}")
    print(f"  Justificación: {outcome_deep.justification}")
    print(f"  SCS obtenido: {outcome_deep.scs:.4f}")
    print(f"  Elasticidades aplicadas: alpha={outcome_deep.reproducibility_meta['alpha_applied']:.2f}, beta={outcome_deep.reproducibility_meta['beta_applied']:.2f}")
    print("  *Nota: Las elasticidades aumentaron debido a la profundidad y reducción de muestras, aplicando veto estricto.")

    # 5. Auditar y persistir en memoria el veredicto del nodo raíz
    print("\n[Paso 4] Generando auditoría reproducible con DecisionAudit...")
    trace = DecisionAudit.audit_decision(
        node_id="root_node",
        X_u=X,
        y_u=y,
        depth=0,
        outcome=outcome_root,
        random_state=101
    )
    
    memory.store(trace)
    print(f"  Traza de decisión guardada en MetaMemory (Tamaño actual: {memory.size()})")
    print(f"  Verificando XAI Justification de la traza: {memory.get_all_traces()[0].justification}")

    print("\n" + "=" * 80)
    print("DEMOSTRACIÓN DE DECISION CORE FINALIZADA")
    print("=" * 80)


if __name__ == '__main__':
    run_decision_core_demo()
