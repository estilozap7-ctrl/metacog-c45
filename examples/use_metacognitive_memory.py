"""
=========================================================
PyC45 Examples
Demonstration of the Metacognitive Memory Subsystem
=========================================================

Autor      : Luis Alberto Buelvas Cogollo / Antigravity
Proyecto   : PyC45
Descripción:
Este script ilustra de forma didáctica el funcionamiento del
Subsistema de Memoria Metacognitiva del paradigma MetaCog-C45.
Simula el análisis de un nodo raíz y sus ramificaciones,
calcula la competencia local, estima la estabilidad mediante bootstrap,
registra las trazas de decisión en MetaMemory e indexa por similitud.
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
    DecisionTrace,
    ExperienceStatistics
)


def run_metacognitive_demo():
    print("=" * 70)
    print("PyC45 - DEMOSTRACIÓN DEL SUBSISTEMA DE MEMORIA METACOGNITIVA")
    print("=" * 70)

    # 1. Simulación de datos locales en un nodo
    print("\n[Paso 1] Generando dataset local en el nodo...")
    rng = np.random.RandomState(42)
    N = 100
    df = pd.DataFrame({
        'edad': rng.randint(18, 70, N),
        'ingreso': rng.uniform(20000, 120000, N),
        'score_credito': rng.uniform(300, 850, N)
    })
    # Definir clase: aprobación de crédito dependiente principalmente de ingreso y score_credito
    df['aprobado'] = ((df['score_credito'] > 600) & (df['ingreso'] > 40000)).astype(int)

    X = df[['edad', 'ingreso', 'score_credito']]
    y = df['aprobado']

    print(f"  Muestras locales en el nodo: {len(y)}")
    print(f"  Distribución de clase: {y.value_counts().to_dict()}")

    # 2. Inicialización del motor metacognitivo y orquestadores
    print("\n[Paso 2] Inicializando subsistemas metacognitivos...")
    engine = ReflectionEngine(B=30, random_state=42)
    estimator = CompetitiveEstimator(competitive_threshold=0.75)
    memory = MetaMemory(w1=0.5, w2=0.5)

    # 3. Simulación de la evaluación de splits candidatos (Greedy Finder)
    print("\n[Paso 3] Evaluando competencia de atributos candidatos (CompetitiveEstimator)...")
    # Gain Ratios hipotéticos obtenidos en la optimización local
    gain_ratios = {
        'edad': 0.12,
        'ingreso': 0.65,
        'score_credito': 0.63
    }
    best_feature = 'ingreso'
    best_threshold = 40000.0

    comp_res = estimator.estimate_competition(gain_ratios, best_feature)
    print(f"  Atributo candidato óptimo: {best_feature} (Gain Ratio: {gain_ratios[best_feature]:.4f})")
    print(f"  Mejor alternativo: {comp_res['alternate_feature']} (Gain Ratio: {gain_ratios.get(comp_res['alternate_feature'], 0.0):.4f})")
    print(f"  Brecha de Gain Ratio (Delta): {comp_res['delta_gr']:.4f}")
    print(f"  Índice de Competitividad CI(u): {comp_res['competitiveness_index']:.4f}")
    print(f"  ¿Es una decisión altamente competitiva?: {comp_res['is_competitive']}")

    # 4. Evaluación de estabilidad con perturbación bootstrap
    print("\n[Paso 4] Ejecutando simulación bootstrap (ReflectionEngine) para estabilidad...")
    candidate_split = {
        'feature': best_feature,
        'threshold': best_threshold,
        'gain_ratio': gain_ratios[best_feature]
    }
    gamma = engine.reflect(X, y, candidate_split)

    print(f"  Estabilidad de Selección de Nodo (NS): {gamma['NS']:.4f}")
    print(f"  Estabilidad de Umbral Continuo (TS): {gamma['TS']:.4f}")
    print(f"  Frecuencia de selección de '{best_feature}': {gamma['p_star']:.4f}")
    print(f"  Distribución empírica del bootstrap: {gamma['p_vector']}")

    # 5. Creación y almacenamiento de la traza de decisión
    print("\n[Paso 5] Creando y registrando DecisionTrace en MetaMemory...")
    trace = DecisionTrace(
        node_id="root_node",
        depth=0,
        n_samples=len(y),
        class_distribution=y.value_counts().to_dict(),
        feature_selected=best_feature,
        threshold_selected=best_threshold,
        gain_ratio=gain_ratios[best_feature],
        ns=gamma['NS'],
        ts=gamma['TS'],
        p_star=gamma['p_star'],
        p_vector=gamma['p_vector'],
        bootstrap_selections=gamma['bootstrap_selections'],
        bootstrap_thresholds=gamma['bootstrap_thresholds']
    )
    memory.store(trace)
    print(f"  Tamaño de MetaMemory: {memory.size()} traza(s) registrada(s).")

    # Registrar más decisiones ficticias para probar indexación y estadísticas
    print("  Registrando trazas de decisión adicionales...")
    trace_hijo_izq = DecisionTrace(
        node_id="child_left", depth=1, n_samples=40, class_distribution={0: 38, 1: 2},
        feature_selected="score_credito", threshold_selected=600.0, gain_ratio=0.72,
        ns=0.85, ts=0.92, p_star=0.85, p_vector={"score_credito": 0.85, "edad": 0.15}
    )
    trace_hijo_der = DecisionTrace(
        node_id="child_right", depth=1, n_samples=60, class_distribution={0: 5, 1: 55},
        feature_selected="edad", threshold_selected=35.0, gain_ratio=0.35,
        ns=0.45, ts=0.55, p_star=0.45, p_vector={"edad": 0.45, "score_credito": 0.55}
    )
    memory.store(trace_hijo_izq)
    memory.store(trace_hijo_der)

    # 6. Consultas por similitud e indexación rápida
    print("\n[Paso 6] Consultando la memoria mediante MemoryIndex (K-NN)...")
    # Buscaremos experiencias similares a un nodo hipotético de n_samples=45 y clase={0: 35, 1: 10}
    similar_traces = memory.query_similar(n_samples=45, class_distribution={0: 35, 1: 10}, k=2)
    print("  Trazas similares encontradas para un nodo de 45 muestras con alta proporción de clase 0:")
    for trace_sim in similar_traces:
        print(f"    - ID: {trace_sim.node_id} (Nuestras: {trace_sim.n_samples}, Variable: {trace_sim.feature_selected})")

    # 7. Computar estadísticas acumuladas e importancia MFI
    print("\n[Paso 7] Extrayendo estadísticas y Metacognitive Feature Importance (MFI)...")
    stats = ExperienceStatistics.get_summary(memory)
    print(f"  Decisiones totales en memoria: {stats['total_decisions']}")
    print(f"  Estabilidad de nodo promedio (avg_NS): {stats['avg_ns']:.4f}")
    print(f"  Estabilidad de umbral promedio (avg_TS): {stats['avg_ts']:.4f}")
    print(f"  Importancia de características metacognitiva (MFI):")
    for feat, score in stats['mfi_normalized'].items():
        print(f"    - {feat}: {score:.4f}")

    print("\n" + "=" * 70)
    print("DEMOSTRACIÓN FINALIZADA EXITOSAMENTE")
    print("=" * 70)


if __name__ == '__main__':
    run_metacognitive_demo()
