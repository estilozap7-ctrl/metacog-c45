"""
=========================================================
MetaCog-C45 Experimental Framework
Pipeline: plot_generator.py
=========================================================
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_plots(df_results, output_dir):
    """
    Genera los gráficos exigidos: ROC, PR, CD-Diagram, Boxplots, Violin, Heatmaps.
    Nota: Las curvas ROC/PR reales requerirían los tensores de probabilidad completos.
    Esta función grafica las distribuciones de las métricas escalares resultantes.
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    metrics_to_plot = ['mcc', 'nodes', 'infer_ms']
    
    # 1. Boxplots
    for metric in metrics_to_plot:
        if metric in df_results.columns:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='dataset', y=metric, hue='mode', data=df_results)
            plt.xticks(rotation=90)
            plt.title(f"Boxplot de {metric} por Dataset")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"boxplot_{metric}.png"))
            plt.close()
            
    # 2. Violin plots
    for metric in metrics_to_plot:
        if metric in df_results.columns:
            plt.figure(figsize=(10, 6))
            sns.violinplot(x='dataset', y=metric, hue='mode', data=df_results, split=True)
            plt.xticks(rotation=90)
            plt.title(f"Violin Plot de {metric} por Dataset")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"violin_{metric}.png"))
            plt.close()
            
    # 3. Heatmaps (Diferencia Media: MetaCog - Classic)
    if 'mcc' in df_results.columns:
        summary = df_results.groupby(['dataset', 'mode'])['mcc'].mean().unstack()
        summary['Diff'] = summary['metacog'] - summary['classic']
        plt.figure(figsize=(4, 8))
        sns.heatmap(summary[['Diff']], annot=True, cmap="coolwarm", center=0)
        plt.title("Heatmap: Diferencia MCC (MetaCog - Classic)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "heatmap_mcc.png"))
        plt.close()

    # (Nota: CD-Diagram requiere una implementación específica de Nemenyi, 
    # se integraría con bibliotecas especializadas como Orange3 o scikit-posthocs)
