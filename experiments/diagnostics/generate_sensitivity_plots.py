"""
=========================================================
MetaCog-C45 — Generate Sensitivity Plots
generate_sensitivity_plots.py

Lee los resultados agregados del estudio de sensibilidad
y genera curvas científicas (PNG) para cada uno de los 7
hiperparámetros, guardándolas directamente en el directorio
de artefactos del agente.
=========================================================
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
CSV_PATH = os.path.join(ROOT, "experiments", "results", "sensitivity_study", "grouped_sensitivity_results.csv")
# Usar el directorio de artefactos de la conversación actual
ARTIFACT_DIR = r"C:\Users\gemcrak\.gemini\antigravity-ide\brain\5d524d62-d539-4dac-82d8-1df5947a6e0c"

def main():
    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] No se encontró el archivo de datos agrupados: {CSV_PATH}")
        sys.exit(1)
        
    df = pd.read_csv(CSV_PATH)
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    
    params = df["param_name"].unique()
    datasets = df["dataset"].unique()
    
    print(f"Generando curvas de sensibilidad para {len(params)} parámetros...")
    
    # Paleta de colores para consistencia
    colors = {
        "iris": "#1f77b4",
        "breast-w": "#2ca02c",
        "balance-scale": "#ff7f0e",
        "diabetes": "#d62728"
    }
    
    for param in params:
        sub_df = df[df["param_name"] == param]
        
        # Configurar gráfico de doble eje Y
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax2 = ax1.twinx()
        
        # Título y etiquetas
        ax1.set_title(f"Sensibilidad del Decision Core al parámetro: {param}", fontsize=12, fontweight="bold", pad=15)
        ax1.set_xlabel(f"Valor de {param}", fontsize=10, labelpad=10)
        ax1.set_ylabel("Ratio de Nodos (MetaCog / Classic)", color="#1f77b4", fontsize=10)
        ax2.set_ylabel("Metacog MCC", color="#d62728", fontsize=10)
        
        # Líneas de referencia para la banda del Comité [0.20, 0.80]
        if param in ["theta_reject", "theta_accept"] or True:
            ax1.axhspan(0.20, 0.80, color="gray", alpha=0.1, label="Banda Objetivo Comité [0.20, 0.80]")
            
        for ds in datasets:
            ds_df = sub_df[sub_df["dataset"] == ds].sort_values("param_value")
            if ds_df.empty:
                continue
                
            x = ds_df["param_value"].values
            ratios = ds_df["ratio_nodes"].values
            mccs = ds_df["metacog_mcc"].values
            
            # Dibujar ratio de nodos (línea sólida)
            ax1.plot(x, ratios, marker="o", linestyle="-", color=colors.get(ds, "#7f7f7f"), label=f"{ds} (Ratio)")
            # Dibujar MCC (línea discontinua con menor opacidad)
            ax2.plot(x, mccs, marker="x", linestyle="--", color=colors.get(ds, "#7f7f7f"), alpha=0.6, label=f"{ds} (MCC)")
            
        # Cuadrícula y límites
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.set_ylim(-0.05, 1.05)
        ax2.set_ylim(-0.1, 1.0)
        
        # Ajustar leyendas juntas
        lines1, labels1 = ax1.get_legend_handles_labels()
        # Filtrar duplicados de la banda gris en la leyenda
        unique_labels = []
        unique_lines = []
        for line, label in zip(lines1, labels1):
            if label not in unique_labels:
                unique_labels.append(label)
                unique_lines.append(line)
                
        ax1.legend(unique_lines, unique_labels, loc="upper left", fontsize=8)
        
        # Ajustar diseño y guardar
        plt.tight_layout()
        plot_path = os.path.join(ARTIFACT_DIR, f"plot_{param}.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  [OK] Gráfico guardado para '{param}' en: {plot_path}")
        
    print("Todos los gráficos generados con éxito.")

if __name__ == "__main__":
    main()
