"""
=========================================================
MetaCog-C45 Experimental Framework
Pipeline: report_generator.py
=========================================================
"""

import os

def generate_reports(df_results, df_stats, output_dir):
    """
    Genera informes en múltiples formatos (CSV, JSON, Markdown, LaTeX)
    a partir de los DataFrames de resultados.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # CSV
    df_results.to_csv(os.path.join(output_dir, "raw_results.csv"), index=False)
    if df_stats is not None:
        df_stats.to_csv(os.path.join(output_dir, "stats_results.csv"), index=False)
    
    # JSON
    df_results.to_json(os.path.join(output_dir, "raw_results.json"), orient='records', indent=2)
    
    # Markdown
    summary = df_results.groupby(['dataset', 'mode']).mean(numeric_only=True).reset_index()
    with open(os.path.join(output_dir, "summary.md"), "w", encoding='utf-8') as f:
        f.write("# Resumen Experimental\n\n")
        f.write(summary.to_markdown(index=False))
        
    # LaTeX
    with open(os.path.join(output_dir, "summary.tex"), "w", encoding='utf-8') as f:
        f.write(summary.to_latex(index=False))
