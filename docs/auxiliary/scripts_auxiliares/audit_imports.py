"""
audit_imports.py — Analiza todos los imports de cada .py en la raíz del proyecto
para determinar qué usa qué, antes de mover archivos.
"""
import os, ast

def get_imports(filepath):
    try:
        src = open(filepath, encoding='utf-8', errors='replace').read()
        tree = ast.parse(src)
        imps = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    imps.append(('import', a.name))
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or '?'
                imps.append(('from', mod))
        return imps
    except Exception as e:
        return [('ERROR', str(e))]

# --- Scan root-level .py files ---
print("=" * 70)
print("AUDIT DE IMPORTS — Archivos .py en la raíz del proyecto")
print("=" * 70)

root_py = sorted(f for f in os.listdir('.') if f.endswith('.py') and os.path.isfile(f))

for fname in root_py:
    imps = get_imports(fname)
    local_deps = [m for t, m in imps if ('core' in m or 'metrics' in m or
                                          'experiments' in m or 'visualization' in m)]
    print(f"\n[{fname}]")
    print(f"  Imports locales del framework: {local_deps if local_deps else 'NINGUNO'}")
    for t, m in imps:
        print(f"    {t}: {m}")

print("\n" + "=" * 70)
print("FIN DEL AUDIT")
print("=" * 70)
