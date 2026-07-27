"""
=============================================================
generate_flowchart_mpl.py  –  v2 (refined)
MetaCog-C45 Framework — Professional Flowchart Generator
=============================================================
Genera el diagrama de flujo completo del framework MetaCog-C45
usando matplotlib puro, nivel publicación Q1.

Salida (en el directorio de trabajo):
  metacog_c45_flowchart.pdf  —  PDF vectorial (LaTeX / Word)
  metacog_c45_flowchart.png  —  PNG 300 DPI  (insertar en tesis)

Uso:
  python generate_flowchart_mpl.py
=============================================================
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ──────────────────────────────────────────────────────────
# PALETA DE COLORES — academic soft
# ──────────────────────────────────────────────────────────
C = {
    # System 1 — azul oscuro
    's1_f':  '#EFF6FF',
    's1_e':  '#1D4ED8',
    's1_t':  '#1E3A8A',

    # System 2 — verde
    's2_f':  '#F0FDF4',
    's2_e':  '#16A34A',
    's2_t':  '#14532D',

    # MetaMemory — naranja
    'mm_f':  '#FFF7ED',
    'mm_e':  '#EA580C',
    'mm_t':  '#9A3412',

    # Reflection Engine — morado
    're_f':  '#FAF5FF',
    're_e':  '#7C3AED',
    're_t':  '#4C1D95',

    # Terminador (Inicio / Fin) — gris
    'tr_f':  '#F1F5F9',
    'tr_e':  '#475569',
    'tr_t':  '#1E293B',

    # General
    'arrow': '#1E293B',
    'bg':    '#FFFFFF',
}

FT  = 'DejaVu Sans'
FS  = 8.5    # label estándar
FSE = 7.8    # ecuación
FSS = 6.5    # texto pequeño

# ──────────────────────────────────────────────────────────
# CANVAS
# ──────────────────────────────────────────────────────────
FW, FH = 9.0, 23.5
fig, ax = plt.subplots(figsize=(FW, FH), dpi=150)
ax.set_xlim(0, FW)
ax.set_ylim(0, FH)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor(C['bg'])
ax.set_facecolor(C['bg'])

# ──────────────────────────────────────────────────────────
# PRIMITIVOS
# ──────────────────────────────────────────────────────────

def rbox(cx, cy, w, h, fill, edge, labels, tc,
         fs=FS, radius=0.12, lw=1.3, eq=None, eq_fs=FSE, zo=4):
    """Nodo rectangular con esquinas redondeadas."""
    p = FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                       boxstyle=f'round,pad=0,rounding_size={radius}',
                       fc=fill, ec=edge, lw=lw, zorder=zo)
    ax.add_patch(p)
    if isinstance(labels, str):
        labels = [labels]
    n = len(labels)
    if eq:
        # Título arriba, ecuación debajo
        for i, l in enumerate(labels):
            off = 0.12 - i * 0.19
            ax.text(cx, cy + off, l, ha='center', va='center',
                    fontsize=fs, fontweight='bold', color=tc,
                    fontfamily=FT, zorder=zo+1)
        ax.text(cx, cy - 0.14, eq, ha='center', va='center',
                fontsize=eq_fs, style='italic', color='#374151',
                fontfamily=FT, zorder=zo+1)
    else:
        for i, l in enumerate(labels):
            off = (n - 1 - 2*i) * 0.095 if n > 1 else 0
            ax.text(cx, cy + off, l, ha='center', va='center',
                    fontsize=fs, fontweight='bold', color=tc,
                    fontfamily=FT, zorder=zo+1)


def diam(cx, cy, w, h, fill, edge, labels, tc, fs=FS, lw=1.3, zo=4):
    """Nodo rombo (decisión)."""
    pts = [[cx, cy+h/2], [cx+w/2, cy], [cx, cy-h/2], [cx-w/2, cy]]
    p = plt.Polygon(pts, closed=True, fc=fill, ec=edge, lw=lw, zorder=zo)
    ax.add_patch(p)
    if isinstance(labels, str):
        labels = [labels]
    n = len(labels)
    for i, l in enumerate(labels):
        off = (n - 1 - 2*i) * 0.07 if n > 1 else 0
        ax.text(cx, cy+off, l, ha='center', va='center',
                fontsize=fs, fontweight='bold', color=tc,
                fontfamily=FT, zorder=zo+1)


def av(x, y1, y2, lw=1.15, color=None, ls='-'):
    """Flecha vertical hacia abajo."""
    c = color or C['arrow']
    ax.annotate('', xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle='->', color=c, lw=lw,
                                linestyle=ls, mutation_scale=9), zorder=3)


def ah(x1, x2, y, lw=1.15, color=None):
    """Flecha horizontal."""
    c = color or C['arrow']
    ax.annotate('', xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle='->', color=c, lw=lw,
                                mutation_scale=9), zorder=3)


def lbl(x, y, txt, ha='center', va='center', fs=FSS,
        bold=True, col='#1E293B'):
    ax.text(x, y, txt, ha=ha, va=va, fontsize=fs,
            fontweight='bold' if bold else 'normal',
            color=col, fontfamily=FT, zorder=6)


def cbox(x, y, w, h, title, fill, ec='#94A3B8', zo=1):
    """Caja contenedor con borde punteado."""
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle='round,pad=0,rounding_size=0.12',
                       fc=fill, ec=ec, lw=0.9,
                       linestyle=(0, (6, 4)),
                       alpha=0.5, zorder=zo)
    ax.add_patch(p)
    ax.text(x+0.18, y+h-0.20, title, ha='left', va='top',
            fontsize=6.0, fontweight='bold', color='#374151',
            fontfamily=FT, zorder=zo+1, style='italic')


# ──────────────────────────────────────────────────────────
# POSICIONES Y (de arriba hacia abajo, valores decrecientes)
# ──────────────────────────────────────────────────────────
# Anchos y alturas estándar
NW  = 3.40   # node width normal
NH  = 0.48   # node height normal
EH  = 0.72   # node height extendido (con ecuación)
DW  = 1.90   # diamond width
DH  = 0.65   # diamond height

CX   = 4.50  # centro horizontal principal
CXR  = 7.20  # centro horizontal rama derecha

Y = {
    'inicio':       22.80,
    'carga':        22.08,
    'preproc':      21.36,
    'entropia':     20.64,
    'info_gain':    19.92,
    'gain_ratio':   19.20,
    'sel_attr':     18.48,
    'hoja_q':       17.62,   # diamond
    'crear_hoja':   17.62,   # right branch
    'fin_rama':     16.86,   # right branch

    'bootstrap':    15.90,
    'node_stab':    15.18,
    'thresh_surv':  14.46,
    'comp_est':     13.74,
    'scs':          12.87,   # extendido
    'consul_mm':    11.93,
    'recup_exp':    11.21,
    'conf_hist':    10.49,
    'reflection':    9.57,   # extendido
    'md_q':          8.63,   # diamond

    'reevaluar':     8.63,   # right branch
    'registrar':     7.87,   # right branch
    'buscar':        7.11,   # right branch

    'aceptar':       7.75,
    'crear_nodo':    7.03,
    'hijo_izq':      6.31,
    'hijo_der':      5.59,
    'pend_q':        4.71,   # diamond extendido

    'arbol_final':   3.83,
    'arbol_const':   3.11,
    'fin':           2.39,
}

# ──────────────────────────────────────────────────────────
# CONTENEDORES
# ──────────────────────────────────────────────────────────
# System 1
cbox(1.00, 16.08, 7.0, 7.00,
     'SYSTEM 1 — C4.5 Classical Decision Process',
     fill='#EBF4FF', ec='#93C5FD')

# System 2
cbox(1.00, 1.28, 7.0, 15.10,
     'SYSTEM 2 — Metacognitive Decision Process',
     fill='#EDFAEE', ec='#86EFAC')

# ──────────────────────────────────────────────────────────
# BANNER TÍTULO
# ──────────────────────────────────────────────────────────
tb = FancyBboxPatch((0.90, 23.10), 7.20, 0.60,
                    boxstyle='round,pad=0,rounding_size=0.10',
                    fc='#F8FAFC', ec='#334155', lw=1.6, zorder=5)
ax.add_patch(tb)
ax.text(4.50, 23.42,
        'Framework MetaCog-C45 Execution Flow',
        ha='center', va='center', fontsize=11.5, fontweight='bold',
        color='#0F172A', fontfamily=FT, zorder=6)

# ──────────────────────────────────────────────────────────
# NODOS — SISTEMA 1
# ──────────────────────────────────────────────────────────
rbox(CX, Y['inicio'],    2.0, NH, C['tr_f'], C['tr_e'], 'Inicio', C['tr_t'], radius=0.24)
rbox(CX, Y['carga'],     NW, NH, C['s1_f'], C['s1_e'], 'Carga del Dataset', C['s1_t'])
rbox(CX, Y['preproc'],   NW, NH, C['s1_f'], C['s1_e'], 'Preprocesamiento', C['s1_t'])
rbox(CX, Y['entropia'],  NW, NH, C['s1_f'], C['s1_e'], 'Calcular Entropia', C['s1_t'])
rbox(CX, Y['info_gain'], NW, NH, C['s1_f'], C['s1_e'], 'Calcular Information Gain', C['s1_t'])
rbox(CX, Y['gain_ratio'],NW, NH, C['s1_f'], C['s1_e'], 'Calcular Gain Ratio', C['s1_t'])
rbox(CX, Y['sel_attr'],  NW, NH, C['s1_f'], C['s1_e'], 'Seleccionar Mejor Atributo', C['s1_t'])

# Decision
diam(CX, Y['hoja_q'], DW, DH, C['s1_f'], C['s1_e'], '¿Nodo Hoja?', C['s1_t'])

# Rama derecha
rbox(CXR, Y['crear_hoja'], 2.0, NH, C['s1_f'], C['s1_e'], 'Crear Nodo Hoja', C['s1_t'])
rbox(CXR, Y['fin_rama'],   1.80, NH, C['tr_f'], C['tr_e'], 'Fin Rama', C['tr_t'], radius=0.24)

# ──────────────────────────────────────────────────────────
# NODOS — SISTEMA 2
# ──────────────────────────────────────────────────────────
rbox(CX, Y['bootstrap'],   NW, NH, C['s2_f'], C['s2_e'], 'Bootstrap Validation', C['s2_t'])
rbox(CX, Y['node_stab'],   NW, NH, C['s2_f'], C['s2_e'], 'Calcular Node Stability (NS)', C['s2_t'])
rbox(CX, Y['thresh_surv'], NW, NH, C['s2_f'], C['s2_e'], 'Calcular Threshold Survival (TS)', C['s2_t'])
rbox(CX, Y['comp_est'],    NW, NH, C['s2_f'], C['s2_e'], 'Calcular Competitive Estimator (CE)', C['s2_t'])

# SCS — extendido con ecuación
rbox(CX, Y['scs'],  NW+0.50, EH, C['s2_f'], C['s2_e'],
     ['Calcular Split Confidence Score (SCS)'],
     C['s2_t'],
     eq='SCS = GR \u00b7 NS\u1d43 \u00b7 TS\u1d47 \u00b7 CE\u02b8',
     eq_fs=FSE)

# MetaMemory
rbox(CX, Y['consul_mm'],  NW, NH, C['mm_f'], C['mm_e'], 'Consultar MetaMemory', C['mm_t'])
rbox(CX, Y['recup_exp'],  NW, NH, C['mm_f'], C['mm_e'], 'Recuperar Experiencias Similares', C['mm_t'])
rbox(CX, Y['conf_hist'],  NW, NH, C['mm_f'], C['mm_e'], 'Calcular Confianza Historica (MH)', C['mm_t'])

# Reflection Engine — extendido con ecuación
rbox(CX, Y['reflection'], NW+0.50, EH, C['re_f'], C['re_e'],
     ['Reflection Engine'],
     C['re_t'],
     eq='MD = \u03bb \u00b7 SCS + (1 \u2212 \u03bb) \u00b7 MH',
     eq_fs=FSE)

# Decision MD >= tau
diam(CX, Y['md_q'], DW+0.20, DH, C['re_f'], C['re_e'], '¿MD \u2265 \u03c4 ?', C['re_t'])

# Rama derecha (No)
rbox(CXR, Y['reevaluar'],  2.0, NH, C['s2_f'], C['s2_e'], 'Reevaluar Particion', C['s2_t'])
rbox(CXR, Y['registrar'],  2.40, NH, C['mm_f'], C['mm_e'], 'Registrar en MetaMemory', C['mm_t'])
rbox(CXR, Y['buscar'],     2.40, NH, C['s2_f'], C['s2_e'], 'Buscar siguiente mejor atributo', C['s2_t'])

# Rama principal (Si)
rbox(CX, Y['aceptar'],    NW, NH, C['s2_f'], C['s2_e'], 'Aceptar Particion', C['s2_t'])
rbox(CX, Y['crear_nodo'], NW, NH, C['s2_f'], C['s2_e'], 'Crear Nodo', C['s2_t'])
rbox(CX, Y['hijo_izq'],   NW, NH, C['s2_f'], C['s2_e'], 'Procesar Hijo Izquierdo', C['s2_t'])
rbox(CX, Y['hijo_der'],   NW, NH, C['s2_f'], C['s2_e'], 'Procesar Hijo Derecho', C['s2_t'])

# Decision pendientes
diam(CX, Y['pend_q'], DW+0.40, DH+0.20, C['s2_f'], C['s2_e'],
     ['¿Existen nodos', 'pendientes?'], C['s2_t'])

# Finales
rbox(CX, Y['arbol_final'], NW, NH, C['s2_f'], C['s2_e'], 'Arbol Final Generado', C['s2_t'])
rbox(CX, Y['arbol_const'], NW+0.60, NH, C['s2_f'], C['s2_e'],
     'Arbol MetaCog-C45 Construido', C['s2_t'])
rbox(CX, Y['fin'], 2.0, NH, C['tr_f'], C['tr_e'], 'Fin', C['tr_t'], radius=0.24)

# ──────────────────────────────────────────────────────────
# FLECHAS — FLUJO VERTICAL PRINCIPAL
# ──────────────────────────────────────────────────────────
def bot(k, h=NH):  return Y[k] - h/2
def top(k, h=NH):  return Y[k] + h/2

# System 1
av(CX, bot('inicio'),    top('carga'))
av(CX, bot('carga'),     top('preproc'))
av(CX, bot('preproc'),   top('entropia'))
av(CX, bot('entropia'),  top('info_gain'))
av(CX, bot('info_gain'), top('gain_ratio'))
av(CX, bot('gain_ratio'),top('sel_attr'))
av(CX, bot('sel_attr'),  top('hoja_q', DH))

# hoja_q "No" → bootstrap
av(CX, bot('hoja_q', DH), top('bootstrap'))

# System 2 downward
av(CX, bot('bootstrap'),   top('node_stab'))
av(CX, bot('node_stab'),   top('thresh_surv'))
av(CX, bot('thresh_surv'), top('comp_est'))
av(CX, bot('comp_est'),    top('scs', EH))
av(CX, bot('scs', EH),     top('consul_mm'))
av(CX, bot('consul_mm'),   top('recup_exp'))
av(CX, bot('recup_exp'),   top('conf_hist'))
av(CX, bot('conf_hist'),   top('reflection', EH))
av(CX, bot('reflection', EH), top('md_q', DH))

# md_q "Si" → aceptar
av(CX, bot('md_q', DH),   top('aceptar'))
av(CX, bot('aceptar'),    top('crear_nodo'))
av(CX, bot('crear_nodo'), top('hijo_izq'))
av(CX, bot('hijo_izq'),   top('hijo_der'))
av(CX, bot('hijo_der'),   top('pend_q', DH+0.20))

# pendientes "No" → árbol
av(CX, bot('pend_q', DH+0.20), top('arbol_final'))
av(CX, bot('arbol_final'), top('arbol_const'))
av(CX, bot('arbol_const'), top('fin'))

# Rama derecha crear_hoja → fin_rama
av(CXR, bot('crear_hoja'), top('fin_rama'))

# Rama derecha reevaluar → registrar → buscar
av(CXR, bot('reevaluar'),  top('registrar'))
av(CXR, bot('registrar'),  top('buscar'))

# ──────────────────────────────────────────────────────────
# FLECHAS HORIZONTALES
# ──────────────────────────────────────────────────────────

# hoja_q → crear_hoja ("Si")
xL = CX + DW/2
xR = CXR - 2.0/2
ah(xL, xR, Y['hoja_q'])
lbl((xL+xR)/2, Y['hoja_q']+0.17, 'Si')

# hoja_q "No" label
lbl(CX-0.22, (bot('hoja_q', DH)+top('bootstrap'))/2, 'No', ha='right')

# md_q → reevaluar ("No")
xL2 = CX + (DW+0.20)/2
xR2 = CXR - 2.0/2
ah(xL2, xR2, Y['md_q'])
lbl((xL2+xR2)/2, Y['md_q']+0.17, 'No')

# md_q "Si" label
lbl(CX-0.22, (bot('md_q', DH)+top('aceptar'))/2, 'Si', ha='right')

# pend_q "No" label
lbl(CX-0.22, (bot('pend_q', DH+0.20)+top('arbol_final'))/2, 'No', ha='right')

# ──────────────────────────────────────────────────────────
# FLECHAS DE RETORNO (LOOPS)
# ──────────────────────────────────────────────────────────

# LOOP 1: buscar_sig → Calcular Gain Ratio  (rama "No" completa de reevaluación)
# Sale por la derecha de buscar, sube, llega a la derecha de gain_ratio
x_loop1 = 8.70
y_bus_top = top('buscar')
y_gr_mid  = Y['gain_ratio']

ax.plot([CXR + 2.40/2, x_loop1, x_loop1,
         CX + NW/2],
        [Y['buscar'],   Y['buscar'], y_gr_mid,
         y_gr_mid],
        color=C['arrow'], lw=1.0, ls='--', zorder=3)
ax.annotate('', xy=(CX + NW/2, y_gr_mid),
            xytext=(CX + NW/2 + 0.001, y_gr_mid),
            arrowprops=dict(arrowstyle='->', color=C['arrow'],
                            lw=1.0, mutation_scale=9), zorder=3)
ax.text(x_loop1 + 0.10,
        (Y['buscar'] + y_gr_mid)/2,
        'Reevaluar → Gain Ratio',
        ha='left', va='center', fontsize=6.0, color='#6B7280',
        fontfamily=FT, rotation=90, zorder=5)

# LOOP 2: pend_q "Si" → Calcular Entropia  (siguiente nodo pendiente)
x_loop2 = 0.32
y_pq_mid = Y['pend_q']
y_en_mid = Y['entropia']

ax.plot([CX - (DW+0.40)/2, x_loop2, x_loop2,
         CX - NW/2],
        [y_pq_mid,           y_pq_mid, y_en_mid,
         y_en_mid],
        color=C['arrow'], lw=1.0, ls='-', zorder=3)
ax.annotate('', xy=(CX - NW/2, y_en_mid),
            xytext=(CX - NW/2 - 0.001, y_en_mid),
            arrowprops=dict(arrowstyle='->', color=C['arrow'],
                            lw=1.0, mutation_scale=9), zorder=3)
lbl(CX - (DW+0.40)/2 - 0.15, y_pq_mid + 0.17, 'Si', ha='right')
ax.text(x_loop2 - 0.12,
        (y_pq_mid + y_en_mid)/2,
        'Procesar Nodo Pendiente',
        ha='right', va='center', fontsize=6.0, color='#6B7280',
        fontfamily=FT, rotation=90, zorder=5)

# ──────────────────────────────────────────────────────────
# SEPARADOR ENTRE S1 y S2
# ──────────────────────────────────────────────────────────
sep_y = 16.14
ax.plot([1.2, 7.8], [sep_y, sep_y],
        color='#CBD5E1', lw=0.8, ls=(0, (5, 3)), zorder=2)
ax.text(4.50, sep_y + 0.10,
        'Metacognitive Layer  (System 2)',
        ha='center', va='bottom', fontsize=6.5, color='#6B7280',
        fontfamily=FT, style='italic', zorder=5)

# ──────────────────────────────────────────────────────────
# LEYENDA
# ──────────────────────────────────────────────────────────
leg_x0, leg_y0 = 0.90, 0.30
leg_w, leg_h   = 7.20, 0.66

lbg = FancyBboxPatch((leg_x0, leg_y0), leg_w, leg_h,
                     boxstyle='round,pad=0,rounding_size=0.08',
                     fc='#F8FAFC', ec='#CBD5E1', lw=0.9, zorder=5)
ax.add_patch(lbg)

ax.text(leg_x0+0.18, leg_y0+leg_h/2,
        'LEGEND:', ha='left', va='center',
        fontsize=7.0, fontweight='bold', color='#374151',
        fontfamily=FT, zorder=6)

items = [
    (1.70, C['s1_f'], C['s1_e'], C['s1_t'], 'Classical Layer (C4.5)'),
    (3.35, C['s2_f'], C['s2_e'], C['s2_t'], 'Metacognitive Evaluation'),
    (5.15, C['re_f'], C['re_e'], C['re_t'], 'Reflection Engine'),
    (6.75, C['mm_f'], C['mm_e'], C['mm_t'], 'MetaMemory'),
]

for ix, ff, fe, ft, lbl_txt in items:
    sq = FancyBboxPatch((ix, leg_y0+leg_h/2-0.11), 0.22, 0.22,
                        boxstyle='round,pad=0,rounding_size=0.03',
                        fc=ff, ec=fe, lw=1.0, zorder=6)
    ax.add_patch(sq)
    ax.text(ix+0.30, leg_y0+leg_h/2, lbl_txt,
            ha='left', va='center', fontsize=7.0, fontweight='bold',
            color=ft, fontfamily=FT, zorder=6)

# ──────────────────────────────────────────────────────────
# GUARDAR
# ──────────────────────────────────────────────────────────
plt.tight_layout(pad=0)

for ext in ('pdf', 'png'):
    fname = f'metacog_c45_flowchart.{ext}'
    fig.savefig(fname, format=ext, bbox_inches='tight',
                facecolor=C['bg'],
                dpi=300 if ext == 'png' else 150)
    print(f'[OK] {ext.upper()} guardado: {fname}')
