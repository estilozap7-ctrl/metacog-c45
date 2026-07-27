import os

def create_svg():
    # SVG canvas dimensions
    width = 920
    height = 2050
    
    # Color Palette definitions (academic, soft colors)
    # Styles: (fill, stroke, text_color, rx, ry, stroke_width)
    styles = {
        'start_end': {
            'fill': '#F1F5F9',
            'stroke': '#475569',
            'text': '#1E293B',
            'stroke_width': 1.5,
            'rx': 18,
            'ry': 18
        },
        'system1': {
            'fill': '#EFF6FF',
            'stroke': '#1D4ED8',
            'text': '#1E3A8A',
            'stroke_width': 1.5,
            'rx': 6,
            'ry': 6
        },
        'system2': {
            'fill': '#F0FDF4',
            'stroke': '#15803D',
            'text': '#14532D',
            'stroke_width': 1.5,
            'rx': 6,
            'ry': 6
        },
        'metamemory': {
            'fill': '#FFF7ED',
            'stroke': '#EA580C',
            'text': '#7C2D12',
            'stroke_width': 1.5,
            'rx': 6,
            'ry': 6
        },
        'reflection': {
            'fill': '#FAF5FF',
            'stroke': '#7E22CE',
            'text': '#581C87',
            'stroke_width': 1.5,
            'rx': 6,
            'ry': 6
        },
        'decision': {
            'fill': '#EFF6FF',
            'stroke': '#1D4ED8',
            'text': '#1E3A8A',
            'stroke_width': 1.5,
        },
        'decision_reflection': {
            'fill': '#FAF5FF',
            'stroke': '#7E22CE',
            'text': '#581C87',
            'stroke_width': 1.5,
        },
        'decision_system2': {
            'fill': '#F0FDF4',
            'stroke': '#15803D',
            'text': '#14532D',
            'stroke_width': 1.5,
        }
    }

    # Node definitions: (id, label, type, cx, cy, w, h, icon, equation)
    nodes = [
        # SYSTEM 1
        ('inicio', 'Inicio', 'start_end', 460, 150, 90, 36, None, None),
        ('carga', 'Carga del Dataset', 'system1', 460, 210, 180, 42, 'dataset', None),
        ('preproc', 'Preprocesamiento', 'system1', 460, 270, 180, 42, None, None),
        ('entropia', 'Calcular Entropía', 'system1', 460, 330, 180, 42, None, None),
        ('info_gain', 'Calcular Information Gain', 'system1', 460, 390, 200, 42, None, None),
        ('gain_ratio', 'Calcular Gain Ratio', 'system1', 460, 450, 180, 42, None, None),
        ('sel_attr', 'Seleccionar Mejor Atributo', 'system1', 460, 510, 220, 42, None, None),
        ('hoja_q', '¿Nodo Hoja?', 'decision', 460, 580, 120, 50, None, None),
        ('crear_hoja', 'Crear Nodo Hoja', 'system1', 640, 580, 130, 42, None, None),
        ('fin_rama', 'Fin Rama', 'start_end', 640, 645, 90, 36, None, None),
        
        # SYSTEM 2
        ('bootstrap', 'Bootstrap Validation', 'system2', 460, 750, 180, 42, None, None),
        ('node_stability', 'Calcular Node Stability (NS)', 'system2', 460, 810, 220, 42, None, None),
        ('threshold_surv', 'Calcular Threshold Survival (TS)', 'system2', 460, 870, 220, 42, None, None),
        ('comp_estimator', 'Calcular Competitive Estimator (CE)', 'system2', 460, 930, 240, 42, None, None),
        ('scs', 'Calcular Split Confidence Score (SCS)', 'system2', 460, 1005, 270, 64, None, 'SCS = GR · NS^α · TS^β · CE^γ'),
        ('consultar_mm', 'Consultar MetaMemory', 'metamemory', 460, 1085, 200, 42, 'memory', None),
        ('recuperar_exp', 'Recuperar Experiencias Similares', 'metamemory', 460, 1145, 230, 42, None, None),
        ('conf_hist', 'Calcular Confianza Histórica (MH)', 'metamemory', 460, 1205, 240, 42, None, None),
        ('reflection', 'Reflection Engine', 'reflection', 460, 1280, 270, 64, 'brain', 'MD = λ · SCS + (1 − λ) · MH'),
        ('md_q', '¿MD ≥ τ ?', 'decision_reflection', 460, 1360, 120, 50, None, None),
        
        # Branch No of MD >= tau
        ('reevaluar', 'Reevaluar Partición', 'system2', 640, 1360, 140, 42, None, None),
        ('registrar_mm', 'Registrar experiencia en MetaMemory', 'metamemory', 640, 1425, 250, 42, 'memory', None),
        ('buscar_sig', 'Buscar siguiente mejor atributo', 'system2', 640, 1490, 230, 42, None, None),
        
        # Branch Si of MD >= tau
        ('aceptar', 'Aceptar Partición', 'system2', 460, 1445, 180, 42, None, None),
        ('crear_nodo', 'Crear Nodo', 'system2', 460, 1505, 180, 42, None, None),
        ('hijo_izq', 'Procesar Hijo Izquierdo', 'system2', 460, 1565, 190, 42, None, None),
        ('hijo_der', 'Procesar Hijo Derecho', 'system2', 460, 1625, 190, 42, None, None),
        ('pendientes_q', '¿Existen nodos pendientes?', 'decision_system2', 460, 1700, 160, 56, None, None),
        
        # Branch No of pendientes_q
        ('arbol_final', 'Árbol Final', 'system2', 460, 1785, 180, 42, None, None),
        ('arbol_construido', 'Árbol MetaCog-C45 construido', 'system2', 460, 1845, 250, 42, 'tree', None),
        ('fin', 'Fin', 'start_end', 460, 1905, 90, 36, None, None),
    ]

    # Convert nodes list to dictionary for lookup
    node_dict = {n[0]: n for n in nodes}

    svg_content = []
    
    # 1. Header and styles
    svg_content.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%" style="background-color: #FFFFFF; font-family: system-ui, -apple-system, \'Segoe UI\', Roboto, Helvetica, Arial, sans-serif;">')
    
    # Defs for drop shadow and markers
    svg_content.append('''  <defs>
    <filter id="shadow" x="-10%" y="-10%" width="120%" height="120%">
      <feDropShadow dx="0" dy="1.5" stdDeviation="2.5" flood-color="#0F172A" flood-opacity="0.07"/>
    </filter>
    <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#1E293B" />
    </marker>
  </defs>''')

    # Define icons as inline paths helpers
    icons = {
        'dataset': '<g stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" fill="none"><path d="M2 13h12M4 13V9m4 4V4m4 9V7" /></g>',
        'memory': '<g stroke="currentColor" stroke-width="1.5" stroke-linejoin="round" stroke-linecap="round" fill="none"><path d="M2,12 L12,12 L12,4.5 L10.5,3 L2,3 Z" /><rect x="4" y="3" width="3.5" height="2.5" stroke-width="1" /><rect x="4.5" y="8.5" width="5" height="3.5" stroke-width="1" /></g>',
        'brain': '<g stroke="currentColor" stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round" fill="none"><path d="M8,14 C8,14 6,13 5,11.5 C4,10 4,9 4.5,8 C4.2,7.5 4,7 4,6.5 C4,5 5.2,3.5 7,3.5 C7.5,3.5 8,3.7 8.5,4 C9,3.7 9.5,3.5 10,3.5 C11.8,3.5 13,5 13,6.5 C13,7 12.8,7.5 12.5,8 C13,9 13,10 12,11.5 C11,13 9,14 9,14" /><path d="M8.5,4 L8.5,13.5" stroke-dasharray="1 1" /></g>',
        'tree': '<g stroke="currentColor" stroke-width="1.5" stroke-linejoin="round" stroke-linecap="round" fill="none"><circle cx="8" cy="3.5" r="1.5" fill="currentColor"/><circle cx="4.5" cy="11.5" r="1.5" fill="currentColor"/><circle cx="11.5" cy="11.5" r="1.5" fill="currentColor"/><line x1="8" y1="5" x2="5.2" y2="10" /><line x1="8" y1="5" x2="10.8" y2="10" /></g>'
    }

    # 2. Draw Title Card
    svg_content.append('''  <!-- TITLE CARD -->
  <rect x="180" y="25" width="560" height="55" rx="8" fill="#F8FAFC" stroke="#334155" stroke-width="1.5" filter="url(#shadow)" />
  <text x="460" y="58" text-anchor="middle" font-size="18" font-weight="700" fill="#0F172A" letter-spacing="0.5">Framework MetaCog-C45 Execution Flow</text>''')

  # 3. Draw Containers (System 1 and System 2 bounds)
    svg_content.append('''
  <!-- CONTAINERS -->
  <!-- System 1 Container -->
  <rect x="180" y="105" width="560" height="580" rx="8" fill="#FBFDFE" stroke="#94A3B8" stroke-width="1" stroke-dasharray="4 4" />
  <text x="195" y="128" font-size="12" font-weight="700" fill="#475569" letter-spacing="0.5">SYSTEM 1: C4.5 CLASSICAL DECISION PROCESS</text>
  
  <!-- System 2 Container -->
  <rect x="180" y="705" width="560" height="1245" rx="8" fill="#FAFCFA" stroke="#94A3B8" stroke-width="1" stroke-dasharray="4 4" />
  <text x="195" y="728" font-size="12" font-weight="700" fill="#475569" letter-spacing="0.5">SYSTEM 2: METACOGNITIVE DECISION PROCESS</text>''')

    svg_content.append('\n  <!-- NODES -->')
    # 4. Draw Nodes
    for node_id, label, n_type, cx, cy, w, h, icon, eq in nodes:
        st = styles[n_type]
        svg_content.append(f'  <!-- Node: {node_id} -->')
        
        # Decide shape
        if 'decision' in n_type:
            # Diamond shape points: top, right, bottom, left
            p1 = f"{cx},{cy - h/2}"
            p2 = f"{cx + w/2},{cy}"
            p3 = f"{cx},{cy + h/2}"
            p4 = f"{cx - w/2},{cy}"
            svg_content.append(f'  <polygon points="{p1} {p2} {p3} {p4}" fill="{st["fill"]}" stroke="{st["stroke"]}" stroke-width="{st["stroke_width"]}" filter="url(#shadow)"/>')
            # Text label
            svg_content.append(f'  <text x="{cx}" y="{cy}" text-anchor="middle" dominant-baseline="central" font-size="12" font-weight="600" fill="{st["text"]}">{label}</text>')
        else:
            # Rounded Rectangle shape
            x = cx - w/2
            y = cy - h/2
            svg_content.append(f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{st["rx"]}" ry="{st["ry"]}" fill="{st["fill"]}" stroke="{st["stroke"]}" stroke-width="{st["stroke_width"]}" filter="url(#shadow)"/>')
            
            # Text layout (handle icons & equations)
            if eq:
                # Two line label with equation
                # Text title
                title_y = cy - 9
                if icon:
                    # Combined width calculation for center-aligning icon and title
                    icon_w = 16
                    gap = 6
                    char_w = 7.2
                    text_w = len(label) * char_w
                    total_w = icon_w + gap + text_w
                    start_x = cx - total_w/2
                    
                    # Draw icon
                    svg_content.append(f'  <g transform="translate({start_x}, {title_y - 8})">{icons[icon]}</g>')
                    # Draw text
                    svg_content.append(f'  <text x="{start_x + icon_w + gap}" y="{title_y}" text-anchor="start" dominant-baseline="central" font-size="12" font-weight="600" fill="{st["text"]}">{label}</text>')
                else:
                    svg_content.append(f'  <text x="{cx}" y="{title_y}" text-anchor="middle" dominant-baseline="central" font-size="12" font-weight="600" fill="{st["text"]}">{label}</text>')
                
                # Draw equation
                eq_y = cy + 13
                # Replace ^a, ^b, ^g with superscripts in SVG tspans
                if '^' in eq:
                    parts = eq.split('·')
                    svg_eq = []
                    for part in parts:
                        part = part.strip()
                        if '^' in part:
                            base, exp = part.split('^')
                            # Map exp
                            if exp == 'α': exp_sym = 'α'
                            elif exp == 'β': exp_sym = 'β'
                            elif exp == 'γ': exp_sym = 'γ'
                            else: exp_sym = exp
                            svg_eq.append(f'{base}<tspan dy="-3" font-size="8">{exp_sym}</tspan><tspan dy="3" font-size="11"> </tspan>')
                        else:
                            svg_eq.append(part)
                    eq_formatted = ' · '.join(svg_eq)
                    svg_content.append(f'  <text x="{cx}" y="{eq_y}" text-anchor="middle" dominant-baseline="central" font-size="11" font-style="italic" fill="#334155">{eq_formatted}</text>')
                else:
                    svg_content.append(f'  <text x="{cx}" y="{eq_y}" text-anchor="middle" dominant-baseline="central" font-size="11" font-style="italic" fill="#334155">{eq}</text>')
            else:
                # Single line label, handle icon if present
                if icon:
                    icon_w = 16
                    gap = 6
                    char_w = 7.2
                    text_w = len(label) * char_w
                    total_w = icon_w + gap + text_w
                    start_x = cx - total_w/2
                    
                    # Draw icon
                    svg_content.append(f'  <g transform="translate({start_x}, {cy - 8})">{icons[icon]}</g>')
                    # Draw text
                    svg_content.append(f'  <text x="{start_x + icon_w + gap}" y="{cy}" text-anchor="start" dominant-baseline="central" font-size="12" font-weight="600" fill="{st["text"]}">{label}</text>')
                else:
                    svg_content.append(f'  <text x="{cx}" y="{cy}" text-anchor="middle" dominant-baseline="central" font-size="12" font-weight="600" fill="{st["text"]}">{label}</text>')

    svg_content.append('\n  <!-- CONNECTIONS -->')
    # 5. Draw Connections (Arrows)
    
    # Straight vertical connections list of (node_from, node_to)
    verticals = [
        ('inicio', 'carga'),
        ('carga', 'preproc'),
        ('preproc', 'entropia'),
        ('entropia', 'info_gain'),
        ('info_gain', 'gain_ratio'),
        ('gain_ratio', 'sel_attr'),
        ('sel_attr', 'hoja_q'),
        ('crear_hoja', 'fin_rama'),
        ('bootstrap', 'node_stability'),
        ('node_stability', 'threshold_surv'),
        ('threshold_surv', 'comp_estimator'),
        ('comp_estimator', 'scs'),
        ('scs', 'consultar_mm'),
        ('consultar_mm', 'recuperar_exp'),
        ('recuperar_exp', 'conf_hist'),
        ('conf_hist', 'reflection'),
        ('reflection', 'md_q'),
        ('reevaluar', 'registrar_mm'),
        ('registrar_mm', 'buscar_sig'),
        ('aceptar', 'crear_nodo'),
        ('crear_nodo', 'hijo_izq'),
        ('hijo_izq', 'hijo_der'),
        ('hijo_der', 'pendientes_q'),
        ('arbol_final', 'arbol_construido'),
        ('arbol_construido', 'fin'),
    ]

    for n_from_id, n_to_id in verticals:
        from_n = node_dict[n_from_id]
        to_n = node_dict[n_to_id]
        
        # From bottom of from_n to top of to_n
        x = from_n[3]
        y1 = from_n[4] + from_n[6]/2
        y2 = to_n[4] - to_n[6]/2
        
        # Add a tiny gap before arrow tip
        y2_arrow = y2 - 4
        
        svg_content.append(f'  <!-- {n_from_id} -> {n_to_id} -->')
        svg_content.append(f'  <line x1="{x}" y1="{y1}" x2="{x}" y2="{y2_arrow}" stroke="#000000" stroke-width="1.5" marker-end="url(#arrow)" />')

    # Special connections
    # 1. hoja_q -> crear_hoja ("Si" branch, horizontal to the right)
    n_from = node_dict['hoja_q']
    n_to = node_dict['crear_hoja']
    x1 = n_from[3] + n_from[5]/2
    y = n_from[4]
    x2 = n_to[3] - n_to[5]/2 - 4
    svg_content.append('  <!-- hoja_q -> crear_hoja (Si) -->')
    svg_content.append(f'  <line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="#000000" stroke-width="1.5" marker-end="url(#arrow)" />')
    svg_content.append(f'  <text x="{(x1+x2)/2}" y="{y-6}" text-anchor="middle" font-size="11" font-weight="700" fill="#1E293B">Sí</text>')

    # 2. hoja_q -> bootstrap ("No" branch, vertical down)
    n_from = node_dict['hoja_q']
    n_to = node_dict['bootstrap']
    x = n_from[3]
    y1 = n_from[4] + n_from[6]/2
    y2 = n_to[4] - n_to[6]/2 - 4
    svg_content.append('  <!-- hoja_q -> bootstrap (No) -->')
    svg_content.append(f'  <line x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" stroke="#000000" stroke-width="1.5" marker-end="url(#arrow)" />')
    svg_content.append(f'  <text x="{x-10}" y="{(y1+y2)/2}" text-anchor="end" dominant-baseline="central" font-size="11" font-weight="700" fill="#1E293B">No</text>')

    # 3. md_q -> reevaluar ("No" branch, horizontal to the right)
    n_from = node_dict['md_q']
    n_to = node_dict['reevaluar']
    x1 = n_from[3] + n_from[5]/2
    y = n_from[4]
    x2 = n_to[3] - n_to[5]/2 - 4
    svg_content.append('  <!-- md_q -> reevaluar (No) -->')
    svg_content.append(f'  <line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="#000000" stroke-width="1.5" marker-end="url(#arrow)" />')
    svg_content.append(f'  <text x="{(x1+x2)/2}" y="{y-6}" text-anchor="middle" font-size="11" font-weight="700" fill="#1E293B">No</text>')

    # 4. md_q -> aceptar ("Si" branch, vertical down)
    n_from = node_dict['md_q']
    n_to = node_dict['aceptar']
    x = n_from[3]
    y1 = n_from[4] + n_from[6]/2
    y2 = n_to[4] - n_to[6]/2 - 4
    svg_content.append('  <!-- md_q -> aceptar (Si) -->')
    svg_content.append(f'  <line x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" stroke="#000000" stroke-width="1.5" marker-end="url(#arrow)" />')
    svg_content.append(f'  <text x="{x-10}" y="{(y1+y2)/2}" text-anchor="end" dominant-baseline="central" font-size="11" font-weight="700" fill="#1E293B">Sí</text>')

    # 5. pendientes_q -> arbol_final ("No" branch, vertical down)
    n_from = node_dict['pendientes_q']
    n_to = node_dict['arbol_final']
    x = n_from[3]
    y1 = n_from[4] + n_from[6]/2
    y2 = n_to[4] - n_to[6]/2 - 4
    svg_content.append('  <!-- pendientes_q -> arbol_final (No) -->')
    svg_content.append(f'  <line x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" stroke="#000000" stroke-width="1.5" marker-end="url(#arrow)" />')
    svg_content.append(f'  <text x="{x-10}" y="{(y1+y2)/2}" text-anchor="end" dominant-baseline="central" font-size="11" font-weight="700" fill="#1E293B">No</text>')

    # 6. Loop: buscar_sig -> gain_ratio (Reevaluación loop, goes right and up)
    # Starts from right edge of buscar_sig (cx=640, cy=1490, w=230, right_x=755)
    # Ends at right edge of gain_ratio (cx=460, cy=450, w=180, right_x=550)
    # We trace outside the containers at x = 800
    svg_content.append('  <!-- Loop: buscar_sig -> gain_ratio -->')
    r = 12 # corner radius
    path_gain_ratio = (
        f"M 755,1490 "
        f"L {800-r},1490 Q 800,1490 800,1490-r "
        f"L 800,{450+r} Q 800,450 {800-r},450 "
        f"L 554,450"
    )
    svg_content.append(f'  <path d="{path_gain_ratio}" fill="none" stroke="#000000" stroke-width="1.5" stroke-dasharray="3 3" marker-end="url(#arrow)" />')
    # Loop label
    svg_content.append(f'  <text x="808" y="970" text-anchor="start" font-size="10" font-weight="600" fill="#475569" transform="rotate(90 808 970)">Reevaluar Partición (Volver a C4.5)</text>')

    # 7. Loop: pendientes_q -> entropia ("Si" branch, goes left and up)
    # Starts from left point of pendientes_q (cx=460, cy=1700, w=160, left_x=380)
    # Ends at left edge of entropia (cx=460, cy=330, w=180, left_x=370)
    # We trace outside the containers at x = 100
    svg_content.append('  <!-- Loop: pendientes_q -> entropia (Si) -->')
    path_entropia = (
        f"M 380,1700 "
        f"L {100+r},1700 Q 100,1700 100,1700-r "
        f"L 100,{330+r} Q 100,330 {100+r},330 "
        f"L 366,330"
    )
    svg_content.append(f'  <path d="{path_entropia}" fill="none" stroke="#000000" stroke-width="1.5" marker-end="url(#arrow)" />')
    svg_content.append(f'  <text x="350" y="1694" text-anchor="middle" font-size="11" font-weight="700" fill="#1E293B">Sí</text>')
    # Loop label on vertical line
    svg_content.append(f'  <text x="92" y="1015" text-anchor="middle" font-size="10" font-weight="600" fill="#475569" transform="rotate(-90 92 1015)">Procesar Siguiente Nodo Pendiente</text>')

    svg_content.append('''  <!-- LEGEND -->
  <g transform="translate(180, 1965)">
    <rect x="0" y="0" width="560" height="40" rx="6" fill="#F8FAFC" stroke="#CBD5E1" stroke-width="1" />
    <text x="15" y="24" font-size="10" font-weight="700" fill="#475569" letter-spacing="0.5">LEYENDA</text>
    
    <!-- Item 1: Blue -->
    <rect x="85" y="13" width="12" height="12" rx="2" fill="#EFF6FF" stroke="#1D4ED8" stroke-width="1.5" />
    <text x="102" y="23" font-size="10" font-weight="600" fill="#1E3A8A">Classical Layer</text>
    
    <!-- Item 2: Green -->
    <rect x="195" y="13" width="12" height="12" rx="2" fill="#F0FDF4" stroke="#15803D" stroke-width="1.5" />
    <text x="212" y="23" font-size="10" font-weight="600" fill="#14532D">Metacognitive Eval.</text>
    
    <!-- Item 3: Purple -->
    <rect x="330" y="13" width="12" height="12" rx="2" fill="#FAF5FF" stroke="#7E22CE" stroke-width="1.5" />
    <text x="347" y="23" font-size="10" font-weight="600" fill="#581C87">Reflection Engine</text>
    
    <!-- Item 4: Orange -->
    <rect x="455" y="13" width="12" height="12" rx="2" fill="#FFF7ED" stroke="#EA580C" stroke-width="1.5" />
    <text x="472" y="23" font-size="10" font-weight="600" fill="#7C2D12">MetaMemory</text>
  </g>''')

    # Close SVG
    svg_content.append('</svg>')
    
    output_path = os.path.join(os.getcwd(), 'metacog_c45_flowchart.svg')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(svg_content))
    print(f"Successfully generated SVG flowchart at: {output_path}")

if __name__ == '__main__':
    create_svg()
