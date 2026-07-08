/**
 * =========================================================
 * PyC45 Technical Report — report.js
 * =========================================================
 * Carga el JSON generado por generate_report.py y construye
 * dinámicamente todos los elementos del informe técnico.
 * =========================================================
 */

/* =========================================================
   Utilidades
   ========================================================= */

const fmt = {
  pct:  v => `${(v * 100).toFixed(2)}%`,
  num:  v => v.toLocaleString('es-CO'),
  dec4: v => v.toFixed(4),
  dec2: v => v.toFixed(2),
};

function tag(value, thresholds = { good: 0.75, ok: 0.55 }) {
  if (value >= thresholds.good) return `<span class="tag good">Bueno</span>`;
  if (value >= thresholds.ok)  return `<span class="tag ok">Moderado</span>`;
  return `<span class="tag bad">Bajo</span>`;
}

function colorClass(value, thresholds = { success: 0.75, warning: 0.55 }) {
  if (value >= thresholds.success) return 'success';
  if (value >= thresholds.warning) return 'warning';
  return 'danger';
}

function setImg(id, b64) {
  const el = document.getElementById(id);
  if (el && b64) el.src = `data:image/png;base64,${b64}`;
}

function setText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

function setHTML(id, html) {
  const el = document.getElementById(id);
  if (el) el.innerHTML = html;
}

/** Crear stat-box HTML */
function statBox(label, value, cls = 'accent') {
  return `
    <div class="stat-box">
      <div class="stat-value ${cls}">${value}</div>
      <div class="stat-label">${label}</div>
    </div>`;
}

/* =========================================================
   Observador de Scroll — animaciones fade-in
   ========================================================= */
function setupScrollObserver() {
  const els = document.querySelectorAll('.fade-in');
  const obs = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        e.target.classList.add('visible');
        obs.unobserve(e.target);
      }
    });
  }, { threshold: 0.08 });
  els.forEach(el => obs.observe(el));
}

/* =========================================================
   TOC — resaltar sección activa
   ========================================================= */
function setupTOC() {
  const sections = document.querySelectorAll('section[id]');
  const links = document.querySelectorAll('.toc-link');

  const obs = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        links.forEach(l => l.classList.remove('active'));
        const active = document.querySelector(`.toc-link[href="#${e.target.id}"]`);
        if (active) {
          active.classList.add('active');
          active.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
        }
      }
    });
  }, { rootMargin: '-40% 0px -50% 0px' });

  sections.forEach(s => obs.observe(s));
}

/* =========================================================
   Builders de secciones
   ========================================================= */

function buildDataset(d) {
  // KPI superiores
  const pctImpago = (((d.class_distribution_sample['1'] || 0) / d.sample_size) * 100).toFixed(1);
  const labelPct = d.target_column && d.target_column.toLowerCase().includes('default') 
    ? '% Impago (muestra)' 
    : (d.target_column && d.target_column.toLowerCase().includes('churn') ? '% Churn (muestra)' : '% Positivo (1) (muestra)');

  setHTML('datasetKpis', [
    statBox('Registros Totales', fmt.num(d.total_rows), 'accent'),
    statBox('Variables', d.total_cols, 'info'),
    statBox('Muestra Analizada', fmt.num(d.sample_size), 'warning'),
    statBox(labelPct, `${pctImpago}%`, 'danger'),
    statBox('Entrenamiento', fmt.num(d.train_size), 'success'),
    statBox('Validación',   fmt.num(d.val_size), 'success'),
  ].join(''));
}

function buildTreeBefore(tb) {
  // Stats superiores
  setHTML('treeBeforeKpis', [
    statBox('Nodos Totales', tb.stats.nodes, 'accent'),
    statBox('Nodos Hoja',   tb.stats.leaves, 'success'),
    statBox('Profundidad',  tb.stats.depth,  'warning'),
    statBox('Accuracy (Train)', fmt.pct(tb.accuracy_train), 'info'),
    statBox('Accuracy (Val)',   fmt.pct(tb.accuracy_val),   colorClass(tb.accuracy_val)),
  ].join(''));

  // Árbol texto
  setText('treeBeforeText', tb.text || '(sin árbol)');

  // Tabla interpretación — extrae variables únicas del árbol
  const varInterpretations = {
    'PAY_0':     { desc: 'Estado de pago más reciente (septiembre)', insight: 'Variable con mayor poder discriminatorio. Retrasos recientes predicen fuertemente el impago.' },
    'BILL_AMT1': { desc: 'Monto facturado en septiembre (NT$)',      insight: 'Clientes con facturas muy altas tienen perfiles de riesgo distintos.' },
    'PAY_2':     { desc: 'Estado de pago en agosto',                 insight: 'Retraso en dos meses consecutivos es señal fuerte de impago.' },
    'LIMIT_BAL': { desc: 'Límite de crédito asignado (NT$)',         insight: 'Clientes con mayor límite tienden a tener mejor historial crediticio.' },
    'PAY_AMT1':  { desc: 'Pago realizado en septiembre (NT$)',       insight: 'Pagos mínimos frecuentes son indicadores de estrés financiero.' },
  };

  // Extraer variables del texto del árbol
  const matches = [...(tb.text || '').matchAll(/📊 (\w+) ≤ ([\d.]+)/g)];
  const rows = matches.map(m => {
    const varName = m[1];
    const threshold = parseFloat(m[2]);
    const info = varInterpretations[varName] || { desc: varName, insight: 'Variable numérica continua evaluada por su Gain Ratio.' };
    return `<tr>
      <td><code>${varName}</code></td>
      <td><code>${threshold}</code></td>
      <td>${info.desc}. ${info.insight}</td>
    </tr>`;
  });

  setHTML('treeInterpretationTable', rows.length > 0 ? rows.join('') : '<tr><td colspan="3">Sin nodos internos detectados.</td></tr>');
}

function buildPruning(before, after) {
  // Comparación antes/después
  setHTML('treeBefore3Kpis', [
    statBox('Nodos', before.stats.nodes, 'accent'),
    statBox('Hojas', before.stats.leaves, 'success'),
    statBox('Profundidad', before.stats.depth, 'warning'),
  ].join(''));
  setHTML('treeAfter3Kpis', [
    statBox('Nodos', after.stats.nodes, 'success'),
    statBox('Hojas', after.stats.leaves, 'success'),
    statBox('Profundidad', after.stats.depth, 'success'),
  ].join(''));

  const el1 = document.getElementById('accBefore');
  if (el1) el1.textContent = fmt.pct(before.accuracy_val);
  const el2 = document.getElementById('accAfter');
  if (el2) el2.textContent = fmt.pct(after.accuracy_val);

  setText('treeAfterText', after.text || '(árbol vacío tras poda)');

  // Referencias en conclusiones
  setText('nodesBeforeConc', before.stats.nodes);
  setText('nodesAfterConc', after.stats.nodes);
}

function buildMetrics(cm, metrics) {
  // Tabla CM
  const cmRows = [
    { type: 'TP (Verdadero Positivo)', desc: 'Impago predicho correctamente', val: cm.tp },
    { type: 'TN (Verdadero Negativo)', desc: 'No-impago predicho correctamente', val: cm.tn },
    { type: 'FP (Falso Positivo)',     desc: 'No-impago clasificado erróneamente como impago', val: cm.fp },
    { type: 'FN (Falso Negativo)',     desc: 'Impago NO detectado (error crítico en crédito)', val: cm.fn },
    { type: 'Total',                   desc: 'Registros de validación evaluados', val: cm.total },
  ];
  setHTML('cmTable', cmRows.map(r =>
    `<tr><td><strong>${r.type}</strong></td><td>${r.desc}</td><td><strong>${r.val}</strong></td></tr>`
  ).join(''));

  // Definiciones y fórmulas para la tabla de métricas
  const metaDefs = {
    'Accuracy':          { formula: '(TP + TN) / Total',        interp: 'Proporción de predicciones correctas sobre el total.' },
    'Precision':         { formula: 'TP / (TP + FP)',            interp: 'De los clasificados como impago, ¿cuántos realmente lo son? Mide falsas alarmas.' },
    'Recall':            { formula: 'TP / (TP + FN)',            interp: 'De los clientes que incumplirán, ¿cuántos detectamos? Crítico para riesgo crediticio.' },
    'Specificity':       { formula: 'TN / (TN + FP)',            interp: 'De los que no incumplirán, ¿cuántos identificamos correctamente?' },
    'F1-Score':          { formula: '2 · P · R / (P + R)',       interp: 'Media armónica de Precisión y Recall. Equilibrio entre ambas métricas.' },
    'Balanced Accuracy': { formula: '(Recall + Specificity) / 2',interp: 'Exactitud ponderada por clase. Más justa que Accuracy con datos desbalanceados.' },
    'Error Rate':        { formula: '1 − Accuracy',              interp: 'Proporción de predicciones incorrectas.' },
    'MCC':               { formula: '(TP·TN − FP·FN) / √(…)',   interp: 'Coeficiente de correlación de Matthews. Métrica robusta ante desbalance de clases.' },
  };

  setHTML('metricsTable', Object.entries(metrics).map(([name, val]) => {
    const def = metaDefs[name] || { formula: '—', interp: '—' };
    const cls = colorClass(val, { success: 0.75, warning: 0.50 });
    return `<tr>
      <td><strong>${name}</strong></td>
      <td><span class="stat-value ${cls}" style="font-size:1rem;">${fmt.pct(val)}</span></td>
      <td><code>${def.formula}</code></td>
      <td style="font-size:0.85rem;">${def.interp}</td>
    </tr>`;
  }).join(''));
}

function buildROC(auc) {
  const el = document.getElementById('aucValue');
  if (el) el.textContent = auc.toFixed(4);

  let cls = 'danger', interpText = '';
  if (auc >= 0.90) { cls = 'success'; interpText = `AUC de ${auc.toFixed(4)} → Clasificación excelente. El modelo discrimina muy bien entre clientes con y sin riesgo de impago.`; }
  else if (auc >= 0.80) { cls = 'success'; interpText = `AUC de ${auc.toFixed(4)} → Clasificación buena. El modelo tiene buen poder predictivo.`; }
  else if (auc >= 0.70) { cls = 'warning'; interpText = `AUC de ${auc.toFixed(4)} → Clasificación aceptable. El modelo discrimina de manera moderada entre las clases.`; }
  else if (auc >= 0.60) { cls = 'warning'; interpText = `AUC de ${auc.toFixed(4)} → Discriminación débil pero superior al azar. El árbol C4.5 captura patrones reales pero limitados por el submuestreo.`; }
  else { interpText = `AUC de ${auc.toFixed(4)} → Cercano al azar. Se recomienda revisar el preprocesamiento y ampliar la muestra.`; }

  if (el) el.className = `stat-value ${cls}`;
  setHTML('aucInterpretation', interpText);
}

function buildFeatureImportance(fi) {
  setHTML('fiTable', fi.map((item, i) => {
    const pct = (item.importance * 100).toFixed(2);
    const bar = `<div style="width:${pct}%;height:6px;background:var(--accent);border-radius:3px;display:inline-block;min-width:4px;"></div>`;
    const varDescriptions = {
      'PAY_0':     'Estado de pago del mes más reciente. Indicador primario de comportamiento crediticio.',
      'BILL_AMT1': 'Monto de deuda facturada más reciente. Refleja nivel de endeudamiento actual.',
      'PAY_2':     'Estado de pago de hace dos meses. Complementa PAY_0 para tendencia histórica.',
      'LIMIT_BAL': 'Límite de crédito asignado. Proxy de historial y solvencia del cliente.',
      'PAY_AMT1':  'Pago más reciente realizado. Clientes que pagan montos bajos tienen más riesgo.',
    };
    const desc = varDescriptions[item.feature] || 'Variable predictora seleccionada por el algoritmo C4.5.';
    return `<tr>
      <td><strong>#${i + 1}</strong></td>
      <td><code>${item.feature}</code></td>
      <td>${bar} <strong>${pct}%</strong></td>
      <td style="font-size:0.85rem;">${desc}</td>
    </tr>`;
  }).join(''));
}

function buildFinalKpis(metrics, auc, after) {
  setHTML('finalKpis', [
    statBox('Accuracy Final', fmt.pct(metrics['Accuracy']), colorClass(metrics['Accuracy'])),
    statBox('Recall',         fmt.pct(metrics['Recall']),   colorClass(metrics['Recall'])),
    statBox('Precision',      fmt.pct(metrics['Precision']), colorClass(metrics['Precision'])),
    statBox('F1-Score',       fmt.pct(metrics['F1-Score']), colorClass(metrics['F1-Score'])),
    statBox('AUC-ROC',        auc.toFixed(4), colorClass(auc, { success: 0.75, warning: 0.60 })),
    statBox('MCC',            metrics['MCC'].toFixed(4), colorClass(metrics['MCC'], { success: 0.50, warning: 0.30 })),
  ].join(''));
}

/* =========================================================
   Entry Point
   ========================================================= */
async function init() {
  const overlay = document.getElementById('loadingOverlay');
  try {
    let data;
    if (window.reportData) {
      data = window.reportData;
    } else {
      const response = await fetch('report_data.json');
      if (!response.ok) throw new Error(`No se pudo cargar report_data.json o report_data.js (${response.status})`);
      data = await response.json();
    }

    // ---- Imágenes de gráficos ----
    const ch = data.charts;
    setImg('chartClassDist', ch.class_distribution);
    setImg('chartEntropy',   ch.entropy_diagram);
    setImg('chartCM',        ch.confusion_matrix);
    setImg('chartMetrics',   ch.metrics);
    setImg('chartROC',       ch.roc);
    setImg('chartFI',        ch.feature_importance);

    // ---- Secciones ----
    buildDataset(data.dataset);

    // Adaptaciones dinámicas para datasets genéricos
    const isDefaultCreditCard = !data.dataset_name || data.dataset_name.includes('default_of_credit_card_clients');
    if (!isDefaultCreditCard) {
      const heroTitle = document.getElementById('hero-title');
      if (heroTitle) {
        if (data.dataset_name.includes('churn')) {
          heroTitle.textContent = 'Predicción de Churn (Fuga de Clientes)';
        } else {
          heroTitle.textContent = `Análisis de Clasificación C4.5 - ${data.dataset_name}`;
        }
      }
      const heroDataset = document.getElementById('hero-dataset');
      if (heroDataset) {
        heroDataset.innerHTML = `<span class="icon">📊</span> Dataset: ${data.dataset_name}`;
      }
      const descTitle = document.getElementById('desc-dataset-title');
      if (descTitle) {
        descTitle.textContent = `📁 Descripción del Dataset: ${data.dataset_name}`;
      }
      const descText = document.getElementById('desc-dataset-text');
      if (descText) {
        descText.innerHTML = `El dataset analizado es <strong>${data.dataset_name}</strong>. Contiene un total de <strong id="totalRows">${fmt.num(data.dataset.total_rows)}</strong> registros y <strong>${data.dataset.total_cols}</strong> columnas. La variable objetivo seleccionada es <code>${data.dataset.target_column}</code>.`;
      }
      const varsSubtitle = document.getElementById('variables-subtitle');
      if (varsSubtitle) {
        varsSubtitle.textContent = `${data.dataset.columns.length - 1} variables predictoras`;
      }
      const tableBody = document.querySelector('#variables-table tbody');
      if (tableBody) {
        tableBody.innerHTML = data.dataset.columns.map(col => {
          const isTarget = col === data.dataset.target_column;
          return `<tr>
            <td><code>${col}</code></td>
            <td>${isTarget ? '<strong>Variable objetivo (Target)</strong>' : 'Variable predictora'}</td>
            <td>${isTarget ? 'Objetivo' : 'Numérica / Codificada'}</td>
          </tr>`;
        }).join('');
      }
      const footerDataset = document.getElementById('footer-dataset-name');
      if (footerDataset) {
        footerDataset.textContent = `Dataset: ${data.dataset_name}`;
      }
    }

    buildTreeBefore(data.tree_before_pruning);
    buildPruning(data.tree_before_pruning, data.tree_after_pruning);
    buildMetrics(data.confusion_matrix, data.metrics);
    buildROC(data.roc_auc);
    buildFeatureImportance(data.feature_importance);
    buildFinalKpis(data.metrics, data.roc_auc, data.tree_after_pruning);

    // Hiperparámetros
    const hp = data.hyperparameters;
    setText('hyperparamsTd', `max_depth=${hp.max_depth}, min_samples=${hp.min_samples_split}, min_gain_ratio=${hp.min_gain_ratio}`);

    // ── Nuevas secciones ──
    renderSystemMetrics(data);
    renderSklearnComparison(data);
    renderExplainability(data);
    renderBusiness(data);
    renderCodeExport(data);
    renderSummaryPage(data);

  } catch (err) {
    overlay.innerHTML = `
      <div style="text-align:center;padding:2rem;">
        <div style="font-size:3rem;margin-bottom:1rem;">⚠️</div>
        <h2 style="color:var(--danger);margin-bottom:1rem;">No se pudo cargar el informe</h2>
        <p style="color:var(--text-muted);max-width:500px;">
          Asegúrate de haber ejecutado primero el generador de datos:
        </p>
        <pre style="margin:1.5rem auto;max-width:500px;background:var(--bg-card);padding:1rem;border-radius:8px;font-family:var(--font-mono);font-size:0.85rem;color:var(--accent-light);text-align:left;">
cd PyC45
python web_verification/generate_report.py</pre>
        <p style="color:var(--text-muted);font-size:0.85rem;">Error técnico: ${err.message}</p>
      </div>`;
    return;
  }

  // Ocultar loading overlay
  overlay.classList.add('hidden');
  setTimeout(() => overlay.remove(), 600);

  // Iniciar observadores
  setupScrollObserver();
  setupTOC();
}

/* =========================================================
   SECCIÓN 10 — Sistema & Complejidad
   ========================================================= */
function renderSystemMetrics(data) {
  const sys = data.system_metrics || {};
  const tree = data.tree_after_pruning || {};

  const kpiEl = document.getElementById('systemKpis');
  if (kpiEl) {
    kpiEl.innerHTML = [
      statBox('⏱ Tiempo de Entrenamiento',
        sys.training_time_s != null ? `${sys.training_time_s.toFixed(4)} s` : 'N/A', 'accent'),
      statBox('✂️ Tiempo de Poda REP',
        sys.pruning_time_s != null ? `${sys.pruning_time_s.toFixed(4)} s` : 'N/A', 'warning'),
      statBox('⚡ Tiempo de Inferencia',
        sys.inference_time_ms != null ? `${sys.inference_time_ms.toFixed(4)} ms/muestra` : 'N/A', 'success'),
      statBox('💾 Mem. Estimada (nodos)',
        sys.memory_nodes_kb != null ? `${sys.memory_nodes_kb.toFixed(1)} KB` : 'N/A', 'info'),
    ].join('');
  }

  const structEl = document.getElementById('treeStructureKpis');
  if (structEl) {
    structEl.innerHTML = [
      statBox('🌳 Nodos Totales',   tree.total_nodes    != null ? tree.total_nodes    : 'N/A', 'accent'),
      statBox('🍃 Hojas',           tree.leaf_nodes     != null ? tree.leaf_nodes     : 'N/A', 'success'),
      statBox('📏 Profundidad Máx', tree.max_depth_real != null ? tree.max_depth_real : 'N/A', 'warning'),
      statBox('📋 Nº de Reglas',    tree.leaf_nodes     != null ? tree.leaf_nodes     : 'N/A', 'info'),
    ].join('');
  }
}

/* =========================================================
   SECCIÓN 11 — Comparación scikit-learn
   ========================================================= */
function renderSklearnComparison(data) {
  const cmp = data.sklearn_comparison;
  const tbody = document.getElementById('sklearn-table-body');
  if (!tbody) return;

  if (!cmp) {
    tbody.innerHTML = `<tr><td colspan="4" style="text-align:center;color:var(--text-muted);">
      scikit-learn no disponible en este entorno — instala con <code>pip install scikit-learn</code>
    </td></tr>`;
    return;
  }

  const metrics = [
    { label: '✅ Exactitud (Accuracy)', pyc: cmp.pyc45_accuracy,  skl: cmp.sklearn_accuracy,  fmt: v => `${(v*100).toFixed(2)}%`  },
    { label: '🎯 Precisión',           pyc: cmp.pyc45_precision, skl: cmp.sklearn_precision, fmt: v => `${(v*100).toFixed(2)}%`  },
    { label: '🔍 Recall',              pyc: cmp.pyc45_recall,    skl: cmp.sklearn_recall,    fmt: v => `${(v*100).toFixed(2)}%`  },
    { label: '⚖️ F1-Score',            pyc: cmp.pyc45_f1,        skl: cmp.sklearn_f1,        fmt: v => `${(v*100).toFixed(2)}%`  },
    { label: '📐 AUC-ROC',             pyc: cmp.pyc45_auc,       skl: cmp.sklearn_auc,       fmt: v => v.toFixed(4)              },
    { label: '🌳 Nodos del Árbol',     pyc: cmp.pyc45_nodes,     skl: cmp.sklearn_nodes,     fmt: v => v                         },
    { label: '📏 Profundidad',         pyc: cmp.pyc45_depth,     skl: cmp.sklearn_depth,     fmt: v => v                         },
  ];

  tbody.innerHTML = metrics.map(m => {
    if (m.pyc == null || m.skl == null) return '';
    const diff = (typeof m.pyc === 'number' && typeof m.skl === 'number')
      ? (m.pyc - m.skl) : null;
    let diffHTML = '—';
    if (diff !== null) {
      const sign = diff >= 0 ? '+' : '';
      const cls  = diff >= 0 ? 'success' : 'danger';
      diffHTML = `<span style="color:var(--${cls})">${sign}${typeof m.pyc === 'number' && m.pyc < 2 ? (diff*100).toFixed(2)+'%' : diff}</span>`;
    }
    return `<tr>
      <td><strong>${m.label}</strong></td>
      <td style="color:var(--accent);font-weight:600">${m.fmt(m.pyc)}</td>
      <td style="color:var(--success)">${m.fmt(m.skl)}</td>
      <td>${diffHTML}</td>
    </tr>`;
  }).join('');
}

/* =========================================================
   SECCIÓN 12 — Explicabilidad Interactiva
   ========================================================= */
function renderExplainability(data) {
  const samples = data.explainability_samples;
  const selector = document.getElementById('sample-selector');
  if (!selector || !samples || !samples.length) return;

  samples.forEach((s, i) => {
    const opt = document.createElement('option');
    const correct = s.actual_label === s.predicted_label;
    opt.value = i;
    opt.textContent = `Muestra ${i+1} — Real: ${s.actual_label} | Pred: ${s.predicted_label} | ${correct ? '✅ Correcto' : '❌ Error'}`;
    selector.appendChild(opt);
  });

  selector.addEventListener('change', () => {
    const idx = parseInt(selector.value);
    if (isNaN(idx)) {
      document.getElementById('path-timeline-container').style.display = 'none';
      document.getElementById('path-placeholder').style.display = '';
      return;
    }
    renderDecisionPath(samples[idx]);
  });
}

function renderDecisionPath(sample) {
  const container   = document.getElementById('path-timeline-container');
  const placeholder = document.getElementById('path-placeholder');
  const timeline    = document.getElementById('decision-path-timeline');

  container.style.display   = '';
  placeholder.style.display = 'none';

  const path    = sample.decision_path || [];
  const correct = sample.actual_label === sample.predicted_label;
  const lastStep = path[path.length - 1] || {};

  // Mini KPI cards
  document.getElementById('path-result-card').innerHTML = `
    <div class="stat-value ${correct ? 'success' : 'danger'}">${correct ? '✅ Correcto' : '❌ Error'}</div>
    <div class="stat-label">Predicción: ${sample.predicted_label} | Real: ${sample.actual_label}</div>`;

  document.getElementById('path-confidence-card').innerHTML = `
    <div class="stat-value accent">${lastStep.node_probability != null ? (lastStep.node_probability*100).toFixed(1)+'%' : 'N/A'}</div>
    <div class="stat-label">Confianza de la Hoja</div>`;

  document.getElementById('path-depth-card').innerHTML = `
    <div class="stat-value warning">${path.length}</div>
    <div class="stat-label">Profundidad del Camino</div>`;

  // Construir timeline
  timeline.innerHTML = path.map((step, i) => {
    const isLast     = i === path.length - 1;
    const isLeaf     = step.is_leaf;
    const leafClass  = isLeaf ? (correct ? 'leaf' : 'failed') : '';
    const icon       = isLeaf ? (correct ? '🍃' : '⚠️') : '📊';

    let title  = isLeaf
      ? `Hoja: Clase ${step.prediction}`
      : `${step.feature} ≤ ${typeof step.threshold === 'number' ? step.threshold.toFixed(4) : step.threshold}`;

    let desc = isLeaf
      ? `Distribución: ${JSON.stringify(step.class_counts || {})}`
      : `Valor del cliente: <strong>${step.feature_value != null ? step.feature_value : 'N/A'}</strong> → va hacia la ${step.direction === 'left' ? 'rama IZQUIERDA (≤)' : 'rama DERECHA (>)'}`;

    let meta = step.node_probability != null
      ? `Confianza del nodo: ${(step.node_probability * 100).toFixed(1)}%  |  Muestras: ${step.samples != null ? step.samples : '—'}`
      : '';

    return `
      <div class="timeline-item ${leafClass}">
        <div class="timeline-content">
          <div class="timeline-title">${icon} Paso ${i+1} — ${title}</div>
          <div class="timeline-desc">${desc}</div>
          ${meta ? `<div class="timeline-meta">${meta}</div>` : ''}
        </div>
      </div>`;
  }).join('');
}

/* =========================================================
   SECCIÓN 13 — Simulador de Negocio
   ========================================================= */
function renderBusiness(data) {
  const bi   = data.business_impact || {};
  const kpis = document.getElementById('businessKpis');
  const tbody = document.getElementById('business-table-body');
  const sumEl = document.getElementById('business-summary');

  if (kpis) {
    kpis.innerHTML = [
      statBox('💰 Ahorro Estimado',
        bi.estimated_savings_usd != null ? `$${bi.estimated_savings_usd.toLocaleString('es-CO')}` : 'N/A', 'success'),
      statBox('⚠️ Costo Falsos Neg.',
        bi.fn_exposure_usd != null ? `$${bi.fn_exposure_usd.toLocaleString('es-CO')}` : 'N/A', 'danger'),
      statBox('📊 Cobertura de Riesgo',
        bi.risk_coverage_pct != null ? `${bi.risk_coverage_pct.toFixed(1)}%` : 'N/A', 'accent'),
      statBox('🏦 ROI del Modelo',
        bi.model_roi_pct != null ? `${bi.model_roi_pct.toFixed(1)}%` : 'N/A', 'warning'),
    ].join('');
  }

  if (tbody) {
    const rows = bi.scenario_table || [];
    if (rows.length) {
      tbody.innerHTML = rows.map(r => `
        <tr>
          <td><strong>${r.scenario}</strong></td>
          <td>${r.description}</td>
          <td>${r.value}</td>
          <td><span class="tag ${r.impact_class || 'ok'}">${r.impact_label || '—'}</span></td>
        </tr>`).join('');
    } else {
      tbody.innerHTML = `<tr><td colspan="4" style="text-align:center;color:var(--text-muted);">
        Agrega datos de <code>business_impact</code> en el script de ejecución.</td></tr>`;
    }
  }

  if (sumEl && bi.summary_text) {
    sumEl.innerHTML = `<strong>💡 Resumen financiero</strong> — ${bi.summary_text}`;
  }
}

/* =========================================================
   SECCIÓN 14 — Exportación de Código
   ========================================================= */
function renderCodeExport(data) {
  const sql     = data.code_export && data.code_export.sql;
  const python  = data.code_export && data.code_export.python;
  const fastapi = data.code_export && data.code_export.fastapi;

  const sqlEl = document.getElementById('code-sql');
  const pyEl  = document.getElementById('code-python');
  const faEl  = document.getElementById('code-fastapi');

  if (sqlEl && sql)     sqlEl.textContent = sql;
  if (pyEl  && python)  pyEl.textContent  = python;
  if (faEl  && fastapi) faEl.textContent  = fastapi;
}

/* =========================================================
   SECCIÓN 15 — Página Resumen
   ========================================================= */
function renderSummaryPage(data) {
  const metrics = data.metrics || {};
  const charts = data.charts || {};
  
  // Set images for summary
  setImg('chartCMSummary', charts.confusion_matrix);
  setImg('chartROCSummary', charts.roc);

  // Populate summary table
  const tbody = document.getElementById('summaryMetricsTableBody');
  if (tbody) {
    const summaryMetrics = [
      { name: 'Accuracy', val: metrics['Accuracy'], formula: '(TP + TN) / Total', interp: 'Exactitud general del modelo en la clasificación.' },
      { name: 'Precision', val: metrics['Precision'], formula: 'TP / (TP + FP)', interp: 'Fiabilidad de las alertas de impago generadas.' },
      { name: 'Recall', val: metrics['Recall'], formula: 'TP / (TP + FN)', interp: 'Capacidad para detectar casos reales de impago.' },
      { name: 'F1-Score', val: metrics['F1-Score'], formula: '2 · P · R / (P + R)', interp: 'Balance armónico entre Precisión y Recall.' }
    ];

    tbody.innerHTML = summaryMetrics.map(m => {
      if (m.val == null) return '';
      const cls = colorClass(m.val, { success: 0.75, warning: 0.50 });
      return `<tr>
        <td><strong>${m.name}</strong></td>
        <td><span class="stat-value ${cls}" style="font-size:1rem;">${fmt.pct(m.val)}</span></td>
        <td><code>${m.formula}</code></td>
        <td style="font-size:0.85rem;">${m.interp}</td>
      </tr>`;
    }).join('');
  }
}

/* =========================================================
   Helpers: Tabs & Copy
   ========================================================= */
function switchTab(event, targetId) {
  const container = event.target.closest('.tabs-container');
  if (!container) return;

  container.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  container.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));

  event.target.classList.add('active');
  const pane = container.querySelector('#' + targetId);
  if (pane) pane.classList.add('active');
}

function copyCode(codeId) {
  const el = document.getElementById(codeId);
  if (!el) return;
  navigator.clipboard.writeText(el.textContent).then(() => {
    const btn = el.closest('.tab-pane').querySelector('.copy-btn');
    if (btn) {
      const original = btn.textContent;
      btn.textContent = '✅ Copiado!';
      setTimeout(() => btn.textContent = original, 2000);
    }
  });
}

// Iniciar cuando el DOM esté listo
document.addEventListener('DOMContentLoaded', init);

