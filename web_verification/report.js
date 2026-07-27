/**
 * =========================================================
 * MetaCog-C45 Technical Report — report.js
 * =========================================================
 * Carga los datos exportados por MetaCog-C45 y construye
 * dinámicamente el Dashboard Científico (Sistema 1 + Sistema 2).
 * =========================================================
 */

/* =========================================================
   Utilidades
   ========================================================= */
const fmt = {
  pct:  v => `${(v * 100).toFixed(2)}%`,
  num:  v => typeof v === 'number' ? v.toLocaleString('es-CO') : v,
  dec4: v => typeof v === 'number' ? v.toFixed(4) : v,
  dec2: v => typeof v === 'number' ? v.toFixed(2) : v,
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

function buildDataset(d, fullData) {
  setHTML('datasetKpis', [
    statBox('Registros Totales', fmt.num(d.total_rows), 'accent'),
    statBox('Variables',         d.total_cols, 'info'),
    statBox('Muestra Entrenada', fmt.num(d.sample_size), 'warning'),
    statBox('Partición Train/Val', `${d.train_size} / ${d.val_size}`, 'success'),
  ].join(''));

  const datasetName = fullData.dataset_name || 'dataset.csv';
  setText('desc-dataset-title', `📁 Descripción del Dataset Evaluado: ${datasetName}`);

  const targetCol = d.target_column || 'target';
  setHTML('desc-dataset-text', `El experimento fue ejecutado sobre el dataset <strong>${datasetName}</strong>. El conjunto contiene <strong id="totalRows">${fmt.num(d.total_rows)}</strong> registros y <strong>${d.total_cols}</strong> variables. La columna objetivo seleccionada es <code>${targetCol}</code>.`);

  const varsSubtitle = document.getElementById('variables-subtitle');
  if (varsSubtitle) varsSubtitle.textContent = `${(d.columns || []).length - 1} variables predictoras`;

  const tableBody = document.querySelector('#variables-table tbody');
  if (tableBody && d.columns) {
    tableBody.innerHTML = d.columns.map(col => {
      const isTarget = col === targetCol;
      return `<tr>
        <td><code>${col}</code></td>
        <td>${isTarget ? '<strong style="color:var(--accent-light);">Variable objetivo (Target)</strong>' : 'Variable predictora'}</td>
        <td>${isTarget ? 'Objetivo (Target)' : 'Numérica / Codificada'}</td>
      </tr>`;
    }).join('');
  }

  const footerDataset = document.getElementById('footer-dataset-name');
  if (footerDataset) footerDataset.textContent = `Dataset: ${datasetName}`;

  const metaDatasetInfo = document.getElementById('metaDatasetInfo');
  if (metaDatasetInfo) metaDatasetInfo.textContent = `${datasetName} (${fmt.num(d.total_rows)} filas, target='${targetCol}')`;
}

/* =========================================================
   SECCIÓN 03 — Sistema 2 (Análisis Metacognitivo)
   ========================================================= */
function buildMetacognitiveSection(data) {
  const meta = data.metacognitive_data || {};
  const enabled = meta.enabled && meta.mode === 'metacog';

  // Badges de estado operativo
  const statusContainer = document.getElementById('system2StatusBadges');
  if (statusContainer) {
    if (enabled) {
      statusContainer.innerHTML = `
        <div class="status-indicator"><span class="dot"></span> Modo: MetaCog-C45 (Sistema 1 + Sistema 2)</div>
        <div class="status-indicator"><span class="dot"></span> Reflection Engine: Activo (B=${data.hyperparameters.B || 50} réplicas)</div>
        <div class="status-indicator"><span class="dot"></span> Decision Core: Activo (θ_accept=${data.hyperparameters.theta_accept || 0.5})</div>
        <div class="status-indicator"><span class="dot"></span> MetaMemory: Indexación activa (${meta.total_decisions || 0} trazas)</div>
      `;
    } else {
      statusContainer.innerHTML = `
        <div class="status-indicator" style="border-color:var(--danger-dim);color:var(--danger);"><span class="dot" style="background:var(--danger);box-shadow:none;"></span> Modo: Classic C4.5 (Sistema 2 Inactivo)</div>
      `;
    }
  }

  // KPIs Globales
  const kpisContainer = document.getElementById('system2GlobalKpis');
  if (kpisContainer) {
    if (enabled) {
      kpisContainer.innerHTML = [
        statBox('Average SCS', fmt.dec4(meta.scs_stats?.mean || 0.0), 'accent'),
        statBox('Average NS',  fmt.dec4(meta.avg_ns || 0.0), 'success'),
        statBox('Average TS',  fmt.dec4(meta.avg_ts || 0.0), 'info'),
        statBox('Average CI',  fmt.dec4(meta.ci_stats?.mean || 0.0), 'warning'),
      ].join('');
    } else {
      kpisContainer.innerHTML = [
        statBox('Average SCS', 'N/A (Modo Classic)', 'muted'),
        statBox('Average NS',  'N/A (Modo Classic)', 'muted'),
        statBox('Average TS',  'N/A (Modo Classic)', 'muted'),
        statBox('Average CI',  'N/A (Modo Classic)', 'muted'),
      ].join('');
    }
  }

  // Veredictos del Decision Core
  const verdictContainer = document.getElementById('verdictBreakdown');
  if (verdictContainer) {
    if (enabled && meta.verdict_counts) {
      const total = meta.total_decisions || 1;
      const vMap = meta.verdict_counts;
      const verdicts = ['ACCEPT', 'REVIEW', 'DEFER', 'COLLAPSE'];
      verdictContainer.innerHTML = verdicts.map(v => {
        const count = vMap[v] || 0;
        const pct = ((count / total) * 100).toFixed(1);
        const cls = v.toLowerCase();
        return `
          <div>
            <div style="display:flex; justify-size:space-between; justify-content:space-between; align-items:center; margin-bottom:0.25rem;">
              <span class="verdict-badge ${cls}">${v}</span>
              <span style="font-family:var(--font-mono); font-size:0.85rem;">${count} divisiones (${pct}%)</span>
            </div>
            <div class="scs-bar-bg">
              <div class="scs-bar-fill" style="width:${pct}%; background:${v === 'ACCEPT' ? '#00d4aa' : v === 'REVIEW' ? '#60a5fa' : v === 'DEFER' ? '#fbbf24' : '#f87171'};"></div>
            </div>
          </div>
        `;
      }).join('');
    } else {
      verdictContainer.innerHTML = '<p style="color:var(--text-muted);font-size:0.9rem;">No hay trazas metacognitivas en modo clásico.</p>';
    }
  }

  // Tabla estadísticas SCS
  const scsTable = document.getElementById('scsStatsTable');
  if (scsTable) {
    if (enabled) {
      const stats = [
        { metric: 'Split Confidence Score (SCS)', min: meta.scs_stats?.min, max: meta.scs_stats?.max, mean: meta.scs_stats?.mean, std: meta.scs_stats?.std },
        { metric: 'Node Stability (NS)',         min: meta.ns_stats?.min,  max: meta.ns_stats?.max,  mean: meta.ns_stats?.mean,  std: meta.ns_stats?.std },
        { metric: 'Threshold Survival (TS)',     min: meta.ts_stats?.min,  max: meta.ts_stats?.max,  mean: meta.ts_stats?.mean,  std: meta.ts_stats?.std },
        { metric: 'Competitiveness Index (CI)',  min: meta.ci_stats?.min,  max: meta.ci_stats?.max,  mean: meta.ci_stats?.mean,  std: meta.ci_stats?.std },
      ];
      scsTable.innerHTML = `
        <div class="table-wrap">
          <table>
            <thead><tr><th>Métrica</th><th>Mín</th><th>Máx</th><th>Promedio</th><th>Desv. Est.</th></tr></thead>
            <tbody>
              ${stats.map(s => `
                <tr>
                  <td><strong>${s.metric}</strong></td>
                  <td><code>${fmt.dec4(s.min)}</code></td>
                  <td><code>${fmt.dec4(s.max)}</code></td>
                  <td><strong style="color:#a78bfa;">${fmt.dec4(s.mean)}</strong></td>
                  <td><code>${fmt.dec4(s.std)}</code></td>
                </tr>
              `).join('')}
            </tbody>
          </table>
        </div>
      `;
    } else {
      scsTable.innerHTML = '<p style="color:var(--text-muted);font-size:0.9rem;">El modo clásico no calcula estadísticos de SCS.</p>';
    }
  }

  // Tabla MFI
  const mfiBody = document.querySelector('#mfiTable tbody');
  if (mfiBody) {
    if (enabled && meta.mfi_normalized && Object.keys(meta.mfi_normalized).length > 0) {
      const sorted = Object.entries(meta.mfi_normalized).sort((a, b) => b[1] - a[1]);
      mfiBody.innerHTML = sorted.map(([feat, val], idx) => {
        const pct = (val * 100).toFixed(2);
        const bar = `<div style="width:${pct}%;height:6px;background:linear-gradient(90deg, #8b5cf6, #00d4aa);border-radius:3px;display:inline-block;min-width:4px;"></div>`;
        return `<tr>
          <td><strong>#${idx + 1}</strong></td>
          <td><code>${feat}</code></td>
          <td>${bar} <strong style="color:#a78bfa; margin-left:6px;">${pct}%</strong></td>
          <td><span class="tag good">Meta-Reflectivo</span></td>
        </tr>`;
      }).join('');
    } else {
      mfiBody.innerHTML = '<tr><td colspan="4" style="text-align:center;color:var(--text-muted);">MFI no disponible (modo clásico o sin trazas).</td></tr>';
    }
  }

  // Tabla DecisionTrace
  const traceTbody = document.getElementById('decisionTraceTbody');
  if (traceTbody) {
    if (enabled && meta.decision_traces && meta.decision_traces.length > 0) {
      traceTbody.innerHTML = meta.decision_traces.map(t => {
        const vCls = (t.verdict || 'ACCEPT').toLowerCase();
        return `<tr>
          <td><code>${t.node_id}</code></td>
          <td>${t.depth}</td>
          <td><code>${t.feature_selected}</code></td>
          <td><code>${t.threshold_selected != null ? t.threshold_selected.toFixed(4) : '—'}</code></td>
          <td><code>${t.gain_ratio.toFixed(4)}</code></td>
          <td><strong style="color:#a78bfa;">${t.scs.toFixed(4)}</strong></td>
          <td><code>${t.ns.toFixed(4)}</code></td>
          <td><code>${t.ts.toFixed(4)}</code></td>
          <td><code>${t.competitiveness_index.toFixed(4)}</code></td>
          <td><span class="verdict-badge ${vCls}">${t.verdict}</span></td>
          <td style="font-size:0.8rem; color:var(--text-secondary); max-width:240px;">${t.justification || '—'}</td>
        </tr>`;
      }).join('');
    } else {
      traceTbody.innerHTML = '<tr><td colspan="11" style="text-align:center;color:var(--text-muted);">No existen trazas DecisionTrace registradas.</td></tr>';
    }
  }
}

function buildTreeBefore(before) {
  setHTML('treeBeforeKpis', [
    statBox('Nodos Totales',   before.stats.nodes, 'accent'),
    statBox('Hojas',           before.stats.leaves, 'success'),
    statBox('Profundidad Máx', before.stats.depth, 'warning'),
    statBox('Accuracy Train',  fmt.pct(before.accuracy_train), 'info'),
  ].join(''));

  setText('treeBeforeText', before.text || '(árbol no generado)');

  const rows = [];
  const lines = (before.text || '').split('\n');
  lines.forEach(line => {
    if (line.includes('<=') && line.includes('Gain=')) {
      const parts = line.split('<=');
      const varName = parts[0].trim().replace(/^[│\s├└─]+/, '');
      const rest = parts[1] ? parts[1].split('[') : [];
      const threshold = rest[0] ? rest[0].trim() : '';
      rows.push(`<tr>
        <td><code>${varName}</code></td>
        <td><code>≤ ${threshold}</code></td>
        <td>Nodo de decisión en el árbol. Evaluado por el framework MetaCog-C45.</td>
      </tr>`);
    }
  });

  setHTML('treeInterpretationTable', rows.length > 0 ? rows.join('') : '<tr><td colspan="3">Sin nodos internos detectados.</td></tr>');
}

function buildPruning(before, after) {
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
}

function buildMetrics(cm, metrics) {
  const cmRows = [
    { type: 'TP (Verdadero Positivo)', desc: 'Clase positiva predicha correctamente', val: cm.tp },
    { type: 'TN (Verdadero Negativo)', desc: 'Clase negativa predicha correctamente', val: cm.tn },
    { type: 'FP (Falso Positivo)',     desc: 'Clase negativa clasificada erróneamente como positiva', val: cm.fp },
    { type: 'FN (Falso Negativo)',     desc: 'Clase positiva NO detectada (error crítico)', val: cm.fn },
    { type: 'Total',                   desc: 'Registros de validación evaluados', val: cm.total },
  ];
  setHTML('cmTable', cmRows.map(r =>
    `<tr><td><strong>${r.type}</strong></td><td>${r.desc}</td><td><strong>${r.val}</strong></td></tr>`
  ).join(''));

  const metaDefs = {
    'Accuracy':          { formula: '(TP + TN) / Total',        interp: 'Proporción de predicciones correctas globales.' },
    'Precision':         { formula: 'TP / (TP + FP)',            interp: 'Fiabilidad de las predicciones positivas.' },
    'Recall':            { formula: 'TP / (TP + FN)',            interp: 'Capacidad de detección de casos positivos reales.' },
    'Specificity':       { formula: 'TN / (TN + FP)',            interp: 'Identificación correcta de la clase negativa.' },
    'F1-Score':          { formula: '2 · P · R / (P + R)',       interp: 'Media armónica entre Precisión y Recall.' },
    'Balanced Accuracy': { formula: '(Recall + Specificity) / 2',interp: 'Exactitud ponderada equilibrada por clase.' },
    'Error Rate':        { formula: '1 − Accuracy',              interp: 'Proporción de predicciones incorrectas.' },
    'MCC':               { formula: '(TP·TN − FP·FN) / √(…)',   interp: 'Coeficiente de correlación de Matthews (robusto ante desbalance).' },
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
  if (auc >= 0.90) { cls = 'success'; interpText = `AUC de ${auc.toFixed(4)} → Clasificación excelente. Discriminación probabilística óptima.`; }
  else if (auc >= 0.80) { cls = 'success'; interpText = `AUC de ${auc.toFixed(4)} → Clasificación buena. Poder predictivo elevado.`; }
  else if (auc >= 0.70) { cls = 'warning'; interpText = `AUC de ${auc.toFixed(4)} → Clasificación aceptable. Discriminación moderada.`; }
  else if (auc >= 0.60) { cls = 'warning'; interpText = `AUC de ${auc.toFixed(4)} → Discriminación moderada-débil superior al azar.`; }
  else { interpText = `AUC de ${auc.toFixed(4)} → Discriminación limitada.`; }

  if (el) el.className = `stat-value ${cls}`;
  setHTML('aucInterpretation', interpText);
}

function buildFeatureImportance(fi) {
  setHTML('fiTable', fi.map((item, i) => {
    const pct = (item.importance * 100).toFixed(2);
    const bar = `<div style="width:${pct}%;height:6px;background:var(--accent);border-radius:3px;display:inline-block;min-width:4px;"></div>`;
    return `<tr>
      <td><strong>#${i + 1}</strong></td>
      <td><code>${item.feature}</code></td>
      <td>${bar} <strong>${pct}%</strong></td>
      <td style="font-size:0.85rem;">Variable predictora seleccionada por el framework MetaCog-C45.</td>
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
      if (!response.ok) throw new Error(`No se pudo cargar report_data.json (${response.status})`);
      data = await response.json();
    }

    // ---- Imágenes de gráficos ----
    const ch = data.charts || {};
    setImg('chartClassDist', ch.class_distribution);
    setImg('chartEntropy',   ch.entropy_diagram);
    setImg('chartCM',        ch.confusion_matrix);
    setImg('chartMetrics',   ch.metrics);
    setImg('chartROC',       ch.roc);
    setImg('chartFI',        ch.feature_importance);

    // ---- Secciones ----
    buildDataset(data.dataset, data);
    buildMetacognitiveSection(data);
    buildTreeBefore(data.tree_before_pruning);
    buildPruning(data.tree_before_pruning, data.tree_after_pruning);
    buildMetrics(data.confusion_matrix, data.metrics);
    buildROC(data.roc_auc);
    buildFeatureImportance(data.feature_importance);
    buildFinalKpis(data.metrics, data.roc_auc, data.tree_after_pruning);

    // Hiperparámetros
    const hp = data.hyperparameters || {};
    setText('hyperparamsTd', `mode=${hp.mode || 'metacog'}, max_depth=${hp.max_depth}, min_samples=${hp.min_samples_split}, B=${hp.B || 50}`);

    // ── Nuevas secciones ──
    renderSystemMetrics(data);
    renderSklearnComparison(data);
    renderExplainability(data);
    renderBusiness(data);
    renderCodeExport(data);
    renderSummaryPage(data);

  } catch (err) {
    if (overlay) {
      overlay.innerHTML = `
        <div style="text-align:center;padding:2rem;">
          <div style="font-size:3rem;margin-bottom:1rem;">⚠️</div>
          <h2 style="color:var(--danger);margin-bottom:1rem;">No se pudo cargar el informe</h2>
          <p style="color:var(--text-muted);max-width:500px;">
            Asegúrate de ejecutar el script con un dataset válido:
          </p>
          <pre style="margin:1.5rem auto;max-width:500px;background:var(--bg-card);padding:1rem;border-radius:8px;font-family:var(--font-mono);font-size:0.85rem;color:var(--accent-light);text-align:left;">
python ejecutar_con_dataset.py --dataset datasets/customer_churn.csv</pre>
          <p style="color:var(--text-muted);font-size:0.85rem;">Error técnico: ${err.message}</p>
        </div>`;
    }
    return;
  }

  // Ocultar loading overlay
  if (overlay) {
    overlay.classList.add('hidden');
    setTimeout(() => overlay.remove(), 600);
  }

  // Iniciar observadores
  setupScrollObserver();
  setupTOC();
}

/* =========================================================
   SECCIÓN 11 — Sistema & Complejidad
   ========================================================= */
function renderSystemMetrics(data) {
  const sys = data.system_metrics || {};

  const kpiEl = document.getElementById('systemMetricsKpis');
  if (kpiEl) {
    kpiEl.innerHTML = [
      statBox('⏱ Tiempo de Entrenamiento',
        sys.training_time_s != null ? `${sys.training_time_s.toFixed(4)} s` : 'N/A', 'accent'),
      statBox('✂️ Tiempo de Poda REP',
        sys.pruning_time_s != null ? `${sys.pruning_time_s.toFixed(4)} s` : 'N/A', 'warning'),
      statBox('⚡ Inferencia Promedio',
        sys.inference_time_ms != null ? `${sys.inference_time_ms.toFixed(4)} ms/muestra` : 'N/A', 'success'),
      statBox('💾 Memoria Estimada',
        sys.memory_nodes_kb != null ? `${sys.memory_nodes_kb.toFixed(1)} KB` : 'N/A', 'info'),
    ].join('');
  }
}

/* =========================================================
   Comparación scikit-learn
   ========================================================= */
function renderSklearnComparison(data) {
  const cmp = data.sklearn_comparison;
  const tbody = document.getElementById('sklearn-table-body');
  if (!tbody) return;

  if (!cmp) {
    tbody.innerHTML = `<tr><td colspan="4" style="text-align:center;color:var(--text-muted);">
      scikit-learn no comparado en esta corrida.
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
    const diff = (typeof m.pyc === 'number' && typeof m.skl === 'number') ? (m.pyc - m.skl) : null;
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
   SECCIÓN 12 — Explicabilidad
   ========================================================= */
function renderExplainability(data) {
  const samples = data.explainability_samples;
  const container = document.getElementById('explainabilityTimeline');
  if (!container || !samples || !samples.length) return;

  container.innerHTML = `
    <div style="margin-bottom:1rem;">
      <label style="font-size:0.85rem; color:var(--text-secondary); margin-right:0.5rem;">Seleccionar muestra de validación:</label>
      <select id="sample-selector" style="background:var(--bg-primary); color:var(--text-primary); border:1px solid var(--border-color); padding:0.4rem 0.8rem; border-radius:6px; font-family:inherit;">
      </select>
    </div>
    <div id="path-timeline-container" style="display:none;">
      <div style="display:flex; gap:1rem; margin-bottom:1rem;" id="path-kpis">
        <div class="stat-box" id="path-result-card"></div>
        <div class="stat-box" id="path-confidence-card"></div>
        <div class="stat-box" id="path-depth-card"></div>
      </div>
      <div class="timeline" id="decision-path-timeline"></div>
    </div>
    <div id="path-placeholder" style="color:var(--text-muted); font-size:0.9rem;">
      Selecciona una muestra en la lista desplegable para inspeccionar su camino de decisión explícito.
    </div>
  `;

  const selector = document.getElementById('sample-selector');
  const defaultOpt = document.createElement('option');
  defaultOpt.value = '';
  defaultOpt.textContent = '-- Seleccionar Muestra --';
  selector.appendChild(defaultOpt);

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

  document.getElementById('path-result-card').innerHTML = `
    <div class="stat-value ${correct ? 'success' : 'danger'}">${correct ? '✅ Correcto' : '❌ Error'}</div>
    <div class="stat-label">Predicción: ${sample.predicted_label} | Real: ${sample.actual_label}</div>`;

  document.getElementById('path-confidence-card').innerHTML = `
    <div class="stat-value accent">${lastStep.node_probability != null ? (lastStep.node_probability*100).toFixed(1)+'%' : 'N/A'}</div>
    <div class="stat-label">Confianza de la Hoja</div>`;

  document.getElementById('path-depth-card').innerHTML = `
    <div class="stat-value warning">${path.length}</div>
    <div class="stat-label">Profundidad del Camino</div>`;

  timeline.innerHTML = path.map((step, i) => {
    const isLeaf     = step.is_leaf;
    const leafClass  = isLeaf ? (correct ? 'leaf' : 'failed') : '';
    const icon       = isLeaf ? (correct ? '🍃' : '⚠️') : '📊';

    let title  = isLeaf
      ? `Hoja: Clase ${step.prediction}`
      : `${step.feature} ≤ ${typeof step.threshold === 'number' ? step.threshold.toFixed(4) : step.threshold}`;

    let desc = isLeaf
      ? `Distribución: ${JSON.stringify(step.class_counts || {})}`
      : `Valor observado: <strong>${step.feature_value != null ? step.feature_value : 'N/A'}</strong> → rama ${step.direction === 'left' ? 'IZQUIERDA (≤)' : 'DERECHA (>)'}`;

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
   SECCIÓN 13 — Valor Operativo
   ========================================================= */
function renderBusiness(data) {
  const bi = data.business_impact || {};
  const container = document.getElementById('businessImpactContent');
  if (!container) return;

  container.innerHTML = `
    <div class="stats-grid">
      ${statBox('💰 Beneficio Estimado', bi.net_benefit_usd != null ? `$${fmt.num(bi.net_benefit_usd)}` : 'N/A', 'success')}
      ${statBox('⚠️ Exposición a Error', bi.fn_exposure_usd != null ? `$${fmt.num(bi.fn_exposure_usd)}` : 'N/A', 'danger')}
      ${statBox('📊 Cobertura de Coincidencia', bi.risk_coverage_pct != null ? `${bi.risk_coverage_pct.toFixed(1)}%` : 'N/A', 'accent')}
      ${statBox('🏦 ROI del Modelo', bi.model_roi_pct != null ? `${bi.model_roi_pct.toFixed(1)}%` : 'N/A', 'warning')}
    </div>
    ${bi.summary_text ? `<div class="info-box insight" style="margin-top:1rem;"><strong>💡 Resumen de Valor:</strong> ${bi.summary_text}</div>` : ''}
  `;
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
   SECCIÓN 15 — Resumen
   ========================================================= */
function renderSummaryPage(data) {
  const metrics = data.metrics || {};
  const charts = data.charts || {};

  setImg('chartCMSummary', charts.confusion_matrix);
  setImg('chartROCSummary', charts.roc);
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
