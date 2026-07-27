/**
 * =========================================================
 * MetaCog-C45 Interactive Slides — presentacion.js
 * =========================================================
 * Maneja la navegación, barra de progreso, atajos de teclado
 * y la carga dinámica de datos del experimento (reportData).
 * =========================================================
 */

document.addEventListener('DOMContentLoaded', () => {
  const slides = Array.from(document.querySelectorAll('.slide'));
  const btnPrev = document.getElementById('btn-prev');
  const btnNext = document.getElementById('btn-next');
  const btnIndex = document.getElementById('btn-index');
  const modalIndex = document.getElementById('modal-index');
  const btnCloseModal = document.getElementById('close-modal');
  const indexList = document.getElementById('index-list');
  const progressPercent = document.getElementById('progress-percent');
  const progressBar = document.getElementById('progress-bar');
  const slideNumText = document.getElementById('slide-num');

  let currentIdx = 0;
  const totalSlides = slides.length;

  // Carga dinámica de metadatos del experimento activo desde window.reportData
  function loadDynamicReportData() {
    const data = window.reportData;
    const heroBadge = document.getElementById('hero-dataset-badge');
    const expSummary = document.getElementById('presentation-experiment-summary');
    const xaiStatus = document.getElementById('presentation-xai-status');

    if (!data) {
      if (heroBadge) heroBadge.textContent = "Dataset: Demostración Sintética";
      if (expSummary) {
        expSummary.innerHTML = `<strong>Experimento Activo:</strong> Modo genérico preparado. Para visualizar resultados específicos en vivo, ejecuta <code>python ejecutar_con_dataset.py</code>.`;
      }
      return;
    }

    // 1. Badge de la portada
    if (heroBadge && data.dataset) {
      const target = data.dataset.target_column || "Clase Target";
      const sample = data.dataset.sample_size || data.dataset.total_rows || 0;
      heroBadge.textContent = `Dataset: ${target} (n=${sample})`;
    }

    // 2. Resumen del experimento en vivo (Slide 8)
    if (expSummary && data.metrics) {
      const acc = data.metrics.Accuracy ? (data.metrics.Accuracy * 100).toFixed(2) + '%' : 'N/A';
      const mcc = data.metrics.MCC !== undefined ? data.metrics.MCC.toFixed(4) : 'N/A';
      const auc = data.roc_auc !== undefined ? data.roc_auc.toFixed(4) : 'N/A';
      
      let metacogInfo = '';
      if (data.metacognition) {
        const mfi = data.metacognition.mfi !== undefined ? data.metacognition.mfi.toFixed(4) : 'N/A';
        const traces = data.metacognition.total_traces || 0;
        metacogInfo = ` | MFI: <strong style="color:var(--success)">${mfi}</strong> (trazas: ${traces})`;
      }

      expSummary.innerHTML = `<strong>Resultados del Experimento Activo:</strong> Exactitud: <strong>${acc}</strong> | MCC: <strong>${mcc}</strong> | ROC-AUC: <strong>${auc}</strong>${metacogInfo}`;
    }

    // 3. Estado de auditoría XAI (Slide 10)
    if (xaiStatus) {
      if (data.metacognition && data.metacognition.verdict_counts) {
        const vc = data.metacognition.verdict_counts;
        const accept = vc.ACCEPT_SYSTEM_1 || vc.ACCEPT || 0;
        const override = vc.METACOGNITIVE_OVERRIDE || vc.OVERRIDE || 0;
        const fallback = vc.REJECT_FALLBACK || vc.FALLBACK || 0;

        xaiStatus.innerHTML = `<strong>Auditoría Metacognitiva Activa:</strong> Veredictos del Sistema 2 — Sistema 1 Acepatado: <strong style="color:var(--success)">${accept}</strong> | Intervenciones Metacognitivas: <strong style="color:var(--warning)">${override}</strong> | Fallbacks: <strong style="color:var(--danger)">${fallback}</strong>`;
      } else {
        xaiStatus.innerHTML = `<strong>Auditoría Metacognitiva Activa:</strong> Trazas de inferencia <code>DecisionTrace</code> disponibles en <code>report_data.json</code>.`;
      }
    }
  }

  // Initialize slides state
  function initSlides() {
    // Read current slide from URL hash if available (e.g. #slide-3)
    const hash = window.location.hash;
    if (hash && hash.startsWith('#slide-')) {
      const parsed = parseInt(hash.replace('#slide-', ''), 10) - 1;
      if (!isNaN(parsed) && parsed >= 0 && parsed < totalSlides) {
        currentIdx = parsed;
      }
    }

    // Build the interactive Index list dynamically
    indexList.innerHTML = '';
    slides.forEach((slide, idx) => {
      const titleEl = slide.querySelector('.slide-title');
      const coverTitleEl = slide.querySelector('.slide-cover h1');
      let titleText = 'Diapositiva ' + (idx + 1);

      if (coverTitleEl) {
        titleText = 'Portada';
      } else if (titleEl) {
        titleText = titleEl.textContent;
      }

      const li = document.createElement('li');
      li.innerHTML = `
        <button class="index-item-link ${idx === currentIdx ? 'current' : ''}" data-idx="${idx}">
          <span class="index-item-num">${String(idx + 1).padStart(2, '0')}</span>
          <span>${titleText}</span>
        </button>
      `;
      indexList.appendChild(li);
    });

    // Add click listeners to index buttons
    indexList.querySelectorAll('.index-item-link').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const targetIdx = parseInt(e.currentTarget.getAttribute('data-idx'), 10);
        goToSlide(targetIdx);
        closeIndex();
      });
    });

    goToSlide(currentIdx);
    loadDynamicReportData();
  }

  // Update layout and active classes based on index
  function goToSlide(idx) {
    if (idx < 0 || idx >= totalSlides) return;
    currentIdx = idx;

    // Update URL hash
    window.location.hash = `slide-${currentIdx + 1}`;

    slides.forEach((slide, i) => {
      slide.classList.remove('active', 'prev');
      if (i === currentIdx) {
        slide.classList.add('active');
        // Reset scroll position on active slide body
        const body = slide.querySelector('.slide-body');
        if (body) body.scrollTop = 0;
      } else if (i < currentIdx) {
        slide.classList.add('prev');
      }
    });

    // Update button states
    btnPrev.disabled = currentIdx === 0;
    btnNext.disabled = currentIdx === totalSlides - 1;

    // Update progress indicator
    const percent = Math.round((currentIdx / (totalSlides - 1)) * 100);
    progressBar.style.width = `${percent}%`;
    progressPercent.textContent = `${percent}%`;
    slideNumText.textContent = `${currentIdx + 1} / ${totalSlides}`;

    // Update index modal highlight
    indexList.querySelectorAll('.index-item-link').forEach((btn, i) => {
      if (i === currentIdx) {
        btn.classList.add('current');
      } else {
        btn.classList.remove('current');
      }
    });
  }

  function nextSlide() {
    if (currentIdx < totalSlides - 1) {
      goToSlide(currentIdx + 1);
    }
  }

  function prevSlide() {
    if (currentIdx > 0) {
      goToSlide(currentIdx - 1);
    }
  }

  // Index Modal actions
  function openIndex() {
    modalIndex.classList.add('active');
  }

  function closeIndex() {
    modalIndex.classList.remove('active');
  }

  // Keyboard navigation
  document.addEventListener('keydown', (e) => {
    // If modal is active, Esc closes it
    if (modalIndex.classList.contains('active')) {
      if (e.key === 'Escape') {
        closeIndex();
      }
      return; // Disable normal nav keys while index is open
    }

    switch(e.key) {
      case 'ArrowRight':
      case 'Space':
      case 'PageDown':
        e.preventDefault();
        nextSlide();
        break;
      case 'ArrowLeft':
      case 'PageUp':
      case 'Backspace':
        e.preventDefault();
        prevSlide();
        break;
      case 'Home':
        e.preventDefault();
        goToSlide(0);
        break;
      case 'End':
        e.preventDefault();
        goToSlide(totalSlides - 1);
        break;
      case 'm':
      case 'M':
      case 'i':
      case 'I':
        openIndex();
        break;
    }
  });

  // Navigation click listeners
  btnPrev.addEventListener('click', prevSlide);
  btnNext.addEventListener('click', nextSlide);
  btnIndex.addEventListener('click', openIndex);
  btnCloseModal.addEventListener('click', closeIndex);

  // Close modal when clicking background
  modalIndex.addEventListener('click', (e) => {
    if (e.target === modalIndex) {
      closeIndex();
    }
  });

  // Touch gestures (swipe next / prev)
  let touchStartX = 0;
  let touchEndX = 0;

  document.addEventListener('touchstart', (e) => {
    touchStartX = e.changedTouches[0].screenX;
  }, false);

  document.addEventListener('touchend', (e) => {
    touchEndX = e.changedTouches[0].screenX;
    handleSwipe();
  }, false);

  function handleSwipe() {
    const diff = touchEndX - touchStartX;
    const threshold = 60; // minimum distance in px
    if (Math.abs(diff) > threshold) {
      if (diff < 0) {
        // Swiped left -> next
        nextSlide();
      } else {
        // Swiped right -> prev
        prevSlide();
      }
    }
  }

  // Initialize
  initSlides();
});
