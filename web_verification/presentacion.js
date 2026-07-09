/**
 * =========================================================
 * PyC45 Interactive Slides — presentacion.js
 * =========================================================
 * Maneja la navegación, barra de progreso, atajos de teclado
 * y la interfaz interactiva de las diapositivas.
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
