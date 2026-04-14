/**
 * DeepShield AI — Upload Module
 * Handles drag-and-drop, file validation, preview, and API upload.
 */

const Upload = (() => {
    const API_BASE = 'http://localhost:8000';
    let selectedFile = null;

    // ── DOM Elements ─────────────────────────────────────────────────────
    function getElements() {
        return {
            zone: document.getElementById('upload-zone'),
            fileInput: document.getElementById('file-input'),
            previewContainer: document.getElementById('preview-container'),
            previewImage: document.getElementById('preview-image'),
            previewRemove: document.getElementById('preview-remove'),
            previewInfo: document.getElementById('preview-info'),
            analyzeBtn: document.getElementById('analyze-btn'),
            loadingContainer: document.getElementById('loading-container'),
            loadingSteps: document.getElementById('loading-steps'),
        };
    }

    // ── File Validation ──────────────────────────────────────────────────
    function validateFile(file) {
        const maxSize = 10 * 1024 * 1024; // 10MB
        const allowedTypes = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp', 'image/tiff'];

        if (!allowedTypes.includes(file.type)) {
            alert(`Unsupported file type: ${file.type}\nAccepted: JPEG, PNG, WebP, BMP, TIFF`);
            return false;
        }

        if (file.size > maxSize) {
            alert(`File too large (${(file.size / 1024 / 1024).toFixed(1)}MB).\nMaximum: 10MB`);
            return false;
        }

        return true;
    }

    // ── Preview ──────────────────────────────────────────────────────────
    function showPreview(file) {
        const els = getElements();
        selectedFile = file;

        // Show image preview
        const reader = new FileReader();
        reader.onload = (e) => {
            els.previewImage.src = e.target.result;
        };
        reader.readAsDataURL(file);

        // Show file info
        const sizeStr = file.size > 1024 * 1024
            ? `${(file.size / 1024 / 1024).toFixed(1)} MB`
            : `${(file.size / 1024).toFixed(0)} KB`;

        els.previewInfo.innerHTML = `
            <span>${file.name}</span>
            <span>${sizeStr}</span>
            <span>${file.type.split('/')[1].toUpperCase()}</span>
        `;

        // Toggle visibility
        els.zone.style.display = 'none';
        els.previewContainer.style.display = 'block';
    }

    function clearPreview() {
        const els = getElements();
        selectedFile = null;
        els.previewImage.src = '';
        els.previewInfo.innerHTML = '';
        els.previewContainer.style.display = 'none';
        els.zone.style.display = 'block';

        // Hide results
        const resultsSection = document.getElementById('results-section');
        if (resultsSection) resultsSection.style.display = 'none';
    }

    // ── Loading Animation ────────────────────────────────────────────────
    function showLoading() {
        const els = getElements();
        els.loadingContainer.style.display = 'block';
        els.analyzeBtn.disabled = true;
        els.analyzeBtn.textContent = 'Analyzing...';

        // Animate steps sequentially
        const steps = els.loadingSteps.querySelectorAll('.loading-step');
        steps.forEach(s => { s.classList.remove('active', 'done'); });

        let currentStep = 0;
        const stepInterval = setInterval(() => {
            if (currentStep > 0) {
                steps[currentStep - 1].classList.remove('active');
                steps[currentStep - 1].classList.add('done');
                steps[currentStep - 1].querySelector('.step-icon').textContent = '✓';
            }
            if (currentStep < steps.length) {
                steps[currentStep].classList.add('active');
                currentStep++;
            } else {
                clearInterval(stepInterval);
            }
        }, 800);

        return stepInterval;
    }

    function hideLoading(stepInterval) {
        clearInterval(stepInterval);
        const els = getElements();
        els.loadingContainer.style.display = 'none';
        els.analyzeBtn.disabled = false;
        els.analyzeBtn.innerHTML = `
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
            </svg>
            Run Full Analysis
        `;
    }

    // ── API Upload ───────────────────────────────────────────────────────
    async function uploadAndAnalyze() {
        if (!selectedFile) return;

        const stepInterval = showLoading();

        try {
            const formData = new FormData();
            formData.append('file', selectedFile);

            const response = await fetch(`${API_BASE}/api/analyze`, {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const error = await response.json().catch(() => ({}));
                throw new Error(error.detail || `Server error: ${response.status}`);
            }

            const data = await response.json();
            hideLoading(stepInterval);

            // Trigger results rendering
            if (window.Results) {
                window.Results.render(data);
            }

        } catch (error) {
            hideLoading(stepInterval);
            console.error('Analysis failed:', error);
            alert(`Analysis failed: ${error.message}\n\nMake sure the backend is running at ${API_BASE}`);
        }
    }

    // ── Health Check ─────────────────────────────────────────────────────
    async function checkHealth() {
        const statusDot = document.querySelector('.status-dot');
        const statusText = document.querySelector('.status-text');

        try {
            const response = await fetch(`${API_BASE}/api/health`, { signal: AbortSignal.timeout(5000) });
            if (response.ok) {
                statusDot.classList.add('online');
                statusDot.classList.remove('offline');
                statusText.textContent = 'API Online';
                return true;
            }
        } catch (e) {
            // API offline
        }

        statusDot.classList.add('offline');
        statusDot.classList.remove('online');
        statusText.textContent = 'API Offline';
        return false;
    }

    // ── Event Binding ────────────────────────────────────────────────────
    function init() {
        const els = getElements();

        // Click to upload
        els.zone.addEventListener('click', () => els.fileInput.click());
        els.fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file && validateFile(file)) {
                showPreview(file);
            }
        });

        // Drag and drop
        els.zone.addEventListener('dragover', (e) => {
            e.preventDefault();
            els.zone.classList.add('drag-over');
        });

        els.zone.addEventListener('dragleave', (e) => {
            e.preventDefault();
            els.zone.classList.remove('drag-over');
        });

        els.zone.addEventListener('drop', (e) => {
            e.preventDefault();
            els.zone.classList.remove('drag-over');
            const file = e.dataTransfer.files[0];
            if (file && validateFile(file)) {
                showPreview(file);
            }
        });

        // Remove preview
        els.previewRemove.addEventListener('click', clearPreview);

        // Analyze button
        els.analyzeBtn.addEventListener('click', uploadAndAnalyze);

        // Smooth scroll for hero CTA
        const heroCta = document.getElementById('hero-cta');
        if (heroCta) {
            heroCta.addEventListener('click', (e) => {
                e.preventDefault();
                document.getElementById('upload-section').scrollIntoView({ behavior: 'smooth' });
            });
        }

        // Health check
        checkHealth();
        setInterval(checkHealth, 30000);
    }

    return { init, checkHealth };
})();

document.addEventListener('DOMContentLoaded', Upload.init);
