/**
 * DeepShield AI — Results Module
 * Renders analysis results: verdict, scores, heatmaps, forensics checks.
 */

const Results = (() => {

    // ── Render Full Results ──────────────────────────────────────────────
    function render(data) {
        if (!data || data.status !== 'success') {
            console.error('Invalid analysis data:', data);
            return;
        }

        const resultsSection = document.getElementById('results-section');
        resultsSection.style.display = 'block';

        // Scroll to results
        setTimeout(() => {
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 100);

        renderVerdict(data.overall);
        renderGradCAM(data.overall, data.classifier);
        renderELA(data.ela);
        renderDCT(data.dct);
        renderForensics(data.forensics);
    }

    // ── Verdict Card ─────────────────────────────────────────────────────
    function renderVerdict(overall) {
        const score = overall.score || 0;
        const verdict = overall.verdict || 'UNKNOWN';
        const confidence = overall.confidence || 0;
        const reasoning = overall.reasoning || [];

        // Score ring animation
        const progress = document.getElementById('verdict-progress');
        const circumference = 2 * Math.PI * 70; // r=70
        const offset = circumference - (score / 100) * circumference;
        setTimeout(() => {
            progress.style.strokeDashoffset = offset;
        }, 200);

        // Score counter animation
        animateCounter('verdict-score', 0, Math.round(score), 1500);

        // Verdict label
        const label = document.getElementById('verdict-label');
        label.textContent = verdict;
        label.className = 'verdict-label';
        if (verdict === 'FAKE' || verdict === 'HIGH') {
            label.classList.add('fake');
        } else if (verdict === 'SUSPICIOUS') {
            label.classList.add('suspicious');
        } else {
            label.classList.add('real');
        }

        // Update ring gradient color based on verdict
        const gradStops = document.querySelectorAll('#verdictGrad stop');
        if (verdict === 'FAKE') {
            gradStops[0].setAttribute('stop-color', '#ff1744');
            gradStops[1].setAttribute('stop-color', '#ff9100');
        } else if (verdict === 'SUSPICIOUS') {
            gradStops[0].setAttribute('stop-color', '#ffab00');
            gradStops[1].setAttribute('stop-color', '#ff6d00');
        } else {
            gradStops[0].setAttribute('stop-color', '#76ff03');
            gradStops[1].setAttribute('stop-color', '#00e5ff');
        }

        // Confidence
        document.getElementById('verdict-confidence').textContent =
            `${overall.prediction} — ${confidence}% confidence • Analyzed in ${document.querySelector('.loading-title')?.dataset?.time || '~2'}s`;

        // Reasoning
        const reasoningEl = document.getElementById('verdict-reasoning');
        reasoningEl.innerHTML = reasoning.map(r =>
            `<div class="reasoning-item">⚡ ${escapeHtml(r)}</div>`
        ).join('');
    }

    // ── Grad-CAM Panel ───────────────────────────────────────────────────
    function renderGradCAM(overall, classifier) {
        const img = document.getElementById('gradcam-image');
        if (overall.gradcam_b64) {
            img.src = `data:image/png;base64,${overall.gradcam_b64}`;
        }

        const info = document.getElementById('classifier-info');
        const prediction = overall.prediction || 'N/A';
        const confidence = overall.confidence || 0;

        info.innerHTML = `
            ${infoRow('Prediction', prediction, prediction === 'FAKE' ? 'high' : 'low')}
            ${infoRow('Confidence', `${confidence}%`, confidence > 70 ? 'high' : confidence > 40 ? 'moderate' : 'low')}
            ${infoRow('ELA Score', `${overall.score || 0}`, scoreClass(overall.score))}
            ${scoreBarRow('Risk Level', overall.score || 0)}
            ${classifier?.model_loaded === false ? `
                <div class="info-row" style="border-left: 3px solid var(--accent-orange); flex-direction: column; align-items: flex-start; gap: 4px;">
                    <span class="info-label" style="color: var(--accent-orange);">⚠ Demo Mode</span>
                    <span class="info-label" style="font-size: 0.75rem;">Using aggregated forensic scores. Add a trained model for higher accuracy.</span>
                </div>
            ` : ''}
        `;
    }

    // ── ELA Panel ────────────────────────────────────────────────────────
    function renderELA(ela) {
        if (!ela) return;

        const heatmap = document.getElementById('ela-heatmap');
        const overlay = document.getElementById('ela-overlay');

        if (ela.heatmap_b64) heatmap.src = `data:image/png;base64,${ela.heatmap_b64}`;
        if (ela.overlay_b64) overlay.src = `data:image/png;base64,${ela.overlay_b64}`;

        const info = document.getElementById('ela-info');
        const stats = ela.stats || {};

        info.innerHTML = `
            ${infoRow('ELA Score', `${ela.score}/100`, scoreClass(ela.score))}
            ${scoreBarRow('Suspicion Level', ela.score)}
            ${infoRow('Max Error', stats.max_error?.toFixed(1) || 'N/A')}
            ${infoRow('Mean Error', stats.mean_error?.toFixed(1) || 'N/A')}
            ${infoRow('Error Std Dev', stats.error_std?.toFixed(1) || 'N/A')}
            ${infoRow('Region Variance', stats.region_variance?.toFixed(1) || 'N/A')}
            <div class="info-row" style="flex-direction: column; align-items: flex-start; gap: 4px;">
                <span class="info-label">Verdict</span>
                <span class="info-label" style="color: var(--text-secondary); font-size: 0.8rem; line-height: 1.4;">${escapeHtml(ela.verdict || '')}</span>
            </div>
        `;
    }

    // ── DCT Panel ────────────────────────────────────────────────────────
    function renderDCT(dct) {
        if (!dct) return;

        const spectral = document.getElementById('dct-spectral');
        const overlay = document.getElementById('dct-overlay');

        if (dct.spectral_map_b64) spectral.src = `data:image/png;base64,${dct.spectral_map_b64}`;
        if (dct.spectral_overlay_b64) overlay.src = `data:image/png;base64,${dct.spectral_overlay_b64}`;

        const info = document.getElementById('dct-info');
        const freq = dct.frequency_distribution || {};
        const periodic = dct.periodic_artifacts || {};

        info.innerHTML = `
            ${infoRow('DCT Score', `${dct.score}/100`, scoreClass(dct.score))}
            ${scoreBarRow('Anomaly Level', dct.score)}
            ${infoRow('Low Freq Ratio', (freq.low_freq_ratio * 100)?.toFixed(1) + '%' || 'N/A')}
            ${infoRow('Mid Freq Ratio', (freq.mid_freq_ratio * 100)?.toFixed(1) + '%' || 'N/A')}
            ${infoRow('High Freq Ratio', (freq.high_freq_ratio * 100)?.toFixed(1) + '%' || 'N/A')}
            ${infoRow('Periodic Artifacts', periodic.has_periodic_artifacts ? '⚠ Detected' : '✓ None', periodic.has_periodic_artifacts ? 'high' : 'low')}
            ${infoRow('AC Energy Peaks', periodic.peak_count || 0)}
            <div class="info-row" style="flex-direction: column; align-items: flex-start; gap: 4px;">
                <span class="info-label">Verdict</span>
                <span class="info-label" style="color: var(--text-secondary); font-size: 0.8rem; line-height: 1.4;">${escapeHtml(dct.verdict || '')}</span>
            </div>
        `;
    }

    // ── Forensics Panel ──────────────────────────────────────────────────
    function renderForensics(forensics) {
        if (!forensics) return;

        const faceImg = document.getElementById('forensics-face');
        if (forensics.annotated_face_b64) {
            faceImg.src = `data:image/png;base64,${forensics.annotated_face_b64}`;
        } else if (forensics.annotated_full_b64) {
            faceImg.src = `data:image/png;base64,${forensics.annotated_full_b64}`;
        }

        const checksContainer = document.getElementById('forensics-checks');
        const checks = forensics.checks || {};

        const checkOrder = [
            { key: 'symmetry', icon: '⚖️', name: 'Facial Symmetry' },
            { key: 'eye_reflections', icon: '👁️', name: 'Eye Reflections' },
            { key: 'boundaries', icon: '🔲', name: 'Boundary Artifacts' },
            { key: 'mouth', icon: '👄', name: 'Mouth / Teeth' },
        ];

        checksContainer.innerHTML = checkOrder.map(({ key, icon, name }) => {
            const check = checks[key] || {};
            const isSuspicious = check.is_suspicious;
            const score = check.score || 0;

            return `
                <div class="forensic-check ${isSuspicious ? 'suspicious' : 'safe'}">
                    <div class="check-header">
                        <span class="check-name">${icon} ${name}</span>
                        <span class="check-badge ${isSuspicious ? 'fail' : 'pass'}">
                            ${isSuspicious ? 'FLAGGED' : 'PASS'}
                        </span>
                    </div>
                    ${scoreBarRow('', score)}
                    <div class="check-detail">${escapeHtml(check.detail || 'No data available')}</div>
                </div>
            `;
        }).join('');

        // Add overall forensics info
        if (!forensics.face_detected) {
            checksContainer.innerHTML = `
                <div class="forensic-check" style="grid-column: 1 / -1; text-align: center; padding: 2rem;">
                    <span style="font-size: 2rem;">🚫</span>
                    <p style="margin-top: 0.5rem; color: var(--text-muted);">No face detected in the image. Face forensics checks require a visible face.</p>
                </div>
            `;
        }
    }

    // ── Helper Functions ─────────────────────────────────────────────────
    function infoRow(label, value, valueClass = '') {
        return `
            <div class="info-row">
                <span class="info-label">${label}</span>
                <span class="info-value ${valueClass}">${value}</span>
            </div>
        `;
    }

    function scoreBarRow(label, score) {
        const barClass = score >= 65 ? 'danger' : score >= 35 ? 'warn' : 'safe';
        return `
            <div class="score-bar">
                <div class="score-bar-fill ${barClass}" style="width: ${Math.min(100, score)}%;"></div>
            </div>
        `;
    }

    function scoreClass(score) {
        if (score >= 65) return 'high';
        if (score >= 35) return 'moderate';
        return 'low';
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    function animateCounter(elementId, start, end, duration) {
        const el = document.getElementById(elementId);
        if (!el) return;

        const range = end - start;
        const startTime = performance.now();

        function update(currentTime) {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            // Ease out cubic
            const eased = 1 - Math.pow(1 - progress, 3);
            const current = Math.round(start + range * eased);
            el.textContent = current;

            if (progress < 1) {
                requestAnimationFrame(update);
            }
        }

        requestAnimationFrame(update);
    }

    // ── Public API ───────────────────────────────────────────────────────
    return { render };
})();

// Make globally accessible for Upload module
window.Results = Results;
