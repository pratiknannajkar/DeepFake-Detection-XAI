/**
 * DeepShield AI — Main Application Controller
 * Manages tab switching, smooth scrolling, and app-level orchestration.
 */

const App = (() => {
    // ── Tab Switching ────────────────────────────────────────────────────
    function initTabs() {
        const tabButtons = document.querySelectorAll('.tab-btn');
        const tabPanels = document.querySelectorAll('.tab-panel');

        tabButtons.forEach(btn => {
            btn.addEventListener('click', () => {
                const targetTab = btn.dataset.tab;

                // Update buttons
                tabButtons.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');

                // Update panels
                tabPanels.forEach(p => {
                    p.classList.remove('active');
                    if (p.id === `panel-${targetTab}`) {
                        p.classList.add('active');
                    }
                });
            });
        });
    }

    // ── Nav Link Highlighting ────────────────────────────────────────────
    function initNavLinks() {
        const navLinks = document.querySelectorAll('.nav-link');

        navLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const target = link.getAttribute('href');
                const section = document.querySelector(target);
                if (section) {
                    section.scrollIntoView({ behavior: 'smooth' });
                }

                navLinks.forEach(l => l.classList.remove('active'));
                link.classList.add('active');
            });
        });

        // Update active link on scroll
        const sections = document.querySelectorAll('.section');
        window.addEventListener('scroll', () => {
            let current = '';
            sections.forEach(section => {
                const sectionTop = section.offsetTop - 150;
                if (window.scrollY >= sectionTop) {
                    current = section.id;
                }
            });

            navLinks.forEach(link => {
                link.classList.remove('active');
                const href = link.getAttribute('href');
                if (href === `#${current}`) {
                    link.classList.add('active');
                }
            });
        });
    }

    // ── Keyboard Shortcuts ───────────────────────────────────────────────
    function initKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Ctrl/Cmd + U = Upload
            if ((e.ctrlKey || e.metaKey) && e.key === 'u') {
                e.preventDefault();
                document.getElementById('file-input')?.click();
            }

            // Escape = Clear preview
            if (e.key === 'Escape') {
                document.getElementById('preview-remove')?.click();
            }
        });
    }

    // ── Smooth Section Scrolling for btn-outline ─────────────────────────
    function initSmoothScroll() {
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function(e) {
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }
            });
        });
    }

    // ── Init ─────────────────────────────────────────────────────────────
    function init() {
        initTabs();
        initNavLinks();
        initKeyboardShortcuts();
        initSmoothScroll();

        console.log(
            '%c🛡️ DeepShield AI %cv1.0.0',
            'color: #00e5ff; font-size: 16px; font-weight: bold;',
            'color: #d500f9; font-size: 12px;'
        );
    }

    return { init };
})();

document.addEventListener('DOMContentLoaded', App.init);
