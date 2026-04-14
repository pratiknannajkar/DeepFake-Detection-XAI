/**
 * DeepShield AI — Animations Module
 * Background particle canvas, scroll animations, and micro-interactions.
 */

const Animations = (() => {
    let canvas, ctx;
    let particles = [];
    let orbs = [];
    let animationId;
    let mouseX = 0, mouseY = 0;

    // ── Particle System ──────────────────────────────────────────────────
    class Particle {
        constructor(w, h) {
            this.reset(w, h);
        }

        reset(w, h) {
            this.x = Math.random() * w;
            this.y = Math.random() * h;
            this.size = Math.random() * 1.5 + 0.5;
            this.speedX = (Math.random() - 0.5) * 0.3;
            this.speedY = (Math.random() - 0.5) * 0.3;
            this.opacity = Math.random() * 0.4 + 0.1;
            this.color = Math.random() > 0.5 ? '0,229,255' : '213,0,249';
        }

        update(w, h) {
            this.x += this.speedX;
            this.y += this.speedY;

            // Mouse repulsion
            const dx = this.x - mouseX;
            const dy = this.y - mouseY;
            const dist = Math.sqrt(dx * dx + dy * dy);
            if (dist < 120) {
                const force = (120 - dist) / 120 * 0.02;
                this.x += dx * force;
                this.y += dy * force;
            }

            // Wrap around
            if (this.x < 0) this.x = w;
            if (this.x > w) this.x = 0;
            if (this.y < 0) this.y = h;
            if (this.y > h) this.y = 0;
        }

        draw(ctx) {
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(${this.color},${this.opacity})`;
            ctx.fill();
        }
    }

    // ── Floating Orb ─────────────────────────────────────────────────────
    class Orb {
        constructor(w, h) {
            this.x = Math.random() * w;
            this.y = Math.random() * h;
            this.radius = Math.random() * 150 + 80;
            this.speedX = (Math.random() - 0.5) * 0.4;
            this.speedY = (Math.random() - 0.5) * 0.4;
            this.hue = Math.random() > 0.5 ? 187 : 290; // Cyan or Magenta
            this.opacity = Math.random() * 0.04 + 0.02;
        }

        update(w, h) {
            this.x += this.speedX;
            this.y += this.speedY;

            if (this.x < -this.radius) this.x = w + this.radius;
            if (this.x > w + this.radius) this.x = -this.radius;
            if (this.y < -this.radius) this.y = h + this.radius;
            if (this.y > h + this.radius) this.y = -this.radius;
        }

        draw(ctx) {
            const gradient = ctx.createRadialGradient(
                this.x, this.y, 0,
                this.x, this.y, this.radius
            );
            gradient.addColorStop(0, `hsla(${this.hue}, 100%, 60%, ${this.opacity})`);
            gradient.addColorStop(1, `hsla(${this.hue}, 100%, 60%, 0)`);

            ctx.beginPath();
            ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
            ctx.fillStyle = gradient;
            ctx.fill();
        }
    }

    // ── Canvas Setup ─────────────────────────────────────────────────────
    function initCanvas() {
        canvas = document.getElementById('bg-canvas');
        if (!canvas) return;
        ctx = canvas.getContext('2d');
        resizeCanvas();

        const w = canvas.width;
        const h = canvas.height;

        // Create particles (reduce count on mobile)
        const count = window.innerWidth < 768 ? 40 : 80;
        particles = [];
        for (let i = 0; i < count; i++) {
            particles.push(new Particle(w, h));
        }

        // Create orbs
        orbs = [];
        for (let i = 0; i < 4; i++) {
            orbs.push(new Orb(w, h));
        }

        animate();
    }

    function resizeCanvas() {
        if (!canvas) return;
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    }

    function animate() {
        if (!ctx || !canvas) return;
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const w = canvas.width;
        const h = canvas.height;

        // Draw orbs
        orbs.forEach(orb => {
            orb.update(w, h);
            orb.draw(ctx);
        });

        // Draw particles
        particles.forEach(p => {
            p.update(w, h);
            p.draw(ctx);
        });

        // Draw connections between nearby particles
        for (let i = 0; i < particles.length; i++) {
            for (let j = i + 1; j < particles.length; j++) {
                const dx = particles[i].x - particles[j].x;
                const dy = particles[i].y - particles[j].y;
                const dist = Math.sqrt(dx * dx + dy * dy);

                if (dist < 100) {
                    const opacity = (1 - dist / 100) * 0.08;
                    ctx.beginPath();
                    ctx.moveTo(particles[i].x, particles[i].y);
                    ctx.lineTo(particles[j].x, particles[j].y);
                    ctx.strokeStyle = `rgba(0,229,255,${opacity})`;
                    ctx.lineWidth = 0.5;
                    ctx.stroke();
                }
            }
        }

        animationId = requestAnimationFrame(animate);
    }

    // ── Scroll Animations ────────────────────────────────────────────────
    function initScrollAnimations() {
        const observer = new IntersectionObserver(
            (entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        entry.target.classList.add('visible');
                    }
                });
            },
            { threshold: 0.1, rootMargin: '0px 0px -50px 0px' }
        );

        document.querySelectorAll('.method-card, .section-header').forEach(el => {
            el.classList.add('animate-on-scroll');
            observer.observe(el);
        });

        // Navbar scroll effect
        window.addEventListener('scroll', () => {
            const navbar = document.getElementById('navbar');
            if (navbar) {
                navbar.classList.toggle('scrolled', window.scrollY > 50);
            }
        });
    }

    // ── Mouse Tracking ───────────────────────────────────────────────────
    function initMouseTracking() {
        document.addEventListener('mousemove', (e) => {
            mouseX = e.clientX;
            mouseY = e.clientY;
        });
    }

    // ── Public API ───────────────────────────────────────────────────────
    function init() {
        initCanvas();
        initScrollAnimations();
        initMouseTracking();

        window.addEventListener('resize', () => {
            resizeCanvas();
        });
    }

    return { init };
})();

// Auto-init when DOM is ready
document.addEventListener('DOMContentLoaded', Animations.init);
