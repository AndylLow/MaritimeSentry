// Enhanced UI JavaScript for Bosphorus Ship Detection System
// Advanced animations and interactions for thesis presentation

class EnhancedUI {
    constructor() {
        this.init();
    }

    init() {
        this.createBackgroundParticles();
        this.initScrollAnimations();
        this.initInteractiveElements();
        this.initCountUpAnimations();
        this.initParallaxEffects();
        this.initTypewriterEffect();
        this.initMicroInteractions();
    }

    // Create floating background particles
    createBackgroundParticles() {
        const container = document.createElement('div');
        container.className = 'background-particles';
        document.body.appendChild(container);

        for (let i = 0; i < 50; i++) {
            const particle = document.createElement('div');
            particle.className = 'particle';
            particle.style.left = Math.random() * 100 + '%';
            particle.style.animationDelay = Math.random() * 20 + 's';
            particle.style.animationDuration = (15 + Math.random() * 10) + 's';
            container.appendChild(particle);
        }
    }

    // Initialize scroll-triggered animations
    initScrollAnimations() {
        const observerOptions = {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        };

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('visible');
                }
            });
        }, observerOptions);

        // Add animation classes to elements
        document.querySelectorAll('.card, .stat-card, .result-image').forEach((el, index) => {
            el.classList.add('fade-in');
            el.style.transitionDelay = (index * 0.1) + 's';
            observer.observe(el);
        });

        // Add slide animations
        document.querySelectorAll('.upload-zone').forEach(el => {
            el.classList.add('slide-in-left');
            observer.observe(el);
        });

        document.querySelectorAll('.info-card').forEach(el => {
            el.classList.add('slide-in-right');
            observer.observe(el);
        });
    }

    // Initialize interactive hover effects
    initInteractiveElements() {
        // Add interactive hover class to buttons and cards
        document.querySelectorAll('.btn, .card, .stat-card').forEach(el => {
            el.classList.add('interactive-hover');
        });

        // Enhanced card interactions
        document.querySelectorAll('.card').forEach(card => {
            card.addEventListener('mouseenter', (e) => {
                this.createRippleEffect(e);
            });
        });

        // Stats cards micro-interactions
        document.querySelectorAll('.stat-card').forEach(card => {
            card.addEventListener('click', () => {
                card.classList.add('micro-bounce');
                setTimeout(() => card.classList.remove('micro-bounce'), 600);
            });
        });
    }

    // Create ripple effect on card hover
    createRippleEffect(e) {
        const card = e.currentTarget;
        const rect = card.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        const ripple = document.createElement('div');
        ripple.style.cssText = `
            position: absolute;
            width: 100px;
            height: 100px;
            background: radial-gradient(circle, rgba(13, 115, 119, 0.1) 0%, transparent 70%);
            border-radius: 50%;
            pointer-events: none;
            transform: translate(-50%, -50%) scale(0);
            left: ${x}px;
            top: ${y}px;
            animation: ripple 0.6s ease-out forwards;
            z-index: 1;
        `;

        card.style.position = 'relative';
        card.appendChild(ripple);

        setTimeout(() => ripple.remove(), 600);
    }

    // Initialize count-up animations for statistics
    initCountUpAnimations() {
        const animateCountUp = (element, target) => {
            const duration = 2000;
            const start = 0;
            const increment = target / (duration / 16);
            let current = start;

            const timer = setInterval(() => {
                current += increment;
                if (current >= target) {
                    current = target;
                    clearInterval(timer);
                }
                element.textContent = Math.floor(current).toLocaleString();
            }, 16);
        };

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const element = entry.target;
                    const target = parseInt(element.getAttribute('data-count') || element.textContent.replace(/,/g, ''));
                    animateCountUp(element, target);
                    observer.unobserve(element);
                }
            });
        });

        document.querySelectorAll('.stat-number').forEach(el => {
            el.setAttribute('data-count', el.textContent.replace(/,/g, ''));
            el.textContent = '0';
            observer.observe(el);
        });
    }

    // Initialize parallax scrolling effects
    initParallaxEffects() {
        let ticking = false;

        const updateParallax = () => {
            const scrolled = window.pageYOffset;
            const parallaxElements = document.querySelectorAll('.parallax-element');

            parallaxElements.forEach((el, index) => {
                const speed = 0.5 + (index * 0.1);
                const yPos = -(scrolled * speed);
                el.style.transform = `translateY(${yPos}px)`;
            });

            ticking = false;
        };

        const requestParallaxUpdate = () => {
            if (!ticking) {
                requestAnimationFrame(updateParallax);
                ticking = true;
            }
        };

        window.addEventListener('scroll', requestParallaxUpdate);
    }

    // Initialize typewriter effect for titles
    initTypewriterEffect() {
        const typewriterElements = document.querySelectorAll('[data-typewriter]');

        typewriterElements.forEach(element => {
            const text = element.textContent;
            const speed = parseInt(element.getAttribute('data-speed')) || 100;
            element.textContent = '';

            let i = 0;
            const typeWriter = () => {
                if (i < text.length) {
                    element.textContent += text.charAt(i);
                    i++;
                    setTimeout(typeWriter, speed);
                } else {
                    element.classList.add('typing-complete');
                }
            };

            // Start typing when element becomes visible
            const observer = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        setTimeout(typeWriter, 500);
                        observer.unobserve(entry.target);
                    }
                });
            });

            observer.observe(element);
        });
    }

    // Initialize micro-interactions
    initMicroInteractions() {
        // Button press effects
        document.querySelectorAll('.btn').forEach(btn => {
            btn.addEventListener('mousedown', () => {
                btn.style.transform = 'scale(0.98)';
            });

            btn.addEventListener('mouseup', () => {
                btn.style.transform = '';
            });

            btn.addEventListener('mouseleave', () => {
                btn.style.transform = '';
            });
        });

        // Form input focus effects
        document.querySelectorAll('input, textarea, select').forEach(input => {
            input.addEventListener('focus', () => {
                input.parentElement.classList.add('pulse-glow');
            });

            input.addEventListener('blur', () => {
                input.parentElement.classList.remove('pulse-glow');
            });
        });

        // Image hover effects
        document.querySelectorAll('.result-image img').forEach(img => {
            img.addEventListener('mouseenter', () => {
                img.style.filter = 'brightness(1.1) contrast(1.1)';
            });

            img.addEventListener('mouseleave', () => {
                img.style.filter = '';
            });
        });
    }

    // Add glow effect to processing elements
    static addProcessingGlow(element) {
        element.classList.add('processing-glow');
    }

    // Remove glow effect
    static removeProcessingGlow(element) {
        element.classList.remove('processing-glow');
    }

    // Animate element entrance
    static animateEntrance(element, animationType = 'fadeIn') {
        element.style.opacity = '0';
        element.style.transform = 'translateY(20px)';
        
        requestAnimationFrame(() => {
            element.style.transition = 'all 0.6s cubic-bezier(0.4, 0, 0.2, 1)';
            element.style.opacity = '1';
            element.style.transform = 'translateY(0)';
        });
    }

    // Show notification with animation
    static showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                <i class="fas fa-info-circle"></i>
                <span>${message}</span>
            </div>
        `;

        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(20px);
            border-radius: 12px;
            padding: 1rem 1.5rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
            border-left: 4px solid var(--ocean);
            transform: translateX(100%);
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            z-index: 10000;
            max-width: 400px;
            color: #2c3e50;
        `;

        document.body.appendChild(notification);

        requestAnimationFrame(() => {
            notification.style.transform = 'translateX(0)';
        });

        setTimeout(() => {
            notification.style.transform = 'translateX(100%)';
            setTimeout(() => notification.remove(), 400);
        }, 4000);
    }
}

// Enhanced Chart Animations
class EnhancedCharts {
    static animateChart(chartElement) {
        const ctx = chartElement.getContext('2d');
        const chart = Chart.getChart(chartElement);
        
        if (chart) {
            chart.data.datasets.forEach((dataset, i) => {
                const meta = chart.getDatasetMeta(i);
                meta.data.forEach((element, index) => {
                    element.hidden = true;
                    setTimeout(() => {
                        element.hidden = false;
                        chart.update('none');
                    }, index * 100);
                });
            });
            chart.update();
        }
    }

    static addChartHoverEffects(chart) {
        const originalDraw = chart.draw;
        chart.draw = function() {
            originalDraw.apply(this, arguments);
            
            // Add gradient overlay
            const ctx = this.ctx;
            const gradient = ctx.createLinearGradient(0, 0, 0, this.height);
            gradient.addColorStop(0, 'rgba(13, 115, 119, 0.1)');
            gradient.addColorStop(1, 'rgba(13, 115, 119, 0)');
            
            ctx.fillStyle = gradient;
            ctx.fillRect(0, 0, this.width, this.height);
        };
    }
}

// Initialize enhanced UI when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new EnhancedUI();
    
    // Add CSS animation keyframes
    const style = document.createElement('style');
    style.textContent = `
        @keyframes ripple {
            to {
                transform: translate(-50%, -50%) scale(2);
                opacity: 0;
            }
        }
        
        .typing-complete::after {
            content: '|';
            animation: blink 1s infinite;
        }
        
        @keyframes blink {
            0%, 50% { opacity: 1; }
            51%, 100% { opacity: 0; }
        }
        
        .notification-content {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
    `;
    document.head.appendChild(style);
});

// Export for global use
window.EnhancedUI = EnhancedUI;
window.EnhancedCharts = EnhancedCharts;