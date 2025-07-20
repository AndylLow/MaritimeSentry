// THESIS WOW EFFECTS - Advanced JavaScript for Academic Presentation
// Exceptional interactive effects to impress thesis advisors

class ThesisWowEffects {
    constructor() {
        this.init();
    }

    init() {
        this.createMatrixRain();
        this.createNeuralNetwork();
        this.initParticleSystem();
        this.init3DCardEffects();
        this.initAdvancedInteractions();
        this.createDataStreams();
        this.initQuantumLoaders();
        this.initHolographicEffects();
    }

    // Create Matrix-style falling characters
    createMatrixRain() {
        const container = document.createElement('div');
        container.className = 'matrix-rain';
        document.body.appendChild(container);

        const characters = '01YOLO船舶检测BOSPHORUS';
        const columns = Math.floor(window.innerWidth / 20);

        for (let i = 0; i < columns; i++) {
            setTimeout(() => {
                this.createMatrixColumn(container, characters, i * 20);
            }, i * 100);
        }
    }

    createMatrixColumn(container, characters, x) {
        const createChar = () => {
            const char = document.createElement('div');
            char.className = 'matrix-char';
            char.textContent = characters[Math.floor(Math.random() * characters.length)];
            char.style.left = x + 'px';
            char.style.animationDuration = (Math.random() * 3 + 2) + 's';
            char.style.animationDelay = Math.random() * 2 + 's';
            container.appendChild(char);

            setTimeout(() => char.remove(), 5000);
        };

        setInterval(createChar, 200);
    }

    // Create neural network visualization
    createNeuralNetwork() {
        const containers = document.querySelectorAll('.processing-glow');
        
        containers.forEach(container => {
            const network = document.createElement('div');
            network.className = 'neural-network';
            container.appendChild(network);

            // Create nodes
            for (let i = 0; i < 8; i++) {
                const node = document.createElement('div');
                node.className = 'neural-node';
                node.style.left = Math.random() * 100 + '%';
                node.style.top = Math.random() * 100 + '%';
                node.style.animationDelay = Math.random() * 2 + 's';
                network.appendChild(node);
            }

            // Create connections
            for (let i = 0; i < 5; i++) {
                const connection = document.createElement('div');
                connection.className = 'neural-connection';
                connection.style.left = Math.random() * 100 + '%';
                connection.style.top = Math.random() * 100 + '%';
                connection.style.width = Math.random() * 50 + 20 + '%';
                connection.style.transform = `rotate(${Math.random() * 360}deg)`;
                connection.style.animationDelay = Math.random() * 3 + 's';
                network.appendChild(connection);
            }
        });
    }

    // Initialize particle explosion system
    initParticleSystem() {
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('interactive-hover') || 
                e.target.closest('.interactive-hover')) {
                this.createParticleExplosion(e.clientX, e.clientY);
            }
        });
    }

    createParticleExplosion(x, y) {
        const explosion = document.createElement('div');
        explosion.className = 'particle-explosion';
        explosion.style.left = x + 'px';
        explosion.style.top = y + 'px';
        explosion.style.zIndex = '9999';
        document.body.appendChild(explosion);

        for (let i = 0; i < 12; i++) {
            const particle = document.createElement('div');
            particle.className = 'explosion-particle';
            
            const angle = (i * 30) * Math.PI / 180;
            const distance = 50 + Math.random() * 50;
            const dx = Math.cos(angle) * distance;
            const dy = Math.sin(angle) * distance;
            
            particle.style.setProperty('--dx', dx + 'px');
            particle.style.setProperty('--dy', dy + 'px');
            
            explosion.appendChild(particle);
        }

        setTimeout(() => explosion.remove(), 1000);
    }

    // Initialize 3D card effects
    init3DCardEffects() {
        const cards = document.querySelectorAll('.card');
        
        cards.forEach(card => {
            const container = document.createElement('div');
            container.className = 'card-3d';
            card.parentNode.insertBefore(container, card);
            container.appendChild(card);

            container.addEventListener('mousemove', (e) => {
                const rect = container.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;
                
                const centerX = rect.width / 2;
                const centerY = rect.height / 2;
                
                const rotateX = (y - centerY) / centerY * 10;
                const rotateY = (centerX - x) / centerX * 10;
                
                card.style.transform = `
                    rotateX(${rotateX}deg) 
                    rotateY(${rotateY}deg) 
                    translateZ(20px)
                `;
            });

            container.addEventListener('mouseleave', () => {
                card.style.transform = '';
            });
        });
    }

    // Initialize advanced interactions
    initAdvancedInteractions() {
        // Add holographic effect to upload zones
        document.querySelectorAll('.upload-zone').forEach(zone => {
            zone.classList.add('holographic');
        });

        // Add geometric patterns to hero sections
        document.querySelectorAll('.hero-section').forEach(hero => {
            hero.classList.add('geometric-pattern');
        });

        // Add advanced hover effects to buttons
        document.querySelectorAll('.btn').forEach(btn => {
            btn.classList.add('advanced-hover');
        });

        // Add glass effect to cards
        document.querySelectorAll('.stat-card').forEach(card => {
            card.classList.add('glass-ultra');
        });
    }

    // Create data stream visualizations
    createDataStreams() {
        const containers = document.querySelectorAll('.data-stream');
        
        containers.forEach(container => {
            for (let i = 0; i < 20; i++) {
                const point = document.createElement('div');
                point.className = 'data-point';
                point.style.setProperty('--delay', (i * 0.1) + 's');
                container.appendChild(point);
            }
        });
    }

    // Initialize quantum loaders
    initQuantumLoaders() {
        document.querySelectorAll('.quantum-loader').forEach(loader => {
            for (let i = 0; i < 4; i++) {
                const particle = document.createElement('div');
                particle.className = 'quantum-particle';
                loader.appendChild(particle);
            }
        });
    }

    // Initialize holographic scanning effects
    initHolographicEffects() {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('holographic');
                }
            });
        });

        document.querySelectorAll('.result-image').forEach(img => {
            observer.observe(img);
        });
    }

    // Advanced typing effect with sound simulation
    static createAdvancedTypewriter(element, text, speed = 100) {
        element.textContent = '';
        let i = 0;
        
        const typeWriter = () => {
            if (i < text.length) {
                element.textContent += text.charAt(i);
                
                // Add cursor blink effect
                if (i === text.length - 1) {
                    element.classList.add('typing-complete');
                }
                
                i++;
                setTimeout(typeWriter, speed + Math.random() * 50);
            }
        };
        
        typeWriter();
    }

    // Morphing shape animation
    static createMorphingShape(container) {
        const shape = document.createElement('div');
        shape.className = 'morphing-shape';
        container.appendChild(shape);
        return shape;
    }

    // Advanced notification system
    static showAdvancedNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `advanced-notification ${type}`;
        
        notification.innerHTML = `
            <div class="notification-content glass-ultra">
                <div class="quantum-loader">
                    <div class="quantum-particle"></div>
                    <div class="quantum-particle"></div>
                    <div class="quantum-particle"></div>
                    <div class="quantum-particle"></div>
                </div>
                <div class="notification-text neon-text">${message}</div>
            </div>
        `;

        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 10000;
            padding: 1rem;
            border-radius: 16px;
            max-width: 400px;
            transform: translateX(100%);
            transition: all 0.6s cubic-bezier(0.68, -0.55, 0.265, 1.55);
        `;

        document.body.appendChild(notification);

        requestAnimationFrame(() => {
            notification.style.transform = 'translateX(0)';
        });

        setTimeout(() => {
            notification.style.transform = 'translateX(100%)';
            setTimeout(() => notification.remove(), 600);
        }, 5000);
    }

    // Dynamic background color shifting
    initDynamicBackground() {
        let hue = 200;
        
        setInterval(() => {
            hue = (hue + 1) % 360;
            document.documentElement.style.setProperty('--dynamic-bg', 
                `hsl(${hue}, 30%, 95%)`);
        }, 100);
    }

    // Performance monitor
    initPerformanceMonitor() {
        const monitor = document.createElement('div');
        monitor.className = 'performance-monitor';
        monitor.style.cssText = `
            position: fixed;
            bottom: 20px;
            left: 20px;
            background: rgba(0, 0, 0, 0.8);
            color: #0d7377;
            padding: 10px;
            border-radius: 8px;
            font-family: monospace;
            font-size: 12px;
            z-index: 9999;
            display: none;
        `;
        
        document.body.appendChild(monitor);

        let frameCount = 0;
        let lastTime = performance.now();

        const updateFPS = () => {
            frameCount++;
            const currentTime = performance.now();
            
            if (currentTime - lastTime >= 1000) {
                const fps = Math.round((frameCount * 1000) / (currentTime - lastTime));
                monitor.textContent = `FPS: ${fps} | Memory: ${(performance.memory?.usedJSHeapSize / 1048576).toFixed(1)}MB`;
                frameCount = 0;
                lastTime = currentTime;
            }
            
            requestAnimationFrame(updateFPS);
        };

        // Show monitor on Ctrl+Shift+P
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.shiftKey && e.key === 'P') {
                monitor.style.display = monitor.style.display === 'none' ? 'block' : 'none';
            }
        });

        updateFPS();
    }
}

// Spectacular page transition effects
class PageTransitions {
    static fadeTransition() {
        document.body.style.opacity = '0';
        document.body.style.transition = 'opacity 0.5s ease';
        
        window.addEventListener('load', () => {
            document.body.style.opacity = '1';
        });
    }

    static slideTransition(direction = 'right') {
        const directions = {
            right: 'translateX(100%)',
            left: 'translateX(-100%)',
            up: 'translateY(-100%)',
            down: 'translateY(100%)'
        };

        document.body.style.transform = directions[direction];
        document.body.style.transition = 'transform 0.6s cubic-bezier(0.68, -0.55, 0.265, 1.55)';
        
        window.addEventListener('load', () => {
            document.body.style.transform = 'translate(0, 0)';
        });
    }
}

// Initialize all wow effects
document.addEventListener('DOMContentLoaded', () => {
    new ThesisWowEffects();
    PageTransitions.fadeTransition();
    
    // Add futuristic grid to body
    document.body.classList.add('futuristic-grid', 'gpu-accelerated');
    
    // Enable advanced console logging
    console.log('%cBosphorus Ship Detection System - Thesis Mode Activated', 
        'color: #0d7377; font-size: 16px; font-weight: bold; text-shadow: 0 0 10px #0d7377;');
});

// Export for global use
window.ThesisWowEffects = ThesisWowEffects;
window.PageTransitions = PageTransitions;