// Maritime Loading Animations
// Bosphorus Ship Detection System

class MaritimeLoader {
    constructor() {
        this.loadingTypes = ['ship', 'radar', 'yolo', 'bridge'];
        this.currentType = 'ship';
        this.progressInterval = null;
        this.gridScanInterval = null;
    }

    // Show loading overlay with specified type
    show(type = 'ship', title = 'Processing Image', subtitle = 'Analyzing vessels in the Bosphorus...') {
        this.currentType = type;
        this.createOverlay(title, subtitle);
        this.startAnimation();
        document.body.style.overflow = 'hidden';
    }

    // Hide loading overlay
    hide() {
        const overlay = document.querySelector('.maritime-loading-overlay');
        if (overlay) {
            overlay.style.opacity = '0';
            setTimeout(() => {
                overlay.remove();
                document.body.style.overflow = '';
                this.cleanup();
            }, 300);
        }
    }

    // Create the loading overlay
    createOverlay(title, subtitle) {
        const overlay = document.createElement('div');
        overlay.className = 'maritime-loading-overlay';
        overlay.style.opacity = '0';
        
        const container = document.createElement('div');
        container.className = 'loading-container';
        
        container.innerHTML = `
            <h3 class="loading-title">${title}</h3>
            <p class="loading-subtitle">${subtitle}</p>
            <div class="loading-animation-container">
                ${this.getAnimationHTML()}
            </div>
            <div class="loading-text">
                <span class="loading-dots">${this.getLoadingMessage()}</span>
            </div>
            <div class="maritime-progress">
                <div class="progress-wave" style="--progress: 0%"></div>
            </div>
        `;
        
        overlay.appendChild(container);
        document.body.appendChild(overlay);
        
        // Fade in
        requestAnimationFrame(() => {
            overlay.style.opacity = '1';
        });
    }

    // Get animation HTML based on type
    getAnimationHTML() {
        switch (this.currentType) {
            case 'ship':
                return `
                    <div class="ship-loading">
                        <div class="ship-container">
                            <div class="ship"></div>
                            <div class="waves"></div>
                        </div>
                    </div>
                `;
                
            case 'radar':
                return `
                    <div class="radar-loading">
                        <div class="radar-circle"></div>
                        <div class="radar-circle"></div>
                        <div class="radar-circle"></div>
                        <div class="radar-sweep"></div>
                        <div class="radar-dot"></div>
                        <div class="radar-dot"></div>
                        <div class="radar-dot"></div>
                    </div>
                `;
                
            case 'yolo':
                return `
                    <div class="yolo-processing">
                        <div class="detection-grid">
                            ${Array(9).fill().map(() => '<div class="grid-cell"></div>').join('')}
                        </div>
                    </div>
                `;
                
            case 'bridge':
                return `
                    <div class="bridge-loading">
                        <div class="bridge-tower"></div>
                        <div class="bridge-tower"></div>
                        <div class="bridge-cable"></div>
                        <div class="bridge-cable"></div>
                        <div class="bridge"></div>
                    </div>
                `;
                
            default:
                return this.getAnimationHTML('ship');
        }
    }

    // Get loading message based on type
    getLoadingMessage() {
        const messages = {
            ship: 'Sailing through detection algorithms',
            radar: 'Scanning maritime traffic',
            yolo: 'Running YOLO object detection',
            bridge: 'Crossing data streams'
        };
        
        return messages[this.currentType] || messages.ship;
    }

    // Start specific animations
    startAnimation() {
        // Start progress bar
        this.startProgress();
        
        // Start type-specific animations
        if (this.currentType === 'yolo') {
            this.startGridScan();
        }
    }

    // Animated progress bar
    startProgress() {
        const progressBar = document.querySelector('.progress-wave');
        if (!progressBar) return;
        
        let progress = 0;
        const increment = Math.random() * 2 + 1; // Random increment between 1-3
        
        this.progressInterval = setInterval(() => {
            progress += increment;
            
            // Slow down near completion
            if (progress > 85) {
                progress += 0.5;
            }
            
            if (progress > 95) {
                progress = 95; // Don't complete until manually hidden
            }
            
            progressBar.style.setProperty('--progress', `${progress}%`);
        }, 100);
    }

    // YOLO grid scanning animation
    startGridScan() {
        const cells = document.querySelectorAll('.grid-cell');
        if (!cells.length) return;
        
        let currentCell = 0;
        
        this.gridScanInterval = setInterval(() => {
            // Remove active from all cells
            cells.forEach(cell => cell.classList.remove('active'));
            
            // Add active to random cells
            const activeCells = Math.floor(Math.random() * 3) + 1;
            const selectedCells = [];
            
            for (let i = 0; i < activeCells; i++) {
                let randomCell;
                do {
                    randomCell = Math.floor(Math.random() * cells.length);
                } while (selectedCells.includes(randomCell));
                
                selectedCells.push(randomCell);
                cells[randomCell].classList.add('active');
            }
            
            currentCell = (currentCell + 1) % cells.length;
        }, 600);
    }

    // Complete progress and hide
    complete() {
        const progressBar = document.querySelector('.progress-wave');
        if (progressBar) {
            progressBar.style.setProperty('--progress', '100%');
        }
        
        setTimeout(() => {
            this.hide();
        }, 1000);
    }

    // Cleanup intervals
    cleanup() {
        if (this.progressInterval) {
            clearInterval(this.progressInterval);
            this.progressInterval = null;
        }
        
        if (this.gridScanInterval) {
            clearInterval(this.gridScanInterval);
            this.gridScanInterval = null;
        }
    }

    // Static methods for easy usage
    static showShipLoading(title, subtitle) {
        const loader = new MaritimeLoader();
        loader.show('ship', title, subtitle);
        return loader;
    }

    static showRadarLoading(title, subtitle) {
        const loader = new MaritimeLoader();
        loader.show('radar', title, subtitle);
        return loader;
    }

    static showYoloLoading(title, subtitle) {
        const loader = new MaritimeLoader();
        loader.show('yolo', title, subtitle);
        return loader;
    }

    static showBridgeLoading(title, subtitle) {
        const loader = new MaritimeLoader();
        loader.show('bridge', title, subtitle);
        return loader;
    }
}

// Enhanced upload progress with maritime theme
function enhanceUploadProgress() {
    const uploadForm = document.querySelector('#uploadForm');
    if (!uploadForm) return;
    
    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const formData = new FormData(this);
        const file = formData.get('file');
        
        if (!file) {
            alert('Please select a file to upload');
            return;
        }
        
        // Show maritime loading
        const loader = MaritimeLoader.showYoloLoading(
            'Uploading & Processing',
            'Preparing your Bosphorus image for ship detection...'
        );
        
        // Simulate upload with XMLHttpRequest for progress
        const xhr = new XMLHttpRequest();
        
        xhr.upload.addEventListener('progress', function(e) {
            if (e.lengthComputable) {
                const percentComplete = (e.loaded / e.total) * 50; // Upload is 50% of total process
                const progressBar = document.querySelector('.progress-wave');
                if (progressBar) {
                    progressBar.style.setProperty('--progress', `${percentComplete}%`);
                }
            }
        });
        
        xhr.addEventListener('load', function() {
            if (xhr.status === 200) {
                try {
                    const response = JSON.parse(xhr.responseText);
                    if (response.redirect) {
                        // Continue progress to 100% then redirect
                        loader.complete();
                        setTimeout(() => {
                            window.location.href = response.redirect;
                        }, 1500);
                    }
                } catch (e) {
                    loader.hide();
                    console.error('Error parsing response:', e);
                }
            } else {
                loader.hide();
                alert('Upload failed. Please try again.');
            }
        });
        
        xhr.addEventListener('error', function() {
            loader.hide();
            alert('Upload failed. Please check your connection and try again.');
        });
        
        xhr.open('POST', this.action);
        xhr.send(formData);
    });
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    enhanceUploadProgress();
});

// Export for global use
window.MaritimeLoader = MaritimeLoader;