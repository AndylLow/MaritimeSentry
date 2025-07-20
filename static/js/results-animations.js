// Results page animations and transitions
// Bosphorus Ship Detection System

document.addEventListener('DOMContentLoaded', function() {
    initializeResultsAnimations();
});

function initializeResultsAnimations() {
    // Initialize entrance animations
    initEntranceAnimations();
    
    // Initialize image transitions
    initImageTransitions();
    
    // Initialize scroll animations
    initScrollAnimations();
    
    // Initialize progressive loading
    initProgressiveLoading();
}

// Entrance animations for detection results
function initEntranceAnimations() {
    // Animate stat cards with staggered delays
    const statCards = document.querySelectorAll('.detection-result');
    
    // Use intersection observer for better performance
    const observer = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate-in');
            }
        });
    }, {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    });

    statCards.forEach((card) => {
        observer.observe(card);
    });

    // Animate detection details
    const detectionDetails = document.querySelector('.detection-details');
    if (detectionDetails) {
        setTimeout(() => {
            detectionDetails.classList.add('animate-in');
        }, 600);
    }

    // Animate detection badges with staggered timing
    const detectionBadges = document.querySelectorAll('.detection-badge');
    detectionBadges.forEach((badge, index) => {
        setTimeout(() => {
            badge.classList.add('animate-in');
        }, 800 + (index * 100));
    });
}

// Image transition effects
function initImageTransitions() {
    const resultImage = document.getElementById('resultImage');
    if (!resultImage) return;

    // Fade in image when loaded
    if (resultImage.complete) {
        resultImage.classList.add('visible');
    } else {
        resultImage.addEventListener('load', () => {
            resultImage.classList.add('visible');
        });
    }

    // Add smooth zoom on click
    resultImage.addEventListener('click', function() {
        this.style.transform = this.style.transform === 'scale(1.5)' ? 'scale(1)' : 'scale(1.5)';
    });

    // Reset zoom on double click
    resultImage.addEventListener('dblclick', function() {
        this.style.transform = 'scale(1)';
    });
}

// Scroll-triggered animations
function initScrollAnimations() {
    const animatedElements = document.querySelectorAll('.card, .stat-card');
    
    const scrollObserver = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
            if (entry.isIntersecting) {
                entry.target.style.transform = 'translateY(0)';
                entry.target.style.opacity = '1';
            }
        });
    }, {
        threshold: 0.1
    });

    animatedElements.forEach((element) => {
        // Set initial state
        element.style.transform = 'translateY(30px)';
        element.style.opacity = '0';
        element.style.transition = 'all 0.6s cubic-bezier(0.4, 0, 0.2, 1)';
        
        scrollObserver.observe(element);
    });
}

// Progressive loading animation
function initProgressiveLoading() {
    const loadingStates = document.querySelectorAll('.detection-loading');
    
    loadingStates.forEach((state, index) => {
        setTimeout(() => {
            state.classList.add('visible');
        }, index * 200);
    });
}

// Utility functions for enhanced interactions
function animateStatCard(card) {
    card.style.transform = 'scale(1.05)';
    card.style.boxShadow = '0 10px 40px rgba(0, 0, 0, 0.2)';
    
    setTimeout(() => {
        card.style.transform = 'scale(1)';
        card.style.boxShadow = '';
    }, 200);
}

// Enhanced image controls
function zoomImage() {
    const image = document.getElementById('resultImage');
    if (!image) return;
    
    const container = image.closest('.result-image-container');
    container.classList.add('transitioning');
    
    setTimeout(() => {
        if (image.style.transform === 'scale(2)') {
            image.style.transform = 'scale(1)';
        } else {
            image.style.transform = 'scale(2)';
        }
        
        setTimeout(() => {
            container.classList.remove('transitioning');
        }, 300);
    }, 50);
}

function toggleFullscreen() {
    const image = document.getElementById('resultImage');
    if (!image) return;
    
    const container = image.closest('.result-image-container');
    
    if (!document.fullscreenElement) {
        container.requestFullscreen().then(() => {
            image.style.maxHeight = '100vh';
            image.style.objectFit = 'contain';
        });
    } else {
        document.exitFullscreen().then(() => {
            image.style.maxHeight = '';
            image.style.objectFit = '';
        });
    }
}

// Smooth transitions for navigation
function transitionToResults(newContent) {
    const resultsContainer = document.querySelector('.container');
    
    // Fade out current content
    resultsContainer.style.opacity = '0';
    resultsContainer.style.transform = 'translateY(20px)';
    
    setTimeout(() => {
        // Update content
        resultsContainer.innerHTML = newContent;
        
        // Fade in new content
        resultsContainer.style.opacity = '1';
        resultsContainer.style.transform = 'translateY(0)';
        
        // Reinitialize animations
        initializeResultsAnimations();
    }, 300);
}

// Export functions for external use
window.ResultsAnimations = {
    animateStatCard,
    zoomImage,
    toggleFullscreen,
    transitionToResults
};