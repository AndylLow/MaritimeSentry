// Upload functionality for Bosphorus Ship Detection System

let currentFile = null;
let uploadZone = null;
let fileInput = null;
let filePreview = null;

// Initialize upload functionality
function initUpload() {
    uploadZone = document.getElementById('uploadZone');
    fileInput = document.getElementById('fileInput');
    filePreview = document.getElementById('filePreview');
    
    if (uploadZone && fileInput) {
        setupDragAndDrop();
        setupFileInput();
        setupFormValidation();
    }
}

// Initialize quick upload (for homepage)
function initQuickUpload() {
    const quickUploadZone = document.getElementById('quickUploadZone');
    const quickFileInput = document.getElementById('quickFileInput');
    
    if (quickUploadZone && quickFileInput) {
        setupQuickDragAndDrop(quickUploadZone, quickFileInput);
    }
}

// Setup drag and drop functionality
function setupDragAndDrop() {
    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadZone.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });
    
    // Highlight drop area when item is dragged over it
    ['dragenter', 'dragover'].forEach(eventName => {
        uploadZone.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        uploadZone.addEventListener(eventName, unhighlight, false);
    });
    
    // Handle dropped files
    uploadZone.addEventListener('drop', handleDrop, false);
    
    // Handle click to open file dialog
    uploadZone.addEventListener('click', () => fileInput.click());
}

// Setup quick drag and drop (for homepage)
function setupQuickDragAndDrop(zone, input) {
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        zone.addEventListener(eventName, preventDefaults, false);
    });
    
    ['dragenter', 'dragover'].forEach(eventName => {
        zone.addEventListener(eventName, () => zone.classList.add('dragover'), false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        zone.addEventListener(eventName, () => zone.classList.remove('dragover'), false);
    });
    
    zone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            // Redirect to upload page with file
            window.location.href = '/upload';
        }
    });
    
    zone.addEventListener('click', () => input.click());
    
    input.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            window.location.href = '/upload';
        }
    });
}

// Setup file input change handler
function setupFileInput() {
    fileInput.addEventListener('change', handleFileSelect);
    
    // Remove file functionality
    const removeFileBtn = document.getElementById('removeFile');
    if (removeFileBtn) {
        removeFileBtn.addEventListener('click', clearFileSelection);
    }
}

// Setup form validation
function setupFormValidation() {
    const form = document.getElementById('uploadForm');
    if (form) {
        form.addEventListener('submit', handleFormSubmit);
    }
}

// Prevent default drag behaviors
function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

// Highlight drop zone
function highlight() {
    uploadZone.classList.add('dragover');
}

// Remove highlight from drop zone
function unhighlight() {
    uploadZone.classList.remove('dragover');
}

// Handle dropped files
function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    
    if (files.length > 0) {
        handleFiles(files);
    }
}

// Handle file selection
function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
        handleFiles(files);
    }
}

// Process selected files
function handleFiles(files) {
    if (files.length === 0) return;
    
    const file = files[0]; // Only handle first file
    
    // Validate file
    if (validateFile(file)) {
        currentFile = file;
        displayFilePreview(file);
        hideUploadZone();
        enableSubmitButton();
    }
}

// Validate uploaded file
function validateFile(file) {
    const maxSize = 100 * 1024 * 1024; // 100MB
    const allowedTypes = [
        'image/jpeg', 'image/jpg', 'image/png', 'image/webp', 'image/gif',
        'video/mp4', 'video/avi', 'video/mov', 'video/mkv'
    ];
    
    // Check file size
    if (file.size > maxSize) {
        showNotification('File too large. Maximum size is 100MB.', 'error');
        return false;
    }
    
    // Check file type
    if (!allowedTypes.includes(file.type)) {
        showNotification('Invalid file type. Please upload an image or video.', 'error');
        return false;
    }
    
    return true;
}

// Display file preview
function displayFilePreview(file) {
    if (!filePreview) return;
    
    const fileName = document.querySelector('.file-name');
    const fileSize = document.querySelector('.file-size');
    
    if (fileName) fileName.textContent = file.name;
    if (fileSize) fileSize.textContent = formatFileSize(file.size);
    
    // Show preview container
    filePreview.classList.remove('d-none');
    
    // Create image preview if it's an image
    if (file.type.startsWith('image/')) {
        createImageThumbnail(file);
    }
}

// Create image thumbnail
function createImageThumbnail(file) {
    const reader = new FileReader();
    
    reader.onload = function(e) {
        const thumbnail = document.createElement('img');
        thumbnail.src = e.target.result;
        thumbnail.className = 'img-thumbnail';
        thumbnail.style.cssText = 'width: 60px; height: 60px; object-fit: cover;';
        
        const icon = filePreview.querySelector('.fas.fa-file-image');
        if (icon) {
            icon.parentNode.replaceChild(thumbnail, icon);
        }
    };
    
    reader.readAsDataURL(file);
}

// Hide upload zone
function hideUploadZone() {
    if (uploadZone) {
        uploadZone.style.display = 'none';
    }
}

// Show upload zone
function showUploadZone() {
    if (uploadZone) {
        uploadZone.style.display = 'block';
    }
}

// Clear file selection
function clearFileSelection() {
    currentFile = null;
    
    if (fileInput) fileInput.value = '';
    if (filePreview) filePreview.classList.add('d-none');
    
    showUploadZone();
    disableSubmitButton();
}

// Enable submit button
function enableSubmitButton() {
    const submitBtn = document.getElementById('submitBtn');
    if (submitBtn) {
        submitBtn.disabled = false;
        submitBtn.classList.remove('disabled');
    }
}

// Disable submit button
function disableSubmitButton() {
    const submitBtn = document.getElementById('submitBtn');
    if (submitBtn) {
        submitBtn.disabled = true;
        submitBtn.classList.add('disabled');
    }
}

// Handle form submission
function handleFormSubmit(e) {
    if (!currentFile) {
        e.preventDefault();
        showNotification('Please select a file to upload.', 'error');
        return false;
    }
    
    // Show processing modal if it exists
    const processingModal = document.getElementById('processingModal');
    if (processingModal) {
        const modal = new bootstrap.Modal(processingModal);
        modal.show();
    }
    
    // Add loading state to submit button
    const submitBtn = document.getElementById('submitBtn');
    if (submitBtn) {
        const originalText = submitBtn.innerHTML;
        submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Processing...';
        submitBtn.disabled = true;
        
        // Reset button if there's an error (form doesn't redirect)
        setTimeout(() => {
            if (submitBtn.disabled) {
                submitBtn.innerHTML = originalText;
                submitBtn.disabled = false;
            }
        }, 30000); // 30 second timeout
    }
    
    return true;
}

// Progress tracking for uploads
function trackUploadProgress(file, progressCallback) {
    const xhr = new XMLHttpRequest();
    const formData = new FormData();
    formData.append('file', file);
    
    // Track upload progress
    xhr.upload.addEventListener('progress', (e) => {
        if (e.lengthComputable) {
            const percentComplete = (e.loaded / e.total) * 100;
            if (progressCallback) {
                progressCallback(percentComplete);
            }
        }
    });
    
    // Handle completion
    xhr.addEventListener('load', () => {
        if (xhr.status === 200) {
            try {
                const response = JSON.parse(xhr.responseText);
                if (response.job_id) {
                    window.location.href = `/results/${response.job_id}`;
                }
            } catch (e) {
                showNotification('Upload completed but response parsing failed.', 'error');
            }
        } else {
            showNotification('Upload failed. Please try again.', 'error');
        }
    });
    
    // Handle errors
    xhr.addEventListener('error', () => {
        showNotification('Upload failed. Please check your connection and try again.', 'error');
    });
    
    xhr.open('POST', '/upload');
    xhr.send(formData);
}

// Update progress bar
function updateProgressBar(percent) {
    const progressBar = document.querySelector('.progress-bar');
    if (progressBar) {
        progressBar.style.width = `${percent}%`;
        progressBar.setAttribute('aria-valuenow', percent);
    }
}

// File type detection and icon assignment
function getFileIcon(file) {
    const type = file.type.toLowerCase();
    
    if (type.startsWith('image/')) {
        return 'fa-file-image';
    } else if (type.startsWith('video/')) {
        return 'fa-file-video';
    } else {
        return 'fa-file';
    }
}

// Batch upload functionality (for future enhancement)
function initBatchUpload() {
    const batchUploadZone = document.getElementById('batchUploadZone');
    const batchFileInput = document.getElementById('batchFileInput');
    
    if (batchUploadZone && batchFileInput) {
        // Allow multiple file selection
        batchFileInput.multiple = true;
        
        batchUploadZone.addEventListener('drop', (e) => {
            preventDefaults(e);
            const files = e.dataTransfer.files;
            handleBatchFiles(files);
        });
        
        batchFileInput.addEventListener('change', (e) => {
            handleBatchFiles(e.target.files);
        });
    }
}

// Handle multiple files for batch processing
function handleBatchFiles(files) {
    const fileList = Array.from(files);
    const validFiles = [];
    
    fileList.forEach(file => {
        if (validateFile(file)) {
            validFiles.push(file);
        }
    });
    
    if (validFiles.length > 0) {
        displayBatchFileList(validFiles);
    }
}

// Display list of files for batch processing
function displayBatchFileList(files) {
    const container = document.getElementById('batchFileList');
    if (!container) return;
    
    container.innerHTML = '';
    
    files.forEach((file, index) => {
        const fileItem = document.createElement('div');
        fileItem.className = 'batch-file-item d-flex align-items-center justify-content-between p-3 mb-2 bg-light rounded';
        fileItem.innerHTML = `
            <div class="d-flex align-items-center">
                <i class="fas ${getFileIcon(file)} text-ocean me-3"></i>
                <div>
                    <div class="fw-bold">${file.name}</div>
                    <small class="text-muted">${formatFileSize(file.size)}</small>
                </div>
            </div>
            <div class="batch-file-actions">
                <button type="button" class="btn btn-sm btn-outline-danger" onclick="removeBatchFile(${index})">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        `;
        
        container.appendChild(fileItem);
    });
    
    container.classList.remove('d-none');
}

// Remove file from batch
function removeBatchFile(index) {
    // Implementation for removing files from batch list
    console.log('Remove batch file at index:', index);
}

// Real-time validation feedback
function initRealTimeValidation() {
    const inputs = document.querySelectorAll('input[type="file"]');
    
    inputs.forEach(input => {
        input.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                const isValid = validateFile(file);
                const feedbackElement = input.nextElementSibling;
                
                if (feedbackElement && feedbackElement.classList.contains('invalid-feedback')) {
                    if (isValid) {
                        input.classList.remove('is-invalid');
                        input.classList.add('is-valid');
                    } else {
                        input.classList.add('is-invalid');
                        input.classList.remove('is-valid');
                    }
                }
            }
        });
    });
}

// Clipboard paste support
function initClipboardSupport() {
    document.addEventListener('paste', (e) => {
        const items = e.clipboardData.items;
        
        for (let i = 0; i < items.length; i++) {
            if (items[i].type.indexOf('image') !== -1) {
                const file = items[i].getAsFile();
                if (file) {
                    handleFiles([file]);
                    showNotification('Image pasted from clipboard!', 'success');
                }
                break;
            }
        }
    });
}

// Initialize all upload functionality
document.addEventListener('DOMContentLoaded', function() {
    initUpload();
    initRealTimeValidation();
    initClipboardSupport();
});

// Export functions for global access
window.initUpload = initUpload;
window.initQuickUpload = initQuickUpload;
window.clearFileSelection = clearFileSelection;
window.removeBatchFile = removeBatchFile;
