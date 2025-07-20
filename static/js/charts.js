// Chart functionality for Bosphorus Ship Detection System

// Chart.js default configuration
Chart.defaults.font.family = "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif";
Chart.defaults.color = '#666';
Chart.defaults.scale.grid.color = '#e9ecef';

// Color scheme
const colors = {
    ocean: '#0a4d52',
    deepOcean: '#0a5d61',
    navy: '#2c3e50',
    lightOcean: '#0d7377',
    success: '#27ae60',
    warning: '#f39c12',
    danger: '#e74c3c',
    info: '#3498db',
    light: '#ecf0f1',
    dark: '#2c3e50'
};

// Create trends chart for dashboard
function createTrendsChart(canvasId, data) {
    const ctx = document.getElementById(canvasId);
    if (!ctx || !data) return;
    
    // Prepare data
    const labels = data.map(item => {
        const date = new Date(item.date);
        return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    });
    
    const imagesData = data.map(item => item.total_images_processed || 0);
    const shipsData = data.map(item => item.total_ships_detected || 0);
    
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Images Processed',
                    data: imagesData,
                    borderColor: colors.ocean,
                    backgroundColor: colors.ocean + '20',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4,
                    pointBackgroundColor: colors.ocean,
                    pointBorderColor: '#fff',
                    pointBorderWidth: 2,
                    pointRadius: 5,
                    pointHoverRadius: 8
                },
                {
                    label: 'Ships Detected',
                    data: shipsData,
                    borderColor: colors.navy,
                    backgroundColor: colors.navy + '20',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4,
                    pointBackgroundColor: colors.navy,
                    pointBorderColor: '#fff',
                    pointBorderWidth: 2,
                    pointRadius: 5,
                    pointHoverRadius: 8
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index'
            },
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        usePointStyle: true,
                        padding: 20,
                        font: {
                            size: 14,
                            weight: '600'
                        }
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: '#fff',
                    bodyColor: '#fff',
                    borderColor: colors.ocean,
                    borderWidth: 1,
                    cornerRadius: 8,
                    displayColors: true,
                    callbacks: {
                        title: function(tooltipItems) {
                            return `Date: ${tooltipItems[0].label}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        font: {
                            size: 12,
                            weight: '500'
                        }
                    }
                },
                y: {
                    beginAtZero: true,
                    grid: {
                        color: '#f0f0f0',
                        lineWidth: 1
                    },
                    ticks: {
                        font: {
                            size: 12,
                            weight: '500'
                        }
                    }
                }
            },
            elements: {
                point: {
                    hoverRadius: 10
                }
            }
        }
    });
}

// Create accuracy donut chart
function createAccuracyChart(canvasId, accuracy) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;
    
    const remaining = 100 - accuracy;
    
    new Chart(ctx, {
        type: 'doughnut',
        data: {
            datasets: [{
                data: [accuracy, remaining],
                backgroundColor: [
                    colors.ocean,
                    '#e9ecef'
                ],
                borderWidth: 0,
                cutout: '70%'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    enabled: false
                }
            }
        }
    });
}

// Create pie chart for vessel types
function createPieChart(canvasId, title, data) {
    const ctx = document.getElementById(canvasId);
    if (!ctx || !data) return;
    
    const labels = Object.keys(data);
    const values = Object.values(data);
    
    // Generate colors for each vessel type
    const backgroundColors = labels.map((_, index) => {
        const colorKeys = Object.keys(colors);
        return colors[colorKeys[index % colorKeys.length]];
    });
    
    new Chart(ctx, {
        type: 'pie',
        data: {
            labels: labels,
            datasets: [{
                data: values,
                backgroundColor: backgroundColors,
                borderColor: '#fff',
                borderWidth: 2,
                hoverBorderWidth: 3
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        usePointStyle: true,
                        padding: 15,
                        font: {
                            size: 12,
                            weight: '500'
                        }
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: '#fff',
                    bodyColor: '#fff',
                    borderColor: colors.ocean,
                    borderWidth: 1,
                    cornerRadius: 8,
                    callbacks: {
                        label: function(context) {
                            const label = context.label || '';
                            const value = context.parsed || 0;
                            const total = context.dataset.data.reduce((a, b) => a + b, 0);
                            const percentage = ((value / total) * 100).toFixed(1);
                            return `${label}: ${value} (${percentage}%)`;
                        }
                    }
                }
            }
        }
    });
}

// Create confidence distribution histogram
function createConfidenceChart(canvasId, confidenceScores) {
    const ctx = document.getElementById(canvasId);
    if (!ctx || !confidenceScores || confidenceScores.length === 0) return;
    
    // Create bins for confidence scores
    const bins = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    const binLabels = ['30-40%', '40-50%', '50-60%', '60-70%', '70-80%', '80-90%', '90-100%'];
    const binCounts = new Array(bins.length - 1).fill(0);
    
    // Count scores in each bin
    confidenceScores.forEach(score => {
        for (let i = 0; i < bins.length - 1; i++) {
            if (score >= bins[i] && score < bins[i + 1]) {
                binCounts[i]++;
                break;
            }
        }
    });
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: binLabels,
            datasets: [{
                label: 'Detection Count',
                data: binCounts,
                backgroundColor: colors.ocean + '80',
                borderColor: colors.ocean,
                borderWidth: 2,
                borderRadius: 4,
                borderSkipped: false
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: '#fff',
                    bodyColor: '#fff',
                    borderColor: colors.ocean,
                    borderWidth: 1,
                    cornerRadius: 8
                }
            },
            scales: {
                x: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        font: {
                            size: 10,
                            weight: '500'
                        }
                    }
                },
                y: {
                    beginAtZero: true,
                    grid: {
                        color: '#f0f0f0'
                    },
                    ticks: {
                        stepSize: 1,
                        font: {
                            size: 12,
                            weight: '500'
                        }
                    }
                }
            }
        }
    });
}

// Create processing time chart
function createProcessingTimeChart(canvasId, jobs) {
    const ctx = document.getElementById(canvasId);
    if (!ctx || !jobs) return;
    
    const labels = jobs.map((job, index) => `Job ${index + 1}`);
    const times = jobs.map(job => job.processing_time || 0);
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Processing Time (seconds)',
                data: times,
                backgroundColor: colors.info + '80',
                borderColor: colors.info,
                borderWidth: 2,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const value = context.parsed.y;
                            return `Processing Time: ${value.toFixed(2)}s`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: {
                        display: false
                    }
                },
                y: {
                    beginAtZero: true,
                    grid: {
                        color: '#f0f0f0'
                    },
                    ticks: {
                        callback: function(value) {
                            return value + 's';
                        }
                    }
                }
            }
        }
    });
}

// Create animated progress ring
function createProgressRing(canvasId, value, maxValue, label) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;
    
    const percentage = (value / maxValue) * 100;
    
    new Chart(ctx, {
        type: 'doughnut',
        data: {
            datasets: [{
                data: [percentage, 100 - percentage],
                backgroundColor: [
                    colors.ocean,
                    '#e9ecef'
                ],
                borderWidth: 0,
                cutout: '80%'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    enabled: false
                }
            },
            animation: {
                animateRotate: true,
                duration: 2000,
                easing: 'easeOutQuart'
            }
        },
        plugins: [{
            id: 'centerText',
            beforeDraw: function(chart) {
                const ctx = chart.ctx;
                const centerX = (chart.chartArea.left + chart.chartArea.right) / 2;
                const centerY = (chart.chartArea.top + chart.chartArea.bottom) / 2;
                
                ctx.save();
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                
                // Draw percentage
                ctx.font = 'bold 24px Arial';
                ctx.fillStyle = colors.ocean;
                ctx.fillText(`${percentage.toFixed(1)}%`, centerX, centerY - 10);
                
                // Draw label
                ctx.font = '12px Arial';
                ctx.fillStyle = '#666';
                ctx.fillText(label, centerX, centerY + 15);
                
                ctx.restore();
            }
        }]
    });
}

// Create real-time detection chart
function createRealtimeChart(canvasId) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;
    
    const chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Ships Detected',
                data: [],
                borderColor: colors.ocean,
                backgroundColor: colors.ocean + '20',
                borderWidth: 2,
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    type: 'linear',
                    position: 'bottom'
                },
                y: {
                    beginAtZero: true
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            },
            animation: {
                duration: 0
            }
        }
    });
    
    return chart;
}

// Update real-time chart with new data
function updateRealtimeChart(chart, timestamp, value) {
    if (!chart) return;
    
    chart.data.labels.push(timestamp);
    chart.data.datasets[0].data.push(value);
    
    // Keep only last 20 data points
    if (chart.data.labels.length > 20) {
        chart.data.labels.shift();
        chart.data.datasets[0].data.shift();
    }
    
    chart.update('none');
}

// Create comparison chart
function createComparisonChart(canvasId, beforeData, afterData) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;
    
    new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['Accuracy', 'Speed', 'Confidence', 'Recall', 'Precision'],
            datasets: [
                {
                    label: 'Before',
                    data: beforeData,
                    borderColor: colors.danger,
                    backgroundColor: colors.danger + '20',
                    borderWidth: 2,
                    pointBackgroundColor: colors.danger
                },
                {
                    label: 'After',
                    data: afterData,
                    borderColor: colors.ocean,
                    backgroundColor: colors.ocean + '20',
                    borderWidth: 2,
                    pointBackgroundColor: colors.ocean
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        usePointStyle: true
                    }
                }
            },
            scales: {
                r: {
                    beginAtZero: true,
                    max: 100,
                    grid: {
                        color: '#e9ecef'
                    },
                    pointLabels: {
                        font: {
                            size: 12,
                            weight: '500'
                        }
                    }
                }
            }
        }
    });
}

// Utility function to generate gradient
function createGradient(ctx, colorStart, colorEnd) {
    const gradient = ctx.createLinearGradient(0, 0, 0, 400);
    gradient.addColorStop(0, colorStart);
    gradient.addColorStop(1, colorEnd);
    return gradient;
}

// Export chart creation functions
window.createTrendsChart = createTrendsChart;
window.createAccuracyChart = createAccuracyChart;
window.createPieChart = createPieChart;
window.createConfidenceChart = createConfidenceChart;
window.createProcessingTimeChart = createProcessingTimeChart;
window.createProgressRing = createProgressRing;
window.createRealtimeChart = createRealtimeChart;
window.updateRealtimeChart = updateRealtimeChart;
window.createComparisonChart = createComparisonChart;
