import os
import time
import uuid
from datetime import datetime, timedelta
from flask import render_template, request, redirect, url_for, flash, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from app import app, db
from models import DetectionJob, DetectionStatistics
from yolo_detector import YOLOShipDetector
from utils import allowed_file, update_statistics
import logging
import json

logger = logging.getLogger(__name__)

# Initialize YOLO detector
detector = YOLOShipDetector()

@app.route('/')
def index():
    """Landing page"""
    # Get recent statistics for display
    recent_stats = DetectionStatistics.query.order_by(DetectionStatistics.date.desc()).first()
    recent_jobs = DetectionJob.query.filter_by(status='completed').order_by(DetectionJob.upload_time.desc()).limit(3).all()
    
    return render_template('index.html', 
                         stats=recent_stats,
                         recent_jobs=recent_jobs)

@app.route('/upload')
def upload_page():
    """Upload interface page"""
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and start processing"""
    try:
        if 'file' not in request.files:
            if 'XMLHttpRequest' in request.headers.get('X-Requested-With', '') or request.is_json:
                return jsonify({'error': 'No file selected'}), 400
            flash('No file selected', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '' or file.filename is None:
            if 'XMLHttpRequest' in request.headers.get('X-Requested-With', '') or request.is_json:
                return jsonify({'error': 'No file selected'}), 400
            flash('No file selected', 'error')
            return redirect(request.url)
        
        if file and file.filename and allowed_file(file.filename):
            # Generate unique filename
            filename = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Create database record
            job = DetectionJob(
                filename=filename,
                original_filename=file.filename,
                status='pending'
            )
            db.session.add(job)
            db.session.commit()
            
            logger.info(f"File uploaded: {filename}, Job ID: {job.id}")
            
            # Process the image
            try:
                job.status = 'processing'
                db.session.commit()
                
                start_time = time.time()
                
                # Run YOLO detection
                detections = detector.detect_ships(file_path, confidence_threshold=0.3)
                
                # Create annotated image
                result_filename = f"result_{filename}"
                result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
                detector.draw_detections(file_path, detections, result_path)
                
                processing_time = time.time() - start_time
                
                # Update job with results
                job.status = 'completed'
                job.processing_time = processing_time
                job.ship_count = detections['ship_count']
                job.confidence_scores = detections['confidence_scores']
                job.bounding_boxes = detections['bounding_boxes']
                job.vessel_types = detections['vessel_types']
                job.result_image_path = result_path
                job.processed_image_path = file_path
                
                db.session.commit()
                
                # Update statistics
                update_statistics(detections)
                
                logger.info(f"Processing completed for job {job.id}")
                
                # Return JSON response for AJAX requests
                if 'XMLHttpRequest' in request.headers.get('X-Requested-With', '') or request.is_json:
                    return jsonify({
                        'success': True,
                        'redirect': url_for('results', job_id=job.id),
                        'job_id': job.id
                    })
                else:
                    return redirect(url_for('results', job_id=job.id))
                
            except Exception as e:
                logger.error(f"Processing error for job {job.id}: {e}")
                job.status = 'failed'
                job.error_message = str(e)
                db.session.commit()
                
                if 'XMLHttpRequest' in request.headers.get('X-Requested-With', '') or request.is_json:
                    return jsonify({'error': f'Processing failed: {str(e)}'}), 500
                flash(f'Processing failed: {str(e)}', 'error')
                return redirect(url_for('upload_page'))
        
        else:
            if 'XMLHttpRequest' in request.headers.get('X-Requested-With', '') or request.is_json:
                return jsonify({'error': 'Invalid file type. Please upload an image (JPG, JPEG, PNG) or video (MP4, AVI, MOV)'}), 400
            flash('Invalid file type. Please upload an image (JPG, JPEG, PNG) or video (MP4, AVI, MOV)', 'error')
            return redirect(request.url)
            
    except Exception as e:
        logger.error(f"Upload error: {e}")
        if 'XMLHttpRequest' in request.headers.get('X-Requested-With', '') or request.is_json:
            return jsonify({'error': f'Upload failed: {str(e)}'}), 500
        flash(f'Upload failed: {str(e)}', 'error')
        return redirect(url_for('upload_page'))

@app.route('/results/<int:job_id>')
def results(job_id):
    """Display detection results"""
    job = DetectionJob.query.get_or_404(job_id)
    
    # Get detection summary
    if job.status == 'completed':
        detections = {
            'ship_count': job.ship_count,
            'confidence_scores': job.confidence_scores or [],
            'vessel_types': job.vessel_types or [],
            'bounding_boxes': job.bounding_boxes or []
        }
        summary = detector.get_detection_summary(detections)
    else:
        summary = None
    
    return render_template('results.html', job=job, summary=summary)

@app.route('/dashboard')
def dashboard():
    """Statistics dashboard"""
    # Get all statistics
    stats = DetectionStatistics.query.order_by(DetectionStatistics.date.desc()).limit(30).all()
    
    # Get recent jobs
    recent_jobs = DetectionJob.query.order_by(DetectionJob.upload_time.desc()).limit(10).all()
    
    # Calculate total statistics
    total_images = DetectionJob.query.filter_by(status='completed').count()
    total_ships = db.session.query(db.func.sum(DetectionJob.ship_count)).filter_by(status='completed').scalar() or 0
    
    # Average processing time
    avg_processing_time = db.session.query(db.func.avg(DetectionJob.processing_time)).filter_by(status='completed').scalar() or 0
    
    # Success rate
    total_jobs = DetectionJob.query.count()
    success_rate = (total_images / total_jobs * 100) if total_jobs > 0 else 0
    
    dashboard_stats = {
        'total_images': total_images,
        'total_ships': int(total_ships),
        'avg_processing_time': round(avg_processing_time, 2),
        'success_rate': round(success_rate, 1)
    }
    
    return render_template('dashboard.html', 
                         stats=stats, 
                         recent_jobs=recent_jobs,
                         dashboard_stats=dashboard_stats)

@app.route('/about')
def about():
    """About page with technical details"""
    return render_template('about.html')

@app.route('/methodology')
def methodology():
    """Research methodology and system architecture"""
    return render_template('methodology.html')

@app.route('/benchmarks')
def benchmarks():
    """Performance benchmarks and academic validation"""
    return render_template('benchmarks.html')

@app.route('/export')
def export_results():
    """Export detection results for academic analysis"""
    return render_template('export_results.html')

@app.route('/results/<filename>')
def serve_result_image(filename):
    """Serve result images"""
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

@app.route('/api/job_status/<int:job_id>')
def job_status(job_id):
    """API endpoint to check job status"""
    job = DetectionJob.query.get_or_404(job_id)
    return jsonify(job.to_dict())

@app.route('/api/statistics')
def api_statistics():
    """API endpoint for statistics data"""
    # Get last 30 days of statistics
    thirty_days_ago = datetime.utcnow().date() - timedelta(days=30)
    stats = DetectionStatistics.query.filter(DetectionStatistics.date >= thirty_days_ago).order_by(DetectionStatistics.date.asc()).all()
    
    return jsonify([stat.to_dict() for stat in stats])

@app.route('/download/result/<int:job_id>')
def download_result(job_id):
    """Download detection result image"""
    job = DetectionJob.query.get_or_404(job_id)
    
    if job.status == 'completed' and job.result_image_path:
        directory = os.path.dirname(job.result_image_path)
        filename = os.path.basename(job.result_image_path)
        return send_from_directory(directory, filename, as_attachment=True)
    else:
        flash('Result not available', 'error')
        return redirect(url_for('results', job_id=job_id))

@app.route('/download/report/<int:job_id>')
def download_report(job_id):
    """Download detection report as JSON"""
    job = DetectionJob.query.get_or_404(job_id)
    
    if job.status == 'completed':
        report = {
            'job_id': job.id,
            'original_filename': job.original_filename,
            'upload_time': job.upload_time.isoformat(),
            'processing_time': job.processing_time,
            'ship_count': job.ship_count,
            'confidence_scores': job.confidence_scores,
            'bounding_boxes': job.bounding_boxes,
            'vessel_types': job.vessel_types,
            'summary': detector.get_detection_summary({
                'ship_count': job.ship_count,
                'confidence_scores': job.confidence_scores or [],
                'vessel_types': job.vessel_types or [],
                'bounding_boxes': job.bounding_boxes or []
            })
        }
        
        # Create temporary JSON file
        import tempfile
        import json
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(report, f, indent=2)
            temp_path = f.name
        
        return send_from_directory(
            os.path.dirname(temp_path), 
            os.path.basename(temp_path),
            as_attachment=True,
            download_name=f'detection_report_{job_id}.json'
        )
    else:
        flash('Report not available', 'error')
        return redirect(url_for('results', job_id=job_id))

@app.errorhandler(413)
def file_too_large(e):
    flash('File too large. Maximum size is 100MB.', 'error')
    return redirect(url_for('upload_page'))

@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    logger.error(f"Server error: {e}")
    return render_template('500.html'), 500
