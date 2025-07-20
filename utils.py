import os
from datetime import datetime, date
from app import db
from models import DetectionStatistics
import logging

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp', 'mp4', 'avi', 'mov', 'mkv'}

def allowed_file(filename):
    """Check if file has allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def update_statistics(detections):
    """Update daily statistics with new detection results"""
    try:
        today = date.today()
        
        # Get or create today's statistics
        stats = DetectionStatistics.query.filter_by(date=today).first()
        if not stats:
            stats = DetectionStatistics(date=today)
            db.session.add(stats)
        
        # Update statistics
        stats.total_images_processed += 1
        stats.total_ships_detected += detections['ship_count']
        
        # Calculate average confidence
        if detections['confidence_scores']:
            current_total_confidence = stats.average_confidence * (stats.total_images_processed - 1)
            new_avg_confidence = sum(detections['confidence_scores']) / len(detections['confidence_scores'])
            stats.average_confidence = (current_total_confidence + new_avg_confidence) / stats.total_images_processed
        
        # Update vessel type counts
        if not stats.vessel_type_counts:
            stats.vessel_type_counts = {}
        
        for vessel_type in detections['vessel_types']:
            if vessel_type in stats.vessel_type_counts:
                stats.vessel_type_counts[vessel_type] += 1
            else:
                stats.vessel_type_counts[vessel_type] = 1
        
        db.session.commit()
        logger.info(f"Statistics updated for {today}")
        
    except Exception as e:
        logger.error(f"Error updating statistics: {e}")
        db.session.rollback()

def format_processing_time(seconds):
    """Format processing time in human-readable format"""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"

def format_confidence(confidence):
    """Format confidence score as percentage"""
    return f"{confidence * 100:.1f}%"

def get_file_size(file_path):
    """Get file size in human-readable format"""
    try:
        size_bytes = os.path.getsize(file_path)
        
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024**2:
            return f"{size_bytes/1024:.1f} KB"
        elif size_bytes < 1024**3:
            return f"{size_bytes/(1024**2):.1f} MB"
        else:
            return f"{size_bytes/(1024**3):.1f} GB"
    except:
        return "Unknown"

def validate_image_file(file_path):
    """Validate that uploaded file is a valid image"""
    try:
        from PIL import Image
        with Image.open(file_path) as img:
            img.verify()
        return True
    except:
        return False
