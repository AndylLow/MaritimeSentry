from app import db
from datetime import datetime
from sqlalchemy import Text, JSON

class DetectionJob(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    upload_time = db.Column(db.DateTime, default=datetime.utcnow)
    processing_time = db.Column(db.Float)  # Time in seconds
    status = db.Column(db.String(50), default='pending')  # pending, processing, completed, failed
    error_message = db.Column(Text)
    
    # Detection results
    ship_count = db.Column(db.Integer, default=0)
    confidence_scores = db.Column(JSON)  # List of confidence scores
    bounding_boxes = db.Column(JSON)  # List of bounding box coordinates
    vessel_types = db.Column(JSON)  # List of detected vessel types
    
    # File paths
    result_image_path = db.Column(db.String(255))
    processed_image_path = db.Column(db.String(255))
    
    def to_dict(self):
        return {
            'id': self.id,
            'filename': self.filename,
            'original_filename': self.original_filename,
            'upload_time': self.upload_time.isoformat() if self.upload_time else None,
            'processing_time': self.processing_time,
            'status': self.status,
            'error_message': self.error_message,
            'ship_count': self.ship_count,
            'confidence_scores': self.confidence_scores,
            'bounding_boxes': self.bounding_boxes,
            'vessel_types': self.vessel_types,
            'result_image_path': self.result_image_path,
            'processed_image_path': self.processed_image_path
        }

class DetectionStatistics(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, default=datetime.utcnow().date)
    total_images_processed = db.Column(db.Integer, default=0)
    total_ships_detected = db.Column(db.Integer, default=0)
    average_confidence = db.Column(db.Float, default=0.0)
    vessel_type_counts = db.Column(JSON)  # Dictionary of vessel types and counts
    
    def to_dict(self):
        return {
            'id': self.id,
            'date': self.date.isoformat() if self.date else None,
            'total_images_processed': self.total_images_processed,
            'total_ships_detected': self.total_ships_detected,
            'average_confidence': self.average_confidence,
            'vessel_type_counts': self.vessel_type_counts
        }
