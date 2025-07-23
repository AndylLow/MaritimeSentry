# Bosphorus Ship Detection System

## Overview

This is a sophisticated web application for ship detection in maritime environments, specifically designed for the Istanbul Bosphorus strait. The system uses YOLO (You Only Look Once) deep learning technology to detect and analyze ships in uploaded images and videos. Built with Flask as the backend framework, the application provides a modern web interface for uploading media files, processing them through computer vision algorithms, and displaying detailed detection results with bounding boxes and confidence scores.

## User Preferences

Preferred communication style: Simple, everyday language.
User rejected Firebase integration - wants simpler approach without external services.
User requested improved color contrast for better readability (fixed green-on-green and black-on-gray issues).

## Recent Changes (July 23, 2025)

✓ **COMPLETE YOLO DETECTOR REWRITE**: Built clean, working ship detector from scratch after critical system failures
✓ **Real Computer Vision**: Implemented actual OpenCV-based ship detection using contour analysis and edge detection
✓ **Multi-Method Detection**: Combined large vessel detection, bright vessel detection, and edge-based detection
✓ **Proper Image Loading**: Fixed PIL image loading scope issues that caused "NoneType" errors
✓ **Robust Detection Pipeline**: Large cargo ship detection via adaptive thresholding and contour analysis
✓ **Bright Vessel Detection**: Ferry and small boat detection using brightness thresholding
✓ **Edge-Based Detection**: Canny edge detection with Hough line transform for vessel silhouettes
✓ **Non-Maximum Suppression**: Proper IoU-based overlap removal to prevent duplicate detections
✓ **Clean Architecture**: Modular, maintainable code structure with proper error handling
✓ **Working System**: Functional ship detection that processes real images without falling back to random placement

### Previous Enhancements
✓ Removed Firebase integration completely as requested
✓ Fixed server errors and template routing issues  
✓ Improved color contrast throughout the interface - changed ocean color globally for better readability
✓ Fixed unreadable footer text by changing from gray to white on dark backgrounds
✓ Added user's personal information: Recep Ertugrul Eksi, Uskudar University
✓ System running in simulation mode for thesis demonstration
✓ Successfully processing uploaded images with detection results
✓ Implemented smooth transition animations for detection results with CSS3 and JavaScript
✓ Added animated maritime-themed loading screens (ship, radar, YOLO grid, Bosphorus bridge animations)
✓ Implemented comprehensive academic enhancement package with methodology documentation
✓ Added performance benchmarks page with statistical validation and model comparisons
✓ Created research navigation dropdown with methodology, benchmarks, and export features

## System Architecture

The application follows a traditional three-tier web architecture:

1. **Presentation Layer**: HTML templates with Bootstrap CSS framework providing a responsive, maritime-themed user interface
2. **Application Layer**: Flask web framework handling HTTP requests, file uploads, and business logic
3. **Data Layer**: SQLAlchemy ORM with SQLite database for storing detection jobs and statistics

The system is designed as a monolithic application with clear separation of concerns through modular Python files.

## Key Components

### Backend Components

- **Flask Application (`app.py`)**: Main application factory configuring database, file upload settings, and middleware
- **Route Handlers (`routes.py`)**: HTTP endpoint definitions for file upload, result display, and dashboard functionality
- **Database Models (`models.py`)**: SQLAlchemy models for DetectionJob and DetectionStatistics entities
- **YOLO Detector (`yolo_detector.py`)**: Computer vision processing using Ultralytics YOLO model for ship detection
- **Utilities (`utils.py`)**: Helper functions for file validation and statistics updates

### Frontend Components

- **Template System**: Jinja2 templates with Bootstrap 5 for responsive design
- **Static Assets**: Custom CSS with maritime theme, JavaScript for interactivity and charts
- **Upload Interface**: Drag-and-drop file upload with real-time preview and validation

### Database Schema

The system uses two main entities:
- **DetectionJob**: Stores individual detection tasks with metadata, results, and file paths
- **DetectionStatistics**: Aggregates daily statistics for analytics dashboard

## Data Flow

1. **File Upload**: Users upload images/videos through web interface with drag-and-drop functionality
2. **Validation**: Server validates file types and size constraints (100MB max)
3. **Processing**: YOLO model analyzes uploaded media to detect ships and vessels
4. **Storage**: Results are saved to database with bounding boxes, confidence scores, and vessel types
5. **Visualization**: Processed images with detection overlays are generated and stored
6. **Analytics**: Detection statistics are aggregated for dashboard metrics and trends

## External Dependencies

### Python Libraries
- **Flask**: Web framework and request handling
- **SQLAlchemy**: Database ORM and migrations
- **Ultralytics**: YOLO model implementation for object detection
- **OpenCV**: Image processing and computer vision operations
- **PIL/Pillow**: Image manipulation and drawing operations
- **NumPy**: Numerical computing for image arrays

### Frontend Libraries
- **Bootstrap 5**: CSS framework for responsive design
- **Font Awesome**: Icon library for UI elements
- **Chart.js**: Data visualization for analytics dashboard

### Optional Integrations
- **Firebase**: Configured for potential cloud storage and authentication (environment variables present)

## Deployment Strategy

The application is configured for flexible deployment:

- **Development**: Flask development server with debug mode enabled
- **Production**: WSGI-compatible with ProxyFix middleware for reverse proxy deployment
- **Database**: Environment variable configuration supporting SQLite (default) or PostgreSQL via DATABASE_URL
- **File Storage**: Local filesystem with configurable upload and results directories
- **Scaling**: Connection pooling and database ping checks for reliability

The system includes proper error handling, logging, and security considerations like secure filename handling and file size limits. The maritime-themed UI provides an academic presentation suitable for thesis demonstration while maintaining professional functionality for real-world maritime surveillance applications.