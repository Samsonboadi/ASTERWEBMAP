#!/usr/bin/env python3
# backend/app.py
"""
Main Flask application for the ASTER Web Explorer backend
"""

import os
import json
import logging
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

from api import create_api_blueprint
from config import Config
from utils.enhanced_logging import setup_logging

# Initialize application
app = Flask(__name__)
app.config.from_object(Config)

# Configure CORS
CORS(app, resources={r"/api/*": {"origins": app.config.get('CORS_ORIGINS', '*')}})

# Setup logging
log_dir = Path(app.config.get('LOG_DIR', 'logs'))
log_dir.mkdir(parents=True, exist_ok=True)
logger = setup_logging(log_dir=log_dir, module_name="aster_backend")

# Register API Blueprint
api_blueprint = create_api_blueprint()
app.register_blueprint(api_blueprint, url_prefix='/api')

@app.route('/')
def index():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "version": app.config.get('VERSION', '1.0.0'),
        "name": "ASTER Web Explorer API"
    })

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'status': 'error',
        'message': 'Resource not found',
        'error': str(error)
    }), 404

@app.errorhandler(500)
def internal_server_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({
        'status': 'error',
        'message': 'Internal server error',
        'error': str(error)
    }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV", "production") == "development"
    
    # Ensure data directories exist
    for directory in ['data/raw', 'data/processed', 'data/output']:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    app.run(host="0.0.0.0", port=port, debug=debug)