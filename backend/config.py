# backend/config.py
"""
Configuration settings for the ASTER Web Explorer backend
"""

import os
from pathlib import Path

class Config:
    """Application configuration"""
    
    # API and server settings
    VERSION = '1.0.0'
    SERVER_URL = os.getenv('SERVER_URL', 'http://localhost:5000')
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*')
    
    # Data directory settings
    BASE_DIR = Path(os.getenv('BASE_DIR', os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR = os.getenv('DATA_DIR', os.path.join(BASE_DIR, '..', 'data'))
    
    # Logging settings
    LOG_DIR = os.getenv('LOG_DIR', os.path.join(BASE_DIR, 'logs'))
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    
    # Processing settings
    MAX_UPLOAD_SIZE = int(os.getenv('MAX_UPLOAD_SIZE', 1024 * 1024 * 1024))  # 1GB
    PROCESS_TIMEOUT = int(os.getenv('PROCESS_TIMEOUT', 3600))  # 1 hour
    MAX_CONCURRENT_PROCESSES = int(os.getenv('MAX_CONCURRENT_PROCESSES', 2))
    
    # Feature flags
    ENABLE_AI_ANALYSIS = os.getenv('ENABLE_AI_ANALYSIS', 'false').lower() == 'true'