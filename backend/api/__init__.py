# backend/api/__init__.py
"""
API Blueprint initialization
"""

from flask import Blueprint

def create_api_blueprint():
    """
    Create the API Blueprint
    
    Returns:
    --------
    Blueprint
        The API Blueprint
    """
    from .routes import api
    return api