# backend/api/routes.py
"""
API routes for the ASTER Web Explorer backend
"""

from flask import Blueprint, jsonify, request, send_file
from werkzeug.utils import secure_filename
import logging
import os
from pathlib import Path
import json
import time
import uuid
import shutil

from .controllers import (
    get_all_scenes, get_scene_by_id, upload_scene, process_scene,
    get_processing_status, get_scene_layers, get_scene_statistics,
    generate_prospectivity_map, generate_report, export_map,
    get_prospectivity_areas
)

# Initialize Blueprint
api = Blueprint('api', __name__)

# Set up logging
logger = logging.getLogger(__name__)

@api.route('/scenes', methods=['GET'])
def list_scenes():
    """Get all available scenes"""
    try:
        scenes = get_all_scenes()
        return jsonify(scenes)
    except Exception as e:
        logger.error(f"Error in list_scenes: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@api.route('/scenes/<scene_id>', methods=['GET'])
def get_scene(scene_id):
    """Get scene details by ID"""
    try:
        scene = get_scene_by_id(scene_id)
        if not scene:
            return jsonify({
                'status': 'error',
                'message': f'Scene {scene_id} not found'
            }), 404
        return jsonify(scene)
    except Exception as e:
        logger.error(f"Error in get_scene: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@api.route('/upload', methods=['POST'])
def upload():
    """Upload a new ASTER file"""
    try:
        if 'file' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'No file part in the request'
            }), 400
            
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'status': 'error',
                'message': 'No file selected'
            }), 400
        
        # Get optional metadata
        metadata = {}
        if 'metadata' in request.form:
            try:
                metadata = json.loads(request.form['metadata'])
            except:
                pass
        
        # Process upload
        result = upload_scene(file, metadata)
        
        return jsonify({
            'status': 'success',
            'message': 'File uploaded successfully',
            **result
        })
        
    except Exception as e:
        logger.error(f"Error in upload: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@api.route('/scenes/<scene_id>/process', methods=['POST'])
def start_processing(scene_id):
    """Start processing a scene"""
    try:
        # Get processing options from request body
        options = request.json or {}
        
        # Start processing
        result = process_scene(scene_id, options)
        
        return jsonify({
            'status': 'success',
            'message': 'Processing started',
            **result
        })
        
    except Exception as e:
        logger.error(f"Error in start_processing: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@api.route('/scenes/<scene_id>/status', methods=['GET'])
def check_status(scene_id):
    """Check processing status for a scene"""
    try:
        status = get_processing_status(scene_id)
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error in check_status: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@api.route('/scenes/<scene_id>/layers', methods=['GET'])
def get_layers(scene_id):
    """Get available layers for a scene"""
    try:
        layers = get_scene_layers(scene_id)
        return jsonify(layers)
    except Exception as e:
        logger.error(f"Error in get_layers: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@api.route('/scenes/<scene_id>/layers/<layer_type>/<layer_name>', methods=['GET'])
def get_layer(scene_id, layer_type, layer_name):
    """Get a specific layer as GeoTIFF"""
    try:
        # Map layer type to directory
        type_dir_map = {
            "mineral": "minerals",
            "alteration": "alteration",
            "geological": "analysis",
            "band": "rgb_composites",
            "ratio": "analysis",
            "gold": "gold_pathfinders"
        }
        
        # Map layer type to file suffix
        type_suffix_map = {
            "mineral": "_map.tif",
            "alteration": "_map.tif",
            "geological": "_map.tif",
            "band": "_composite.tif",
            "ratio": "_ratio.tif",
            "gold": "_map.tif"
        }
        
        if layer_type not in type_dir_map:
            return jsonify({
                'status': 'error',
                'message': f'Invalid layer type: {layer_type}'
            }), 400
        
        # Determine the directory and file path
        processed_dir = Path(Config.DATA_DIR) / "processed" / scene_id
        layer_dir = processed_dir / type_dir_map[layer_type]
        layer_file = layer_dir / f"{layer_name}{type_suffix_map.get(layer_type, '.tif')}"
        
        # Check if the file exists
        if not layer_file.exists():
            return jsonify({
                'status': 'error',
                'message': f'Layer file not found: {layer_file}'
            }), 404
        
        # Determine the MIME type
        mime_type = "image/tiff"
        
        # Return the file
        return send_file(
            layer_file,
            mimetype=mime_type,
            as_attachment=False
        )
        
    except Exception as e:
        logger.error(f"Error in get_layer: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@api.route('/scenes/<scene_id>/statistics', methods=['GET'])
def get_statistics(scene_id):
    """Get statistics for a scene"""
    try:
        statistics = get_scene_statistics(scene_id)
        return jsonify(statistics)
    except Exception as e:
        logger.error(f"Error in get_statistics: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@api.route('/scenes/<scene_id>/generate-prospectivity', methods=['POST'])
def create_prospectivity_map(scene_id):
    """Generate a prospectivity map"""
    try:
        options = request.json or {}
        result = generate_prospectivity_map(scene_id, options)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in create_prospectivity_map: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@api.route('/scenes/<scene_id>/report', methods=['POST'])
def create_report(scene_id):
    """Generate a report"""
    try:
        options = request.json or {}
        report_type = options.get('reportType', 'comprehensive')
        result = generate_report(scene_id, report_type, options)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in create_report: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@api.route('/scenes/<scene_id>/export/<map_type>/<map_name>', methods=['GET'])
def download_map(scene_id, map_type, map_name):
    """Export a map as GeoTIFF"""
    try:
        format = request.args.get('format', 'geotiff')
        result = export_map(scene_id, map_type, map_name, format)
        return send_file(
            result['file_path'],
            as_attachment=True,
            download_name=result['file_name'],
            mimetype=result['mime_type']
        )
    except Exception as e:
        logger.error(f"Error in download_map: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@api.route('/scenes/<scene_id>/prospectivity-areas', methods=['GET'])
def fetch_prospectivity_areas(scene_id):
    """Get gold prospectivity areas as GeoJSON"""
    try:
        threshold = float(request.args.get('threshold', 0.7))
        areas = get_prospectivity_areas(scene_id, threshold)
        return jsonify(areas)
    except Exception as e:
        logger.error(f"Error in fetch_prospectivity_areas: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500