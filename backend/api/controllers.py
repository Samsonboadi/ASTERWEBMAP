# backend/api/controllers.py
"""
Controller functions for the ASTER Web Explorer backend
"""

import os
import json
import logging
import datetime
import subprocess
import threading
import uuid
import glob
import zipfile
from pathlib import Path
import shutil
import tempfile
from werkzeug.utils import secure_filename
import rasterio
import numpy as np
import geopandas as gpd
from shapely.geometry import shape, mapping
import mimetypes

from config import Config
from enums import ProcessingStatus, ProcessingStages, ReportTypes
from processors.aster_processor import ASTERProcessor
from processors.aster_l2_processor import ASTER_L2_Processor, MineralIndices, SceneMetadata, GeographicBounds
from processors.aster_geological_mapper import ASTER_Geological_Mapper, AlterationIndices, GeologicalFeatures
from processors.aster_band_math import MapCombiner, TargetedMapGenerator, CombinationMethod
from processors.aster_advanced_analysis import ASTER_Advanced_Analysis
from processors.gold_prospectivity_mapper import GoldProspectivityMapper


# Initialize logger
logger = logging.getLogger(__name__)

# Mock database for development - in production, use a real database
# In-memory storage for scenes and processing status
_scenes = {}
_processing_status = {}
_processing_threads = {}

def get_all_scenes():
    """
    Get all available scenes
    
    Returns:
    --------
    list
        List of scene objects
    """
    # In production, query database
    data_dir = Path(Config.DATA_DIR) / "processed"
    data_dir.mkdir(parents=True, exist_ok=True)
    scene_dirs = [d for d in data_dir.glob("*") if d.is_dir()]
    
    scenes = []
    for scene_dir in scene_dirs:
        # Try to read metadata
        metadata_path = scene_dir / "metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    scenes.append({
                        "id": scene_dir.name,
                        "name": metadata.get("name", scene_dir.name),
                        "date": metadata.get("acquisition_date", "Unknown"),
                        "cloudCover": metadata.get("cloud_cover", 0),
                        "status": _processing_status.get(scene_dir.name, {}).get("status", ProcessingStatus.IDLE.value)
                    })
            except Exception as e:
                logger.error(f"Error reading metadata for {scene_dir.name}: {str(e)}")
                # Add with minimal information
                scenes.append({
                    "id": scene_dir.name,
                    "name": scene_dir.name,
                    "date": "Unknown",
                    "cloudCover": 0,
                    "status": _processing_status.get(scene_dir.name, {}).get("status", ProcessingStatus.IDLE.value)
                })
        else:
            # Add with minimal information
            scenes.append({
                "id": scene_dir.name,
                "name": scene_dir.name,
                "date": "Unknown",
                "cloudCover": 0,
                "status": _processing_status.get(scene_dir.name, {}).get("status", ProcessingStatus.IDLE.value)
            })
    
    return scenes

def get_scene_by_id(scene_id):
    """
    Get scene details by ID
    
    Parameters:
    -----------
    scene_id : str
        Scene ID
        
    Returns:
    --------
    dict
        Scene object or None if not found
    """
    # Try to read metadata
    metadata_path = Path(Config.DATA_DIR) / "processed" / scene_id / "metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                
                # Get processing status
                status_info = _processing_status.get(scene_id, {})
                status = status_info.get("status", ProcessingStatus.IDLE.value)
                
                # Get bounds from metadata
                bounds = metadata.get("bounds", None)
                
                # Check for thumbnail
                thumbnail_path = Path(Config.DATA_DIR) / "processed" / scene_id / "thumbnail.png"
                has_thumbnail = thumbnail_path.exists()
                
                # Combine metadata with status
                scene = {
                    "id": scene_id,
                    "name": metadata.get("name", scene_id),
                    "date": metadata.get("acquisition_date", "Unknown"),
                    "cloudCover": metadata.get("cloud_cover", 0),
                    "status": status,
                    "bounds": bounds,
                    "thumbnail": f"/api/scenes/{scene_id}/thumbnail" if has_thumbnail else None
                }
                
                return scene
        except Exception as e:
            logger.error(f"Error reading metadata for {scene_id}: {str(e)}")
            return None
    else:
        # Look for the directory without metadata
        scene_dir = Path(Config.DATA_DIR) / "processed" / scene_id
        if scene_dir.exists() and scene_dir.is_dir():
            return {
                "id": scene_id,
                "name": scene_id,
                "date": "Unknown",
                "cloudCover": 0,
                "status": _processing_status.get(scene_id, {}).get("status", ProcessingStatus.IDLE.value)
            }
        return None

def upload_scene(file, metadata=None):
    """
    Upload a new scene file
    
    Parameters:
    -----------
    file : FileStorage
        Uploaded file
    metadata : dict, optional
        Additional metadata
        
    Returns:
    --------
    dict
        Upload result with scene ID
    """
    # Generate a unique scene ID
    scene_id = f"scene_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    # Create scene directory
    scene_dir = Path(Config.DATA_DIR) / "raw" / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)
    
    # Save file
    filename = secure_filename(file.filename)
    file_path = scene_dir / filename
    file.save(str(file_path))
    
    # Save metadata
    if metadata is None:
        metadata = {}
    
    # Add original filename and upload date to metadata
    metadata.update({
        "original_filename": filename,
        "upload_date": datetime.datetime.now().isoformat(),
        "file_size": os.path.getsize(file_path)
    })
    
    # Write metadata to file
    with open(scene_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Update processing status
    _processing_status[scene_id] = {
        "status": ProcessingStatus.IDLE.value,
        "progress": 0,
        "stage": None,
        "error": None,
        "start_time": None,
        "end_time": None
    }
    
    logger.info(f"Uploaded new scene {scene_id}: {filename}")
    
    return {
        "sceneId": scene_id,
        "filename": filename
    }


def _process_scene_worker(scene_id, options):
    """
    Worker function for processing a scene in the background
    
    Parameters:
    -----------
    scene_id : str
        Scene ID
    options : dict
        Processing options
    """
    try:
        # Update status to PROCESSING
        _processing_status[scene_id] = {
            "status": ProcessingStatus.PROCESSING.value,
            "progress": 0,
            "stage": ProcessingStages.EXTRACT.value,
            "error": None,
            "start_time": datetime.datetime.now().isoformat(),
            "end_time": None
        }
        
        logger.info(f"Started processing scene {scene_id} with options: {options}")
        
        # Initialize the processor
        processor = ASTERProcessor(scene_id, Config.DATA_DIR)
        
        # Process the scene
        success = processor.process(options)
        
        if success:
            # Update status to COMPLETED
            _processing_status[scene_id] = {
                "status": ProcessingStatus.COMPLETED.value,
                "progress": 100,
                "stage": None,
                "error": None,
                "start_time": _processing_status[scene_id]["start_time"],
                "end_time": datetime.datetime.now().isoformat()
            }
            
            logger.info(f"Processing completed for scene {scene_id}")
        else:
            # Update status to FAILED
            _processing_status[scene_id] = {
                "status": ProcessingStatus.FAILED.value,
                "progress": 0,
                "stage": None,
                "error": "Processing failed",
                "start_time": _processing_status[scene_id]["start_time"],
                "end_time": datetime.datetime.now().isoformat()
            }
            
            logger.error(f"Processing failed for scene {scene_id}")
        
    except Exception as e:
        logger.error(f"Error processing scene {scene_id}: {str(e)}")
        
        # Update processing status to FAILED
        _processing_status[scene_id] = {
            "status": ProcessingStatus.FAILED.value,
            "progress": _processing_status.get(scene_id, {}).get("progress", 0),
            "stage": _processing_status.get(scene_id, {}).get("stage", None),
            "error": str(e),
            "start_time": _processing_status.get(scene_id, {}).get("start_time", None),
            "end_time": datetime.datetime.now().isoformat()
        }


def process_scene(scene_id, options):
    """
    Start processing a scene
    
    Parameters:
    -----------
    scene_id : str
        Scene ID
    options : dict
        Processing options
        
    Returns:
    --------
    dict
        Processing result
    """
    # Check if the scene exists
    raw_dir = Path(Config.DATA_DIR) / "raw" / scene_id
    if not raw_dir.exists() or not raw_dir.is_dir():
        raise Exception(f"Scene {scene_id} not found")
    
    # Check if the scene is already being processed
    status_info = _processing_status.get(scene_id, {})
    current_status = status_info.get("status", ProcessingStatus.IDLE.value)
    
    if current_status == ProcessingStatus.PROCESSING.value:
        raise Exception(f"Scene {scene_id} is already being processed")
    
    # Set initial processing status
    _processing_status[scene_id] = {
        "status": ProcessingStatus.PROCESSING.value,
        "progress": 0,
        "stage": ProcessingStages.EXTRACT.value,
        "error": None,
        "start_time": datetime.datetime.now().isoformat(),
        "end_time": None
    }
    
    # Convert options from frontend naming to backend naming
    processing_options = {
        "extract": True,
        "process_minerals": options.get("minerals", True),
        "process_alteration": options.get("alteration", True),
        "process_gold_pathfinders": options.get("goldPathfinders", True),
        "enhanced_visualization": options.get("enhancedVisualization", True),
        "process_advanced_analysis": options.get("advancedAnalysis", True)
    }
    
    # Start processing in a separate thread
    thread = threading.Thread(
        target=_process_scene_worker,
        args=(scene_id, processing_options)
    )
    thread.daemon = True
    thread.start()
    
    # Store thread reference (for potential future cancellation)
    _processing_threads[scene_id] = thread
    
    logger.info(f"Started processing thread for scene {scene_id}")
    
    return {
        "sceneId": scene_id,
        "status": ProcessingStatus.PROCESSING.value
    }


def get_processing_status(scene_id):
    """
    Get processing status for a scene
    
    Parameters:
    -----------
    scene_id : str
        Scene ID
        
    Returns:
    --------
    dict
        Processing status
    """
    # Check if status exists in memory
    if scene_id in _processing_status:
        return dict(_processing_status[scene_id])
    
    # Check if there's a status file for the scene
    status_path = Path(Config.DATA_DIR) / "raw" / scene_id / "status.json"
    if status_path.exists():
        try:
            with open(status_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading status file for {scene_id}: {str(e)}")
    
    # Return default status if no status found
    return {
        "status": ProcessingStatus.IDLE.value,
        "progress": 0,
        "stage": None,
        "error": None,
        "start_time": None,
        "end_time": None
    }


def get_scene_layers(scene_id):
    """
    Get available layers for a scene
    
    Parameters:
    -----------
    scene_id : str
        Scene ID
        
    Returns:
    --------
    dict
        Available layers categorized by type
    """
    # Check if the scene exists
    processed_dir = Path(Config.DATA_DIR) / "processed" / scene_id
    if not processed_dir.exists() or not processed_dir.is_dir():
        raise Exception(f"Processed scene {scene_id} not found")
    
    # Check if processing is completed
    status = get_processing_status(scene_id)
    if status.get("status") != ProcessingStatus.COMPLETED.value:
        logger.warning(f"Scene {scene_id} processing not completed")
    
    # Get layers by type
    mineral_dir = processed_dir / "minerals"
    alteration_dir = processed_dir / "alteration"
    gold_dir = processed_dir / "gold_pathfinders"
    rgb_dir = processed_dir / "rgb_composites"
    analysis_dir = processed_dir / "analysis"
    
    # Initialize layer collections
    layers = {
        "mineral": [],
        "alteration": [],
        "gold": [],
        "band": [],
        "ratio": [],
        "geological": []
    }
    
    # Find mineral layers
    if mineral_dir.exists():
        mineral_files = list(mineral_dir.glob("*_map.tif"))
        layers["mineral"] = [file.stem.split("_")[0] for file in mineral_files]
    
    # Find alteration layers
    if alteration_dir.exists():
        alteration_files = list(alteration_dir.glob("*_map.tif"))
        layers["alteration"] = [file.stem.split("_")[0] for file in alteration_files]
    
    # Find gold pathfinder layers
    if gold_dir.exists():
        gold_files = list(gold_dir.glob("*_map.tif"))
        layers["gold"] = [file.stem.split("_")[0] for file in gold_files]
    
    # Find band combination layers
    if rgb_dir.exists():
        band_files = list(rgb_dir.glob("*_composite.tif"))
        layers["band"] = [file.stem.split("_")[0] for file in band_files]
    
    # Find band ratio layers
    if analysis_dir.exists():
        ratio_files = list(analysis_dir.glob("*_ratio.tif"))
        layers["ratio"] = [file.stem.split("_")[0] for file in ratio_files]
    
    # Find geological feature layers
    if analysis_dir.exists():
        geological_files = list(analysis_dir.glob("*_features.tif"))
        layers["geological"] = [file.stem.split("_")[0] for file in geological_files]
    
    return layers


def _create_placeholder_tiff(filepath):
    """
    Create a placeholder GeoTIFF file for development/testing
    
    Parameters:
    -----------
    filepath : Path
        Path to save the GeoTIFF
    """
    # Create a simple array
    data = np.random.random((100, 100)).astype(np.float32)
    
    # Create a simple transform
    transform = rasterio.transform.from_bounds(
        west=0,
        south=0,
        east=100,
        north=100,
        width=100,
        height=100
    )
    
    # Write to file
    with rasterio.open(
        filepath,
        'w',
        driver='GTiff',
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs='+proj=latlong',
        transform=transform,
        nodata=np.nan
    ) as dst:
        dst.write(data, 1)




def get_layer_file(scene_id, layer_type, layer_name):
    """
    Get the file path for a specific layer
    
    Parameters:
    -----------
    scene_id : str
        Scene ID
    layer_type : str
        Type of layer
    layer_name : str
        Name of layer
        
    Returns:
    --------
    Path
        Path to the layer file
    """
    # Map layer type to directory and file extension
    type_mapping = {
        "mineral": ("minerals", "_map.tif"),
        "alteration": ("alteration", "_map.tif"),
        "gold": ("gold_pathfinders", "_map.tif"),
        "band": ("rgb_composites", "_composite.tif"),
        "ratio": ("analysis", "_ratio.tif"),
        "geological": ("analysis", "_features.tif")
    }
    
    if layer_type not in type_mapping:
        raise ValueError(f"Invalid layer type: {layer_type}")
    
    dir_name, file_suffix = type_mapping[layer_type]
    
    processed_dir = Path(Config.DATA_DIR) / "processed" / scene_id
    layer_dir = processed_dir / dir_name
    layer_file = layer_dir / f"{layer_name}{file_suffix}"
    
    if not layer_file.exists():
        raise FileNotFoundError(f"Layer file not found: {layer_file}")
    
    return layer_file



def _create_placeholder_thumbnail(filepath):
    """
    Create a placeholder thumbnail image
    
    Parameters:
    -----------
    filepath : Path
        Path to save the thumbnail
    """
    try:
        # Create a simple RGB array
        data = np.random.randint(0, 255, (100, 100, 3)).astype(np.uint8)
        
        # Save as PNG
        import matplotlib.pyplot as plt
        plt.imsave(filepath, data)
    except Exception as e:
        logger.error(f"Error creating thumbnail: {str(e)}")
        # Create an empty file as fallback
        with open(filepath, 'wb') as f:
            f.write(b'')

def get_scene_statistics(scene_id):
    """
    Get statistics for a scene
    
    Parameters:
    -----------
    scene_id : str
        Scene ID
        
    Returns:
    --------
    dict
        Scene statistics
    """
    # Check if the scene exists
    processed_dir = Path(Config.DATA_DIR) / "processed" / scene_id
    if not processed_dir.exists() or not processed_dir.is_dir():
        raise Exception(f"Processed scene {scene_id} not found")
    
    # Get metadata
    metadata_path = processed_dir / "metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        except Exception as e:
            logger.error(f"Error reading metadata for {scene_id}: {str(e)}")
            metadata = {}
    else:
        metadata = {}
    
    # Count mineral maps
    mineral_dir = processed_dir / "minerals"
    mineral_count = len(list(mineral_dir.glob("*_map.tif"))) if mineral_dir.exists() else 0
    
    # Count alteration maps
    alteration_dir = processed_dir / "alteration"
    alteration_count = len(list(alteration_dir.glob("*_map.tif"))) if alteration_dir.exists() else 0
    
    # Count gold pathfinder maps
    gold_dir = processed_dir / "gold_pathfinders"
    pathfinder_count = len(list(gold_dir.glob("*_map.tif"))) if gold_dir.exists() else 0
    
    # Calculate gold pathfinder coverage
    gold_pathfinder_coverage = 0
    if gold_dir.exists() and pathfinder_count > 0:
        # Try to calculate average coverage from first pathfinder map
        for map_file in gold_dir.glob("*_map.tif"):
            try:
                with rasterio.open(map_file) as src:
                    data = src.read(1)
                    valid_mask = ~np.isnan(data)
                    if np.any(valid_mask):
                        coverage = np.sum(data[valid_mask] > 0) / np.sum(valid_mask) * 100
                        gold_pathfinder_coverage = coverage
                        break
            except Exception as e:
                logger.error(f"Error calculating coverage for {map_file}: {str(e)}")
    
    # Determine dominant minerals and alteration
    dominant_minerals = "Unknown"
    dominant_alteration = "Unknown"
    
    if mineral_dir.exists():
        # Try to determine dominant minerals
        minerals = []
        for map_file in mineral_dir.glob("*_map.tif"):
            try:
                mineral_name = map_file.stem.split("_")[0].replace("_", " ").title()
                with rasterio.open(map_file) as src:
                    data = src.read(1)
                    valid_mask = ~np.isnan(data)
                    if np.any(valid_mask):
                        avg_value = np.mean(data[valid_mask])
                        minerals.append((mineral_name, avg_value))
            except Exception as e:
                logger.error(f"Error analyzing {map_file}: {str(e)}")
                
        # Sort minerals by average value
        minerals.sort(key=lambda x: x[1], reverse=True)
        
        # Take top 2 minerals
        if minerals:
            dominant_minerals = ", ".join([m[0] for m in minerals[:2]])
    
    if alteration_dir.exists():
        # Try to determine dominant alteration
        alterations = []
        for map_file in alteration_dir.glob("*_map.tif"):
            try:
                alteration_name = map_file.stem.split("_")[0].replace("_", " ").title()
                with rasterio.open(map_file) as src:
                    data = src.read(1)
                    valid_mask = ~np.isnan(data)
                    if np.any(valid_mask):
                        avg_value = np.mean(data[valid_mask])
                        alterations.append((alteration_name, avg_value))
            except Exception as e:
                logger.error(f"Error analyzing {map_file}: {str(e)}")
                
        # Sort alterations by average value
        alterations.sort(key=lambda x: x[1], reverse=True)
        
        # Take top alteration
        if alterations:
            dominant_alteration = alterations[0][0]
    
    # Create the statistics object
    stats = {
        "acquisitionDate": metadata.get("acquisition_date", "Unknown"),
        "cloudCover": metadata.get("cloud_cover", 0),
        "dominantAlteration": dominant_alteration,
        "dominantMinerals": dominant_minerals,
        "goldPathfinderCoverage": round(gold_pathfinder_coverage, 1),
        "mineralCount": mineral_count,
        "alterationCount": alteration_count,
        "pathfinderCount": pathfinder_count,
        "processingTime": metadata.get("processing_time", "Unknown"),
        "sceneArea": metadata.get("scene_area", "Unknown")
    }
    
    return stats

def generate_prospectivity_map(scene_id, options):
    """
    Generate a prospectivity map
    
    Parameters:
    -----------
    scene_id : str
        Scene ID
    options : dict
        Map generation options
        
    Returns:
    --------
    dict
        Generation result
    """
    # Check if the scene exists
    processed_dir = Path(Config.DATA_DIR) / "processed" / scene_id
    if not processed_dir.exists() or not processed_dir.is_dir():
        raise Exception(f"Processed scene {scene_id} not found")
    
    # Create prospectivity directory if it doesn't exist
    prospectivity_dir = processed_dir / "prospectivity"
    prospectivity_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse options
    threshold = options.get("threshold", 0.7)
    pathfinders = options.get("pathfinders", [])
    alterations = options.get("alterations", [])
    
    logger.info(f"Generating prospectivity map for scene {scene_id} with threshold {threshold}")
    logger.info(f"Pathfinders: {pathfinders}")
    logger.info(f"Alterations: {alterations}")
    
    # Initialize prospectivity mapper
    mapper = GoldProspectivityMapper(output_directory=str(prospectivity_dir))
    
    # Add pathfinder maps
    gold_dir = processed_dir / "gold_pathfinders"
    if gold_dir.exists():
        for pathfinder in pathfinders:
            pathfinder_map = gold_dir / f"{pathfinder}_map.tif"
            if pathfinder_map.exists():
                try:
                    mapper.add_mineral_map(pathfinder, str(pathfinder_map), weight=0.8)
                    logger.info(f"Added pathfinder map: {pathfinder}")
                except Exception as e:
                    logger.error(f"Error adding pathfinder map {pathfinder}: {str(e)}")
    
    # Add alteration maps
    alteration_dir = processed_dir / "alteration"
    if alteration_dir.exists():
        for alteration in alterations:
            alteration_map = alteration_dir / f"{alteration}_map.tif"
            if alteration_map.exists():
                try:
                    mapper.add_mineral_map(alteration, str(alteration_map), weight=0.6)
                    logger.info(f"Added alteration map: {alteration}")
                except Exception as e:
                    logger.error(f"Error adding alteration map {alteration}: {str(e)}")
    
    try:
        # Generate prospectivity map
        prospectivity_map = mapper.generate_prospectivity_map("gold_prospectivity.tif")
        logger.info(f"Generated prospectivity map: {prospectivity_map}")
        
        # Create visualization
        visualization = mapper.visualize_map("gold_prospectivity_visualization.png", 
                                          colormap="prospectivity", 
                                          title="Gold Prospectivity Map")
        logger.info(f"Generated visualization: {visualization}")
        
        # Extract high prospectivity areas
        high_areas = mapper.get_high_prospectivity_areas(threshold, "high_prospectivity.tif")
        logger.info(f"Generated high prospectivity areas: {high_areas}")
        
        # Export to GeoJSON for web visualization
        geojson = mapper.export_to_geojson(threshold, "high_prospectivity.geojson")
        logger.info(f"Exported to GeoJSON: {geojson}")
        
        return {
            "sceneId": scene_id,
            "prospectivityFile": prospectivity_map,
            "visualizationFile": visualization,
            "threshold": threshold,
            "success": True
        }
    except Exception as e:
        logger.error(f"Error generating prospectivity map: {str(e)}")
        return {
            "sceneId": scene_id,
            "success": False,
            "error": str(e)
        }

def generate_report(scene_id, report_type, options):
    """
    Generate an analysis report
    
    Parameters:
    -----------
    scene_id : str
        Scene ID
    report_type : str
        Type of report to generate
    options : dict
        Report generation options
        
    Returns:
    --------
    dict
        Generation result
    """
    # Check if the scene exists
    processed_dir = Path(Config.DATA_DIR) / "processed" / scene_id
    if not processed_dir.exists() or not processed_dir.is_dir():
        raise Exception(f"Processed scene {scene_id} not found")
    
    # Create reports directory if it doesn't exist
    reports_dir = processed_dir / "reports"
    reports_dir.mkdir(exist_ok=True)
    
    # Generate a placeholder report HTML file
    report_file = reports_dir / f"{report_type}_report.html"
    with open(report_file, 'w') as f:
        f.write(f"""
        <!DOCTYPE html>
        <html>
            <head>
                <title>{report_type.capitalize()} Report for Scene {scene_id}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2 {{ color: #2c3e50; }}
                </style>
            </head>
            <body>
                <h1>{report_type.capitalize()} Report for Scene {scene_id}</h1>
                <p>Generated on: {datetime.datetime.now().isoformat()}</p>
                <h2>Analysis Results</h2>
                <p>This is a placeholder report. In a real implementation, this would contain detailed analysis results.</p>
            </body>
        </html>
        """)
    
    # Return the report URL
    return {
        "sceneId": scene_id,
        "reportType": report_type,
        "reportUrl": f"/api/scenes/{scene_id}/reports/{report_type}_report.html"
    }

def export_map(scene_id, map_type, map_name, format="geotiff"):
    """
    Export a map as a downloadable file
    
    Parameters:
    -----------
    scene_id : str
        Scene ID
    map_type : str
        Type of map (minerals, alteration, etc.)
    map_name : str
        Name of the map
    format : str
        Export format
        
    Returns:
    --------
    dict
        Export result
    """
    # Check if the scene exists
    processed_dir = Path(Config.DATA_DIR) / "processed" / scene_id
    if not processed_dir.exists() or not processed_dir.is_dir():
        raise Exception(f"Processed scene {scene_id} not found")
    
    # Map type to directory
    type_dir_map = {
        "mineral": "minerals",
        "alteration": "alteration",
        "gold": "gold_pathfinders",
        "band": "rgb_composites",
        "ratio": "rgb_composites",
        "prospectivity": "prospectivity"
    }
    
    if map_type not in type_dir_map:
        raise Exception(f"Invalid map type: {map_type}")
    
    # Get the directory
    map_dir = processed_dir / type_dir_map[map_type]
    if not map_dir.exists() or not map_dir.is_dir():
        raise Exception(f"Map directory {map_dir} not found")
    
    # Determine file extension based on format
    if format.lower() == "geotiff":
        ext = ".tif"
        mime_type = "image/tiff"
    elif format.lower() == "png":
        ext = ".png"
        mime_type = "image/png"
    else:
        raise Exception(f"Unsupported export format: {format}")
    
    # Look for the file
    if map_type in ["mineral", "alteration", "gold"]:
        file_path = map_dir / f"{map_name}_map{ext}"
    elif map_type in ["band"]:
        file_path = map_dir / f"{map_name}_composite{ext}"
    elif map_type in ["ratio"]:
        file_path = map_dir / f"{map_name}_ratio{ext}"
    elif map_type == "prospectivity":
        file_path = map_dir / f"{map_name}{ext}"
    else:
        file_path = map_dir / f"{map_name}{ext}"
    
    if not file_path.exists():
        raise Exception(f"Map file {file_path} not found")
    
    # Return file info
    return {
        "file_path": str(file_path),
        "file_name": file_path.name,
        "mime_type": mime_type
    }

def get_prospectivity_areas(scene_id, threshold=0.7):
    """
    Get gold prospectivity areas as GeoJSON
    
    Parameters:
    -----------
    scene_id : str
        Scene ID
    threshold : float
        Threshold value for prospectivity
        
    Returns:
    --------
    dict
        GeoJSON of prospectivity areas
    """
    # Check if the scene exists
    processed_dir = Path(Config.DATA_DIR) / "processed" / scene_id
    if not processed_dir.exists() or not processed_dir.is_dir():
        raise Exception(f"Processed scene {scene_id} not found")
    
    # Check if prospectivity data exists
    prospectivity_dir = processed_dir / "prospectivity"
    geojson_file = prospectivity_dir / "high_prospectivity.geojson"
    
    if not prospectivity_dir.exists() or not geojson_file.exists():
        # No prospectivity data found, try to generate it
        options = {
            "threshold": threshold,
            "pathfinders": ["gold_alteration", "pyrite", "arsenopyrite"],
            "alterations": ["advanced_argillic", "phyllic", "silicification"]
        }
        generate_prospectivity_map(scene_id, options)
    
    # Check if file exists after generation attempt
    if geojson_file.exists():
        try:
            with open(geojson_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading GeoJSON file: {str(e)}")
    
    # If we couldn't generate or read the file, return a placeholder result
    logger.warning(f"Returning placeholder prospectivity areas for scene {scene_id}")
    
    # Placeholder GeoJSON features
    features = {}
    
    # High prospectivity (above threshold)
    high_features = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-2.3, 9.1],
                            [-2.3, 9.2],
                            [-2.2, 9.2],
                            [-2.2, 9.1],
                            [-2.3, 9.1]
                        ]
                    ]
                },
                "properties": {
                    "category": "high",
                    "value": 0.85,
                    "confidence": 0.9
                }
            }
        ]
    }
    
    # Medium prospectivity (between 0.5 and threshold)
    medium_features = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-2.4, 9.0],
                            [-2.4, 9.1],
                            [-2.3, 9.1],
                            [-2.3, 9.0],
                            [-2.4, 9.0]
                        ]
                    ]
                },
                "properties": {
                    "category": "medium",
                    "value": 0.65,
                    "confidence": 0.8
                }
            }
        ]
    }
    
    # Low prospectivity (below 0.5)
    low_features = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-2.5, 8.9],
                            [-2.5, 9.0],
                            [-2.4, 9.0],
                            [-2.4, 8.9],
                            [-2.5, 8.9]
                        ]
                    ]
                },
                "properties": {
                    "category": "low",
                    "value": 0.35,
                    "confidence": 0.7
                }
            }
        ]
    }
    
    features = {
        "high": high_features,
        "medium": medium_features,
        "low": low_features
    }
    
    return features
    
def getLayerUrl(sceneId, layerType, layerName):
    """
    Get a specific layer for a scene
    
    Parameters:
    -----------
    sceneId : str
        Scene ID
    layerType : str
        Type of layer (minerals, alteration, etc.)
    layerName : str
        Name of the layer
        
    Returns:
    --------
    str
        URL to the layer
    """
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
    
    if layerType not in type_dir_map:
        raise ValueError(f"Invalid layer type: {layerType}")
    
    # Determine the directory and file path
    processed_dir = Path(Config.DATA_DIR) / "processed" / sceneId
    layer_dir = processed_dir / type_dir_map[layerType]
    layer_file = layer_dir / f"{layerName}{type_suffix_map.get(layerType, '.tif')}"
    
    # Check if the file exists
    if not layer_file.exists():
        raise FileNotFoundError(f"Layer file not found: {layer_file}")
    
    # Return the URL
    relative_path = f"scenes/{sceneId}/layers/{layerType}/{layerName}"
    url = f"{Config.SERVER_URL}/api/{relative_path}"
    
    return url