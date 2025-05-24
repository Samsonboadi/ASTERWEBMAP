"""
Controller functions for the ASTER Web Explorer backend
"""

import os
import json
import logging
import datetime
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
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

# Thread pool executor for managing background tasks (Recommendation: Better thread management)
executor = ThreadPoolExecutor(max_workers=4)

def get_all_scenes():
    """
    Get all available scenes, only including those that have completed processing.

    Returns:
    --------
    list
        List of scene objects
    """
    data_dir = Path(Config.DATA_DIR) / "processed"
    data_dir.mkdir(parents=True, exist_ok=True)
    scene_dirs = [d for d in data_dir.glob("*") if d.is_dir()]
    
    logger.info(f"Found {len(scene_dirs)} scene directories in {data_dir}")
    
    scenes = []
    for scene_dir in scene_dirs:
        metadata_path = scene_dir / "metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Get processing status
                status = _processing_status.get(scene_dir.name, {}).get("status", ProcessingStatus.IDLE.value)
                
                # Only include completed scenes (Recommendation: Handle in-progress scenes)
                if status == ProcessingStatus.COMPLETED.value:
                    scenes.append({
                        "id": scene_dir.name,
                        "name": metadata.get("name", scene_dir.name),
                        "date": metadata.get("acquisition_date", metadata.get("upload_date", "Unknown")),
                        "cloudCover": metadata.get("cloud_cover", 0),
                        "status": status
                    })
                    logger.info(f"Added completed scene {scene_dir.name} to list")
                else:
                    logger.info(f"Skipping scene {scene_dir.name} as it is not completed (status: {status})")
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
                logger.info(f"Added scene {scene_dir.name} with minimal info due to error")
        else:
            logger.warning(f"No metadata found for {scene_dir.name}, skipping")
    
    logger.info(f"Returning {len(scenes)} scenes")
    return scenes

def get_scene_by_id(scene_id):
    """
    Get scene details by ID.

    Parameters:
    -----------
    scene_id : str
        Scene ID
        
    Returns:
    --------
    dict
        Scene object or None if not found
    """
    metadata_path = Path(Config.DATA_DIR) / "processed" / scene_id / "metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                
            status_info = _processing_status.get(scene_id, {})
            status = status_info.get("status", ProcessingStatus.IDLE.value)
            
            bounds = metadata.get("bounds", None)
            
            thumbnail_path = Path(Config.DATA_DIR) / "processed" / scene_id / "thumbnail.png"
            has_thumbnail = thumbnail_path.exists()
            
            scene = {
                "id": scene_id,
                "name": metadata.get("name", scene_id),
                "date": metadata.get("acquisition_date", "Unknown"),
                "cloudCover": metadata.get("cloud_cover", 0),
                "status": status,
                "bounds": bounds,
                "thumbnail": f"/api/scenes/{scene_id}/thumbnail" if has_thumbnail else None,
                "processingMode": metadata.get("processing_mode", "Full")  # Added for VNIR-only mode feedback
            }
            
            return scene
        except Exception as e:
            logger.error(f"Error reading metadata for {scene_id}: {str(e)}")
            return None
    else:
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
    Upload a new scene file without automatically starting processing.

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
    scene_id = f"scene_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    scene_dir = Path(Config.DATA_DIR) / "raw" / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)
    
    filename = secure_filename(file.filename)
    file_path = scene_dir / filename
    file.save(str(file_path))
    
    if metadata is None:
        metadata = {}
    
    metadata.update({
        "original_filename": filename,
        "upload_date": datetime.datetime.now().isoformat(),
        "file_size": os.path.getsize(file_path),
        "name": metadata.get("name", filename)
    })
    
    with open(scene_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    _processing_status[scene_id] = {
        "status": ProcessingStatus.IDLE.value,
        "progress": 0,
        "stage": None,
        "error": None,
        "start_time": None,
        "end_time": None
    }
    
    processed_dir = Path(Config.DATA_DIR) / "processed" / scene_id
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    with open(processed_dir / "metadata.json", 'w') as f:
        metadata.update({
            "scene_id": scene_id,
            "bounds": {
                "west": -2.67,
                "east": -1.99,
                "south": 8.76,
                "north": 9.43
            },
            "cloud_cover": 0,
            "acquisition_date": datetime.datetime.now().strftime('%Y-%m-%d')
        })
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Uploaded new scene {scene_id}: {filename}")
    
    return {
        "sceneId": scene_id,
        "filename": filename,
        "status": "success",
        "processingStarted": False  # Let the frontend initiate processing
    }

def _process_scene_worker(scene_id, options):
    """
    Worker function for processing a scene in the background.

    Parameters:
    -----------
    scene_id : str
        Scene ID
    options : dict
        Processing options
    """
    try:
        logger.info(f"Started processing scene {scene_id} with options: {options}")
        
        processor = ASTERProcessor(scene_id, Config.DATA_DIR)
        
        vnir_only_mode = options.get('vnir_only_mode', False)
        
        processor.update_status(
            ProcessingStatus.PROCESSING, 
            5, 
            ProcessingStages.EXTRACT,
            "Extracting data"
        )
        
        processor.extract_data()
        
        if vnir_only_mode:
            vnir_file = None
            for file_path in processor.extracted_dir.glob('**/*.hdf'):
                if 'vnir' in file_path.name.lower() or 'AST_09XT' in file_path.name.upper():
                    vnir_file = file_path
                    break
            
            if vnir_file is None:
                hdf_files = list(processor.extracted_dir.glob('**/*.hdf'))
                if hdf_files:
                    vnir_file = hdf_files[0]
            
            if vnir_file:
                logger.info(f"VNIR-only mode: Using {vnir_file} for both VNIR and SWIR inputs")
                
                from processors.vnir_processor import VNIRProcessor
                from processors.aster_l2_processor import SceneMetadata, GeographicBounds
                scene_metadata = SceneMetadata(
                    bounds=GeographicBounds(
                        west=-2.67, east=-1.99, south=8.76, north=9.43
                    ),
                    solar_azimuth=152.9561090000,
                    solar_elevation=53.5193190000,
                    cloud_cover=5.0,
                    acquisition_date="2000-12-18",
                    utm_zone=30
                )
                
                vnir_processor = VNIRProcessor(
                    vnir_file=str(vnir_file),
                    metadata=scene_metadata
                )
                
                rgb_dir = processor.processed_dir / 'rgb_composites'
                rgb_dir.mkdir(exist_ok=True)
                
                vnir_processor.create_true_color_composite(rgb_dir / "true_color_composite.tif")
                vnir_processor.create_false_color_composite(rgb_dir / "false_color_composite.tif")
                
                minerals_dir = processor.processed_dir / 'minerals'
                minerals_dir.mkdir(exist_ok=True)
                
                if options.get('process_minerals', False):  # Recommendation: Respect mineral processing option
                    logger.info("Generating VNIR-specific mineral outputs")
                    vnir_processor.save_band_as_geotiff(1, minerals_dir / "band1_blue.tif")
                    vnir_processor.save_band_as_geotiff(2, minerals_dir / "band2_green.tif")
                    vnir_processor.save_band_as_geotiff(3, minerals_dir / "band3_nir.tif")
                    vnir_processor.create_ndvi_map(minerals_dir / "ndvi_map.tif")
                else:
                    logger.info("Skipping VNIR mineral processing because process_minerals is False")
                
                metadata_file = processor.raw_dir / 'metadata.json'
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    with open(processor.processed_dir / 'metadata.json', 'w') as f:
                        json.dump({
                            'scene_id': scene_id,
                            'name': metadata.get('original_filename', scene_id),
                            'date': metadata.get('upload_date', datetime.datetime.now().isoformat()),
                            'bounds': {
                                'west': scene_metadata.bounds.west,
                                'east': scene_metadata.bounds.east,
                                'south': scene_metadata.bounds.south,
                                'north': scene_metadata.bounds.north
                            },
                            'cloud_cover': scene_metadata.cloud_cover,
                            'solar_azimuth': scene_metadata.solar_azimuth,
                            'solar_elevation': scene_metadata.solar_elevation,
                            'acquisition_date': scene_metadata.acquisition_date,
                            'processing_mode': 'VNIR-only'
                        }, f, indent=2)
                
                processor.update_status(
                    ProcessingStatus.COMPLETED, 
                    100, 
                    None,
                    "VNIR-only processing completed"
                )
                
                logger.info(f"VNIR-only processing completed for scene {scene_id}")
                return True
            else:
                logger.error(f"No HDF files found in the extracted directory for scene {scene_id}")
                processor.update_status(
                    ProcessingStatus.FAILED, 
                    0, 
                    None,
                    "No HDF files found in the extracted directory"
                )
                return False
        else:
            success = processor.process(options)
            
            if success:
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
                _processing_status[scene_id] = {
                    "status": ProcessingStatus.FAILED.value,
                    "progress": 0,
                    "stage": None,
                    "error": "Processing failed",
                    "start_time": _processing_status[scene_id]["start_time"],
                    "end_time": datetime.datetime.now().isoformat()
                }
                logger.error(f"Processing failed for scene {scene_id}")
            
            return success
                
    except Exception as e:
        logger.error(f"Error processing scene {scene_id}: {str(e)}")
        
        _processing_status[scene_id] = {
            "status": ProcessingStatus.FAILED.value,
            "progress": _processing_status.get(scene_id, {}).get("progress", 0),
            "stage": _processing_status.get(scene_id, {}).get("stage", None),
            "error": str(e),
            "start_time": _processing_status.get(scene_id, {}).get("start_time", None),
            "end_time": datetime.datetime.now().isoformat()
        }
        
        return False

def process_scene(scene_id, options):
    """
    Start processing a scene with the provided options.

    Parameters:
    -----------
    scene_id : str
        Scene ID
    options : dict
        Processing options from the frontend, e.g., 
        {'minerals': True, 'alteration': True, 'goldPathfinders': True, 'enhancedVisualization': True}

    Returns:
    --------
    dict
        Processing result with scene ID, status, and options used
    """
    # Check if the scene exists
    raw_dir = Path(Config.DATA_DIR) / "raw" / scene_id
    if not raw_dir.exists() or not raw_dir.is_dir():
        logger.error(f"Scene {scene_id} not found in {raw_dir}")
        raise Exception(f"Scene {scene_id} not found")

    # Check if the scene is already being processed
    status_info = _processing_status.get(scene_id, {})
    current_status = status_info.get("status", ProcessingStatus.IDLE.value)

    if current_status == ProcessingStatus.PROCESSING.value:
        logger.warning(f"Scene {scene_id} is already being processed. Stopping current process and restarting.")
        if scene_id in _processing_threads:
            logger.info(f"Removing reference to previous processing thread for {scene_id}")
            _processing_threads.pop(scene_id, None)

    # Log the raw options received from the frontend
    logger.info(f"Received raw options from frontend for scene {scene_id}: {options}")

    # Convert frontend options to backend processing options
    processing_options = {
        "extract": options.get("extract", True),
        "process_minerals": options.get("minerals", True),
        "process_alteration": options.get("alteration", True),
        "process_gold_pathfinders": options.get("goldPathfinders", True),
        "enhanced_visualization": options.get("enhancedVisualization", True),
        "process_advanced_analysis": options.get("process_advanced_analysis", False),
        "vnir_only_mode": options.get("vnir_only_mode", False)
    }

    # Log the converted options to ensure they are correct
    logger.info(f"Converted processing options for scene {scene_id}: {processing_options}")

    # Set initial processing status
    _processing_status[scene_id] = {
        "status": ProcessingStatus.PROCESSING.value,
        "progress": 0,
        "stage": ProcessingStages.EXTRACT.value,
        "error": None,
        "start_time": datetime.datetime.now().isoformat(),
        "end_time": None
    }

    # Start processing in a thread pool executor (Recommendation: Better thread management)
    future = executor.submit(_process_scene_worker, scene_id, processing_options)
    _processing_threads[scene_id] = future

    logger.info(f"Started processing task for scene {scene_id} with final options: {processing_options}")

    return {
        "sceneId": scene_id,
        "status": ProcessingStatus.PROCESSING.value,
        "options": processing_options
    }

def get_processing_status(scene_id):
    """
    Get processing status for a scene.

    Parameters:
    -----------
    scene_id : str
        Scene ID
        
    Returns:
    --------
    dict
        Processing status
    """
    if scene_id in _processing_status:
        return dict(_processing_status[scene_id])
    
    status_path = Path(Config.DATA_DIR) / "raw" / scene_id / "status.json"
    if status_path.exists():
        try:
            with open(status_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading status file for {scene_id}: {str(e)}")
    
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
    Get available layers for a scene with better validation and filtering.

    Parameters:
    -----------
    scene_id : str
        Scene ID
        
    Returns:
    --------
    dict
        Available layers categorized by type, only including layers that actually exist
    """
    processed_dir = Path(Config.DATA_DIR) / "processed" / scene_id
    if not processed_dir.exists() or not processed_dir.is_dir():
        raise Exception(f"Processed scene {scene_id} not found")
    
    status = get_processing_status(scene_id)
    if status.get("status") != ProcessingStatus.COMPLETED.value:
        logger.warning(f"Scene {scene_id} processing not completed, available layers may be limited")
    
    # Check metadata to determine processing mode
    metadata_path = processed_dir / "metadata.json"
    processing_mode = "Full"
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                processing_mode = metadata.get("processing_mode", "Full")
        except Exception as e:
            logger.error(f"Error reading metadata for {scene_id}: {str(e)}")
    
    # Define directory mappings
    mineral_dir = processed_dir / "minerals"
    alteration_dir = processed_dir / "alteration"
    gold_dir = processed_dir / "gold_pathfinders"
    rgb_dir = processed_dir / "rgb_composites"
    analysis_dir = processed_dir / "analysis"
    
    layers = {
        "mineral": [],
        "alteration": [],
        "gold": [],
        "band": [],
        "ratio": [],
        "geological": [],
        "vnir": []  # Add VNIR category for VNIR-only mode
    }
    
    # Helper function to validate and add layers
    def add_layer_if_exists(layer_list, layer_name, file_path):
        if file_path.exists() and file_path.is_file():
            layer_list.append(layer_name)
            logger.debug(f"Added layer: {layer_name} -> {file_path}")
            return True
        else:
            logger.debug(f"Skipped missing layer: {layer_name} -> {file_path}")
            return False
    
    # Check mineral layers with validation
    if mineral_dir.exists():
        mineral_files = list(mineral_dir.glob("*_map.tif"))
        for file in mineral_files:
            layer_name = file.stem.replace("_map", "")
            add_layer_if_exists(layers["mineral"], layer_name, file)
        
        logger.info(f"Found mineral layers: {layers['mineral']}")
        
        # For VNIR-only mode, also check for VNIR-specific products
        if processing_mode == "VNIR-only":
            vnir_products = {
                "ndvi": "ndvi_map.tif",
                "band1": "band1_blue_map.tif", 
                "band2": "band2_green_map.tif",
                "band3": "band3_nir_map.tif"
            }
            
            for simple_name, file_name in vnir_products.items():
                vnir_file = mineral_dir / file_name
                add_layer_if_exists(layers["vnir"], simple_name, vnir_file)
            
            logger.info(f"Found VNIR layers: {layers['vnir']}")
    
    # Check alteration layers (not available in VNIR-only mode)
    if alteration_dir.exists() and processing_mode != "VNIR-only":
        alteration_files = list(alteration_dir.glob("*_map.tif"))
        for file in alteration_files:
            layer_name = file.stem.replace("_map", "")
            add_layer_if_exists(layers["alteration"], layer_name, file)
        
        logger.info(f"Found alteration layers: {layers['alteration']}")
    
    # Check gold pathfinder layers with validation
    if gold_dir.exists():
        gold_files = list(gold_dir.glob("*_map.tif"))
        available_gold = []
        
        for file in gold_files:
            layer_name = file.stem.replace("_map", "")
            if add_layer_if_exists(available_gold, layer_name, file):
                pass  # Layer was added by add_layer_if_exists
        
        # In VNIR-only mode, only show VNIR-compatible pathfinders that actually exist
        if processing_mode == "VNIR-only":
            vnir_compatible_gold = ["pyrite", "arsenopyrite"]  # These use VNIR bands
            layers["gold"] = [g for g in available_gold if g in vnir_compatible_gold]
        else:
            layers["gold"] = available_gold
        
        logger.info(f"Found gold pathfinder layers: {layers['gold']}")
    
    # Check band combination layers with validation
    if rgb_dir.exists():
        band_files = list(rgb_dir.glob("*_composite.tif"))
        for file in band_files:
            layer_name = file.stem.replace("_composite", "")
            add_layer_if_exists(layers["band"], layer_name, file)
        
        logger.info(f"Found band combination layers: {layers['band']}")
    
    # Check ratio layers with validation
    if analysis_dir.exists():
        ratio_files = list(analysis_dir.glob("*_ratio.tif"))
        for file in ratio_files:
            layer_name = file.stem.replace("_ratio", "")
            add_layer_if_exists(layers["ratio"], layer_name, file)
        
        logger.info(f"Found ratio layers: {layers['ratio']}")
    
    # Check geological layers with validation
    if analysis_dir.exists():
        geological_files = list(analysis_dir.glob("*_features.tif"))
        for file in geological_files:
            layer_name = file.stem.replace("_features", "")
            add_layer_if_exists(layers["geological"], layer_name, file)
        
        logger.info(f"Found geological layers: {layers['geological']}")
    
    # Add processing mode info to response
    layers["_metadata"] = {
        "processing_mode": processing_mode,
        "vnir_only": processing_mode == "VNIR-only",
        "total_layers": sum(len(layer_list) for key, layer_list in layers.items() if key != "_metadata")
    }
    
    logger.info(f"Returning {layers['_metadata']['total_layers']} validated layers for scene {scene_id}")
    return layers

def validate_layer_exists(scene_id, layer_type, layer_name):
    """
    Validate that a specific layer exists for a scene.
    
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
    bool
        True if layer exists, False otherwise
    """
    try:
        processed_dir = Path(Config.DATA_DIR) / "processed" / scene_id
        
        # Map layer types to directory and file patterns
        type_mapping = {
            "mineral": ("minerals", "_map.tif"),
            "alteration": ("alteration", "_map.tif"),
            "geological": ("analysis", "_features.tif"),
            "band": ("rgb_composites", "_composite.tif"),
            "ratio": ("analysis", "_ratio.tif"),
            "gold": ("gold_pathfinders", "_map.tif"),
            "vnir": ("minerals", "_map.tif")
        }
        
        if layer_type not in type_mapping:
            return False
        
        dir_name, file_suffix = type_mapping[layer_type]
        
        # Special handling for VNIR layers
        if layer_type == "vnir":
            vnir_mapping = {
                'ndvi': 'ndvi',
                'band1': 'band1_blue',
                'band2': 'band2_green',
                'band3': 'band3_nir'
            }
            layer_name = vnir_mapping.get(layer_name, layer_name)
        
        layer_dir = processed_dir / dir_name
        layer_file = layer_dir / f"{layer_name}{file_suffix}"
        
        exists = layer_file.exists()
        logger.info(f"Layer validation for {scene_id}/{layer_type}/{layer_name}: {exists} (path: {layer_file})")
        return exists
        
    except Exception as e:
        logger.error(f"Error validating layer {scene_id}/{layer_type}/{layer_name}: {str(e)}")
        return False
        

def _create_placeholder_tiff(filepath):
    """
    Create a placeholder GeoTIFF file for development/testing.

    Parameters:
    -----------
    filepath : Path
        Path to save the GeoTIFF
    """
    data = np.random.random((100, 100)).astype(np.float32)
    
    transform = rasterio.transform.from_bounds(
        west=0,
        south=0,
        east=100,
        north=100,
        width=100,
        height=100
    )
    
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
    Get the file path for a specific layer.

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
    Create a placeholder thumbnail image.

    Parameters:
    -----------
    filepath : Path
        Path to save the thumbnail
    """
    try:
        data = np.random.randint(0, 255, (100, 100, 3)).astype(np.uint8)
        import matplotlib.pyplot as plt
        plt.imsave(filepath, data)
    except Exception as e:
        logger.error(f"Error creating thumbnail: {str(e)}")
        with open(filepath, 'wb') as f:
            f.write(b'')

def get_scene_statistics(scene_id):
    """
    Get statistics for a scene.

    Parameters:
    -----------
    scene_id : str
        Scene ID
        
    Returns:
    --------
    dict
        Scene statistics
    """
    processed_dir = Path(Config.DATA_DIR) / "processed" / scene_id
    if not processed_dir.exists() or not processed_dir.is_dir():
        raise Exception(f"Processed scene {scene_id} not found")
    
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
    
    mineral_dir = processed_dir / "minerals"
    alteration_dir = processed_dir / "alteration"
    gold_dir = processed_dir / "gold_pathfinders"
    
    mineral_count = len(list(mineral_dir.glob("*_map.tif"))) if mineral_dir.exists() else 0
    alteration_count = len(list(alteration_dir.glob("*_map.tif"))) if alteration_dir.exists() else 0
    pathfinder_count = len(list(gold_dir.glob("*_map.tif"))) if gold_dir.exists() else 0
    
    gold_pathfinder_coverage = 0
    if gold_dir.exists() and pathfinder_count > 0:
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
    
    dominant_minerals = "Unknown"
    dominant_alteration = "Unknown"
    
    if mineral_dir.exists():
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
                
        minerals.sort(key=lambda x: x[1], reverse=True)
        if minerals:
            dominant_minerals = ", ".join([m[0] for m in minerals[:2]])
    
    if alteration_dir.exists():
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
                
        alterations.sort(key=lambda x: x[1], reverse=True)
        if alterations:
            dominant_alteration = alterations[0][0]
    
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
    Generate a prospectivity map.

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
    processed_dir = Path(Config.DATA_DIR) / "processed" / scene_id
    if not processed_dir.exists() or not processed_dir.is_dir():
        raise Exception(f"Processed scene {scene_id} not found")
    
    prospectivity_dir = processed_dir / "prospectivity"
    prospectivity_dir.mkdir(parents=True, exist_ok=True)
    
    threshold = options.get("threshold", 0.7)
    pathfinders = options.get("pathfinders", [])
    alterations = options.get("alterations", [])
    
    logger.info(f"Generating prospectivity map for scene {scene_id} with threshold {threshold}")
    logger.info(f"Pathfinders: {pathfinders}")
    logger.info(f"Alterations: {alterations}")
    
    mapper = GoldProspectivityMapper(output_directory=str(prospectivity_dir))
    
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
        prospectivity_map = mapper.generate_prospectivity_map("gold_prospectivity.tif")
        logger.info(f"Generated prospectivity map: {prospectivity_map}")
        
        visualization = mapper.visualize_map("gold_prospectivity_visualization.png", 
                                          colormap="prospectivity", 
                                          title="Gold Prospectivity Map")
        logger.info(f"Generated visualization: {visualization}")
        
        high_areas = mapper.get_high_prospectivity_areas(threshold, "high_prospectivity.tif")
        logger.info(f"Generated high prospectivity areas: {high_areas}")
        
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
    Generate an analysis report.

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
    processed_dir = Path(Config.DATA_DIR) / "processed" / scene_id
    if not processed_dir.exists() or not processed_dir.is_dir():
        raise Exception(f"Processed scene {scene_id} not found")
    
    reports_dir = processed_dir / "reports"
    reports_dir.mkdir(exist_ok=True)
    
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
    
    return {
        "sceneId": scene_id,
        "reportType": report_type,
        "reportUrl": f"/api/scenes/{scene_id}/reports/{report_type}_report.html"
    }

def export_map(scene_id, map_type, map_name, format="geotiff"):
    """
    Export a map as a downloadable file.

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
    processed_dir = Path(Config.DATA_DIR) / "processed" / scene_id
    if not processed_dir.exists() or not processed_dir.is_dir():
        raise Exception(f"Processed scene {scene_id} not found")
    
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
    
    map_dir = processed_dir / type_dir_map[map_type]
    if not map_dir.exists() or not map_dir.is_dir():
        raise Exception(f"Map directory {map_dir} not found")
    
    if format.lower() == "geotiff":
        ext = ".tif"
        mime_type = "image/tiff"
    elif format.lower() == "png":
        ext = ".png"
        mime_type = "image/png"
    else:
        raise Exception(f"Unsupported export format: {format}")
    
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
    
    return {
        "file_path": str(file_path),
        "file_name": file_path.name,
        "mime_type": mime_type
    }

def get_prospectivity_areas(scene_id, threshold=0.7):
    """
    Get gold prospectivity areas as GeoJSON.

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
    processed_dir = Path(Config.DATA_DIR) / "processed" / scene_id
    if not processed_dir.exists() or not processed_dir.is_dir():
        raise Exception(f"Processed scene {scene_id} not found")
    
    prospectivity_dir = processed_dir / "prospectivity"
    geojson_file = prospectivity_dir / "high_prospectivity.geojson"
    
    if not prospectivity_dir.exists() or not geojson_file.exists():
        options = {
            "threshold": threshold,
            "pathfinders": ["gold_alteration", "pyrite", "arsenopyrite"],
            "alterations": ["advanced_argillic", "phyllic", "silicification"]
        }
        generate_prospectivity_map(scene_id, options)
    
    if geojson_file.exists():
        try:
            with open(geojson_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading GeoJSON file: {str(e)}")
    
    logger.warning(f"Returning placeholder prospectivity areas for scene {scene_id}")
    
    features = {}
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
    Get a specific layer for a scene.

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
    type_dir_map = {
        "mineral": "minerals",
        "alteration": "alteration",
        "geological": "analysis",
        "band": "rgb_composites",
        "ratio": "analysis",
        "gold": "gold_pathfinders"
    }
    
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
    
    processed_dir = Path(Config.DATA_DIR) / "processed" / sceneId
    layer_dir = processed_dir / type_dir_map[layerType]
    layer_file = layer_dir / f"{layerName}{type_suffix_map.get(layerType, '.tif')}"
    
    if not layer_file.exists():
        raise FileNotFoundError(f"Layer file not found: {layer_file}")
    
    relative_path = f"scenes/{sceneId}/layers/{layerType}/{layerName}"
    url = f"{Config.SERVER_URL}/api/{relative_path}"
    
    return url