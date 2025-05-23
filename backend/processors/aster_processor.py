"""
ASTER Processor Module
Primary integration point for ASTER data processing
"""

import os
# Set environment variable to avoid CPU count warning
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
import traceback
import re
import zipfile
import numpy as np
from h5py import Dataset

from enums import ProcessingStatus, ProcessingStages

logger = logging.getLogger(__name__)

# Configure logging to include file output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('aster_processing.log')
    ]
)

class ASTERProcessor:
    """
    Main processor class for integrating all ASTER processing functionality.
    This class coordinates between the various specialized processors.
    """
    
    def __init__(self, scene_id: str, data_dir: str, config: Optional[Dict] = None):
        """
        Initialize the ASTER processor
        
        Parameters:
        -----------
        scene_id : str
            ID of the scene to process
        data_dir : str
            Base directory for data
        config : Dict, optional
            Configuration options
        """
        self.scene_id = scene_id
        self.data_dir = Path(data_dir)
        self.config = config or {}
        
        # Define directory paths
        self.raw_dir = self.data_dir / 'raw' / scene_id
        self.extracted_dir = self.raw_dir / 'extracted'
        self.processed_dir = self.data_dir / 'processed' / scene_id
        self.log_dir = self.data_dir / 'logs'
        
        # Ensure directories exist
        for dir_path in [self.raw_dir, self.extracted_dir, self.processed_dir, self.log_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Setup scene-specific logger
        self.scene_logger = self._setup_scene_logger()
        
        logger.info(f"Initialized ASTER processor for scene {scene_id}")
        self.scene_logger.info(f"Configuration: {json.dumps(self.config, indent=2)}")
    
    def _setup_scene_logger(self):
        """Set up scene-specific logger"""
        log_file = self.log_dir / f"{self.scene_id}_processing.log"
        scene_logger = logging.getLogger(f"scene_{self.scene_id}")
        scene_logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplication
        for handler in scene_logger.handlers[:]:
            scene_logger.removeHandler(handler)
        
        # Add file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        scene_logger.addHandler(file_handler)
        
        return scene_logger
    
    def get_status(self) -> Dict:
        """
        Get the current processing status
        
        Returns:
        --------
        Dict
            Processing status information
        """
        status_file = self.raw_dir / 'status.json'
        
        if status_file.exists():
            with open(status_file, 'r') as f:
                return json.load(f)
        else:
            return {
                'status': ProcessingStatus.IDLE.value,
                'progress': 0,
                'stage': None,
                'last_updated': datetime.now().isoformat()
            }
    
    def update_status(self, status: ProcessingStatus, progress: int, 
                     stage: Optional[ProcessingStages] = None, message: Optional[str] = None) -> None:
        """
        Update the processing status
        
        Parameters:
        -----------
        status : ProcessingStatus
            Current processing status
        progress : int
            Progress percentage (0-100)
        stage : ProcessingStages, optional
            Current processing stage
        message : str, optional
            Status message
        """
        status_file = self.raw_dir / 'status.json'
        
        status_data = {
            'status': status.value if hasattr(status, 'value') else status,
            'progress': progress,
            'last_updated': datetime.now().isoformat()
        }
        
        if stage:
            status_data['stage'] = stage.value if hasattr(stage, 'value') else stage
        
        if message:
            status_data['message'] = message
        
        with open(status_file, 'w') as f:
            json.dump(status_data, f, indent=2)
            
        logger.info(f"Updated status for scene {self.scene_id}: {status} ({progress}%)")
        self.scene_logger.info(f"Status: {status} ({progress}%): {message or 'No message'}")
    
    def extract_data(self) -> bool:
        """
        Extract ASTER data from zip/archive files
        
        Returns:
        --------
        bool
            True if extraction was successful
        """
        zip_files = list(self.raw_dir.glob('*.zip'))
        
        if not zip_files:
            logger.info(f"No zip files found in {self.raw_dir}")
            self.scene_logger.info(f"No zip files found in {self.raw_dir}")
            return False
        
        try:
            for zip_file in zip_files:
                zip_subdir = self.extracted_dir / zip_file.stem
                if zip_subdir.exists():
                    logger.info(f"Skipping {zip_file.name} - directory {zip_subdir} already exists")
                    self.scene_logger.info(f"Skipping {zip_file.name} - directory {zip_subdir} already exists")
                    continue
                
                logger.info(f"Extracting {zip_file}")
                self.scene_logger.info(f"Extracting {zip_file}")
                
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    file_list = zip_ref.namelist()
                    logger.info(f"Files in zip: {file_list}")
                    self.scene_logger.info(f"Files in zip: {file_list}")
                    
                    zip_ref.extractall(zip_subdir)
                    logger.info(f"Extracted {len(file_list)} files from {zip_file}")
                    self.scene_logger.info(f"Extracted {len(file_list)} files from {zip_file}")
            
            extracted_files = list(self.extracted_dir.glob('**/*.hdf'))
            logger.info(f"Extracted {len(extracted_files)} HDF files")
            self.scene_logger.info(f"Extracted {len(extracted_files)} HDF files")
            
            if len(extracted_files) < 2:
                logger.warning("Expected at least 2 HDF files (VNIR and SWIR), but found fewer")
                self.scene_logger.warning("Expected at least 2 HDF files (VNIR and SWIR), but found fewer")
                
            met_files = list(self.extracted_dir.glob('**/*.met'))
            logger.info(f"Extracted {len(met_files)} metadata files")
            self.scene_logger.info(f"Extracted {len(met_files)} metadata files")
            
            return True
        except Exception as e:
            logger.error(f"Error extracting data: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            self.scene_logger.error(f"Error extracting data: {str(e)}")
            self.scene_logger.error(f"Stack trace: {traceback.format_exc()}")
            return False
    
    def find_aster_files(self) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Find matching VNIR and SWIR files based on timestamp and file size.
        
        Returns:
        --------
        Tuple[Optional[Path], Optional[Path]]
            Paths to VNIR and SWIR files
        """
        logger.info(f"Looking for HDF files in: {self.extracted_dir}")
        self.scene_logger.info(f"Looking for HDF files in: {self.extracted_dir}")
        
        files = list(self.extracted_dir.glob('**/*.hdf'))
        logger.info(f"Found {len(files)} HDF files")
        self.scene_logger.info(f"Found {len(files)} HDF files")
        
        if not files:
            logger.warning("No HDF files found")
            self.scene_logger.warning("No HDF files found")
            return None, None
        
        files_by_time = {}
        for file in files:
            parts = file.name.split('_')
            if len(parts) >= 3:
                timestamp = parts[2]  # Use third part as timestamp (adjust as needed)
                if timestamp not in files_by_time:
                    files_by_time[timestamp] = []
                files_by_time[timestamp].append(file)
                logger.info(f"Grouped file {file.name} under timestamp {timestamp}")
                self.scene_logger.info(f"Grouped file {file.name} under timestamp {timestamp}")
        
        vnir_file, swir_file = None, None
        for timestamp, group_files in files_by_time.items():
            if len(group_files) >= 2:
                # Identify VNIR and SWIR by keywords or file size
                for file in group_files:
                    if 'vnir' in file.name.lower():
                        vnir_file = file
                    elif 'swir' in file.name.lower():
                        swir_file = file
                
                if not (vnir_file and swir_file):
                    group_files.sort(key=lambda x: x.stat().st_size, reverse=True)
                    vnir_file = group_files[0]  # Largest file as VNIR
                    swir_file = group_files[1]  # Second largest as SWIR
                
                logger.info(f"Found VNIR: {vnir_file}")
                logger.info(f"Found SWIR: {swir_file}")
                self.scene_logger.info(f"Found VNIR: {vnir_file}")
                self.scene_logger.info(f"Found SWIR: {swir_file}")
                break
        
        if not vnir_file:
            logger.warning("No VNIR file found")
            self.scene_logger.warning("No VNIR file found")
        if not swir_file:
            logger.warning("No SWIR file found")
            self.scene_logger.warning("No SWIR file found")
        
        return vnir_file, swir_file

    def _verify_file_content(self, file_path: Path, expected_type: str) -> bool:
        """
        Verify the content of an HDF file to determine if it contains VNIR or SWIR data
        
        Parameters:
        -----------
        file_path : Path
            Path to the HDF file
        expected_type : str
            Expected type ('VNIR' or 'SWIR')
            
        Returns:
        --------
        bool
            True if the file contains the expected type of data
        """
        try:
            with Dataset(file_path, 'r') as ds:
                variables = list(ds.variables.keys())
                
                has_vnir_prefix = any('SurfaceRadianceVNIR' in var for var in variables)
                has_swir_prefix = any('SurfaceRadianceSWIR' in var for var in variables)
                
                has_vnir_bands = any(f'Band{i}' in ''.join(variables) for i in range(1, 4))
                has_swir_bands = any(f'Band{i}' in ''.join(variables) for i in range(4, 10))
                
                if expected_type == 'VNIR':
                    return has_vnir_prefix or has_vnir_bands
                elif expected_type == 'SWIR':
                    return has_swir_prefix or has_swir_bands
                else:
                    return False
        except Exception as e:
            logger.error(f"Error verifying file content of {file_path}: {str(e)}")
            self.scene_logger.error(f"Error verifying file content of {file_path}: {str(e)}")
            return False
    
    def find_matching_swir_file(self, vnir_file: Path) -> Optional[Path]:
        """
        Find the matching SWIR file for a given VNIR file
        
        Parameters:
        -----------
        vnir_file : Path
            Path to the VNIR file
            
        Returns:
        --------
        Optional[Path]
            Path to the matching SWIR file, or None if not found
        """
        dir_path = vnir_file.parent
        vnir_stem = vnir_file.stem
        
        if "_" in vnir_stem:
            parts = vnir_stem.split("_")
            if len(parts) >= 2:
                timestamp_part_index = -2
                timestamp_part = parts[timestamp_part_index]
                if len(timestamp_part) == 14 and timestamp_part.isdigit():
                    timestamp_seconds = int(timestamp_part[-6:])
                    new_seconds = (timestamp_seconds + 1) % 1000000
                    new_timestamp = timestamp_part[:-6] + f"{new_seconds:06d}"
                    
                    parts[timestamp_part_index] = new_timestamp
                    swir_stem = "_".join(parts)
                    swir_file = dir_path / f"{swir_stem}{vnir_file.suffix}"
                    
                    if swir_file.exists():
                        logger.info(f"Found matching SWIR file using timestamp+1 pattern: {swir_file}")
                        self.scene_logger.info(f"Found matching SWIR file using timestamp+1 pattern: {swir_file}")
                        return swir_file
        
        for file_path in dir_path.glob(f"*SWIR*{vnir_file.suffix}"):
            logger.info(f"Found potential SWIR file with 'SWIR' in name: {file_path}")
            self.scene_logger.info(f"Found potential SWIR file with 'SWIR' in name: {file_path}")
            return file_path
        
        base_parts = []
        for part in vnir_stem.split("_"):
            if part.startswith("AST"):
                base_parts.append(part)
                continue
            if len(part) > 8 and part.isdigit():
                base_parts.append(part)
                
        if base_parts:
            base_pattern = "_".join(base_parts)
            for file_path in dir_path.glob(f"*{base_pattern}*{vnir_file.suffix}"):
                if file_path != vnir_file:
                    logger.info(f"Found potential SWIR file with matching base pattern: {file_path}")
                    self.scene_logger.info(f"Found potential SWIR file with matching base pattern: {file_path}")
                    return file_path
        
        logger.warning(f"No matching SWIR file found for {vnir_file}")
        self.scene_logger.warning(f"No matching SWIR file found for {vnir_file}")
        return None
    
    def _extract_coordinates_from_met(self, met_file: Path) -> Optional[Dict]:
        """
        Extract coordinates from MET file
        
        Parameters:
        -----------
        met_file : Path
            Path to the MET file
            
        Returns:
        --------
        Optional[Dict]
            Dictionary with extracted coordinates, or None if extraction failed
        """
        try:
            with open(met_file, 'r') as f:
                content = f.read()
                
            gpolygon_pattern = r'GROUP\s+=\s+GPOLYGON(.*?)END_GROUP\s+=\s+GPOLYGON'
            gpolygon_match = re.search(gpolygon_pattern, content, re.DOTALL)
            
            if gpolygon_match:
                logger.info("Found GPOLYGON section")
                self.scene_logger.info("Found GPOLYGON section")
                gpolygon_content = gpolygon_match.group(1)
                
                lon_pattern = r'GRINGPOINTLONGITUDE.*?VALUE\s+=\s+\((.*?)\)'
                lon_match = re.search(lon_pattern, gpolygon_content, re.DOTALL)
                lons = []
                if lon_match:
                    lon_str = lon_match.group(1)
                    try:
                        lons = [float(x.strip()) for x in lon_str.split(',')]
                    except Exception as e:
                        logger.error(f"Error parsing longitudes: {str(e)}")
                        self.scene_logger.error(f"Error parsing longitudes: {str(e)}")
                
                lat_pattern = r'GRINGPOINTLATITUDE.*?VALUE\s+=\s+\((.*?)\)'
                lat_match = re.search(lat_pattern, gpolygon_content, re.DOTALL)
                lats = []
                if lat_match:
                    lat_str = lat_match.group(1)
                    try:
                        lats = [float(x.strip()) for x in lat_str.split(',')]
                    except Exception as e:
                        logger.error(f"Error parsing latitudes: {str(e)}")
                        self.scene_logger.error(f"Error parsing latitudes: {str(e)}")
                
                if lons and lats:
                    return {
                        'west': min(lons),
                        'east': max(lons),
                        'south': min(lats),
                        'north': max(lats)
                    }
            
            return None
        except Exception as e:
            logger.error(f"Error reading MET file: {str(e)}")
            self.scene_logger.error(f"Error reading MET file: {str(e)}")
            return None
    
    def process(self, options: Dict = None) -> bool:
        """
        Process the ASTER data
        
        Parameters:
        -----------
        options : Dict, optional
            Processing options
                
        Returns:
        --------
        bool
            True if processing was successful
        """
        options = options or {}
        vnir_only_mode = options.get('vnir_only_mode', False)
        
        # Initialize processing report
        report_file = self.processed_dir / f"{self.scene_id}_processing_report.txt"
        with open(report_file, 'w') as f:
            f.write(f"ASTER Processing Report for Scene {self.scene_id}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Processing Date: {datetime.now().isoformat()}\n")
            f.write(f"Configuration: {json.dumps(options, indent=2)}\n\n")
            f.write("Processing Steps:\n")
        
        try:
            # Update status to processing
            self.update_status(
                ProcessingStatus.PROCESSING, 
                0, 
                ProcessingStages.EXTRACT,
                "Starting processing"
            )
            
            # Extract data if needed
            if options.get('extract', True):
                self.update_status(
                    ProcessingStatus.PROCESSING, 
                    5, 
                    ProcessingStages.EXTRACT,
                    "Extracting data"
                )
                with open(report_file, 'a') as f:
                    f.write("\n1. Data Extraction:\n")
                    f.write("------------------\n")
                
                success = self.extract_data()
                if not success:
                    self.update_status(
                        ProcessingStatus.FAILED, 
                        0, 
                        None,
                        "Data extraction failed"
                    )
                    with open(report_file, 'a') as f:
                        f.write("Status: Failed - Data extraction failed\n")
                    return False
                with open(report_file, 'a') as f:
                    f.write("Status: Successfully extracted data\n")
            
            # Find ASTER files
            vnir_file, swir_file = self.find_aster_files()
            
            # Check if we should use VNIR-only mode
            if not swir_file and not vnir_only_mode:
                logger.info("SWIR file not found, switching to VNIR-only mode")
                self.scene_logger.info("SWIR file not found, switching to VNIR-only mode")
                vnir_only_mode = True
            
            # Process in VNIR-only mode if specified or if SWIR is missing
            if vnir_only_mode:
                return self._process_vnir_only(vnir_file, options, report_file)
            
            # Check if both files are available for full processing
            if not (vnir_file and swir_file):
                self.update_status(
                    ProcessingStatus.FAILED, 
                    0, 
                    None,
                    "Could not find VNIR and SWIR files"
                )
                with open(report_file, 'a') as f:
                    f.write("Status: Failed - Could not find VNIR and SWIR files\n")
                return False
            
            # Initialize L2 processor with dynamic metadata
            from .aster_l2_processor import ASTER_L2_Processor, SceneMetadata, GeographicBounds
            
            # Extract georeference info
            temp_processor = ASTER_L2_Processor(
                vnir_file=str(vnir_file),
                swir_file=str(swir_file),
                metadata=None
            )
            georef = temp_processor.extract_georeference(vnir_file)
            if not georef or 'bounds' not in georef:
                logger.error(f"Could not extract geographic bounds from {vnir_file}")
                self.scene_logger.error(f"Could not extract geographic bounds from {vnir_file}")
                self.update_status(
                    ProcessingStatus.FAILED,
                    0,
                    None,
                    f"Could not extract geographic bounds from {vnir_file}"
                )
                with open(report_file, 'a') as f:
                    f.write("Status: Failed - Could not extract geographic bounds\n")
                return False
            
            scene_metadata = SceneMetadata(
                bounds=GeographicBounds(
                    west=georef['bounds']['west'],
                    east=georef['bounds']['east'],
                    south=georef['bounds']['south'],
                    north=georef['bounds']['north']
                ),
                solar_azimuth=georef.get('solar_azimuth', 152.9561090000),
                solar_elevation=georef.get('solar_elevation', 53.5193190000),
                cloud_cover=georef.get('cloud_cover', 5.0),
                acquisition_date=georef.get('acquisition_date', "2000-12-18"),
                utm_zone=georef.get('utm_zone')
            )
            
            # Check cloud cover threshold
            cloud_threshold = options.get('cloud_threshold', 20.0)
            if scene_metadata.cloud_cover > cloud_threshold:
                logger.warning(f"Skipping scene due to high cloud cover: {scene_metadata.cloud_cover}% > {cloud_threshold}%")
                self.scene_logger.warning(f"Skipping scene due to high cloud cover: {scene_metadata.cloud_cover}% > {cloud_threshold}%")
                self.update_status(
                    ProcessingStatus.SKIPPED,
                    0,
                    None,
                    f"High cloud cover: {scene_metadata.cloud_cover}%"
                )
                with open(report_file, 'a') as f:
                    f.write(f"Status: Skipped - High cloud cover: {scene_metadata.cloud_cover}%\n")
                return False
            
            self.update_status(
                ProcessingStatus.PROCESSING, 
                10, 
                ProcessingStages.EXTRACT,
                "Initializing processor"
            )
            with open(report_file, 'a') as f:
                f.write(f"Geographic Bounds: W={scene_metadata.bounds.west}, E={scene_metadata.bounds.east}, "
                        f"S={scene_metadata.bounds.south}, N={scene_metadata.bounds.north}\n")
                f.write(f"Cloud Cover: {scene_metadata.cloud_cover}%\n")
                f.write(f"Acquisition Date: {scene_metadata.acquisition_date}\n")
            
            processor = ASTER_L2_Processor(
                vnir_file=str(vnir_file),
                swir_file=str(swir_file),
                metadata=scene_metadata
            )
            
            # Process minerals if requested
            if options.get('process_minerals', True):
                self.update_status(
                    ProcessingStatus.PROCESSING, 
                    20, 
                    ProcessingStages.MINERAL_MAPPING,
                    "Processing mineral maps"
                )
                with open(report_file, 'a') as f:
                    f.write("\n2. Mineral Mapping:\n")
                    f.write("-----------------\n")
                
                minerals_dir = self.processed_dir / 'minerals'
                minerals_dir.mkdir(exist_ok=True)
                
                # Check write permissions
                if not os.access(minerals_dir, os.W_OK):
                    logger.error(f"No write permission for {minerals_dir}")
                    self.scene_logger.error(f"No write permission for {minerals_dir}")
                    self.update_status(
                        ProcessingStatus.FAILED,
                        0,
                        None,
                        f"No write permission for {minerals_dir}"
                    )
                    with open(report_file, 'a') as f:
                        f.write(f"Status: Failed - No write permission for {minerals_dir}\n")
                    return False
                
                from .aster_l2_processor import MineralIndices
                
                logger.info(f"Starting mineral mapping for {len(list(MineralIndices))} minerals")
                self.scene_logger.info(f"Starting mineral mapping for {len(list(MineralIndices))} minerals")
                
                # Try a subset of common minerals first to ensure the process works
                priority_minerals = [
                    MineralIndices.ALUNITE,
                    MineralIndices.KAOLINITE,
                    MineralIndices.CALCITE,
                    MineralIndices.CHLORITE,
                    MineralIndices.IRON_OXIDE
                ]
                
                # Process priority minerals first
                for mineral in priority_minerals:
                    try:
                        logger.info(f"Processing priority mineral {mineral.value}")
                        self.scene_logger.info(f"Processing priority mineral {mineral.value}")
                        
                        # First check if the required bands are valid
                        valid_data = processor.validate_data(mineral)
                        if not valid_data:
                            logger.warning(f"Skipping {mineral.value} - required bands are invalid")
                            self.scene_logger.warning(f"Skipping {mineral.value} - required bands are invalid")
                            with open(report_file, 'a') as f:
                                f.write(f"  - {mineral.value}: Skipped - required bands are invalid\n")
                            continue
                            
                        # Process the mineral map
                        processor.save_mineral_map(mineral, minerals_dir)
                        logger.info(f"Processed {mineral.value} map")
                        self.scene_logger.info(f"Processed {mineral.value} map")
                        with open(report_file, 'a') as f:
                            f.write(f"  - {mineral.value}: Processed successfully\n")
                    except Exception as e:
                        logger.error(f"Error processing {mineral.value} map: {str(e)}")
                        logger.error(f"Stack trace: {traceback.format_exc()}")
                        self.scene_logger.error(f"Error processing {mineral.value} map: {str(e)}")
                        self.scene_logger.error(f"Stack trace: {traceback.format_exc()}")
                        with open(report_file, 'a') as f:
                            f.write(f"  - {mineral.value}: Error - {str(e)}\n")
                
                # Process remaining minerals
                remaining_minerals = [m for m in MineralIndices if m not in priority_minerals]
                for mineral in remaining_minerals:
                    try:
                        valid_data = processor.validate_data(mineral)
                        if not valid_data:
                            logger.warning(f"Skipping {mineral.value} - required bands are invalid")
                            self.scene_logger.warning(f"Skipping {mineral.value} - required bands are invalid")
                            with open(report_file, 'a') as f:
                                f.write(f"  - {mineral.value}: Skipped - required bands are invalid\n")
                            continue
                            
                        processor.save_mineral_map(mineral, minerals_dir)
                        logger.info(f"Processed {mineral.value} map")
                        self.scene_logger.info(f"Processed {mineral.value} map")
                        with open(report_file, 'a') as f:
                            f.write(f"  - {mineral.value}: Processed successfully\n")
                    except Exception as e:
                        logger.error(f"Error processing {mineral.value} map: {str(e)}")
                        logger.error(f"Stack trace: {traceback.format_exc()}")
                        self.scene_logger.error(f"Error processing {mineral.value} map: {str(e)}")
                        self.scene_logger.error(f"Stack trace: {traceback.format_exc()}")
                        with open(report_file, 'a') as f:
                            f.write(f"  - {mineral.value}: Error - {str(e)}\n")
            # Initialize geological mapper
            from .aster_geological_mapper import ASTER_Geological_Mapper
            
            geological_mapper = ASTER_Geological_Mapper(processor)
            
            # Process alteration maps if requested
            if options.get('process_alteration', True):
                self.update_status(
                    ProcessingStatus.PROCESSING, 
                    40, 
                    ProcessingStages.ALTERATION_MAPPING,
                    "Processing alteration maps"
                )
                with open(report_file, 'a') as f:
                    f.write("\n3. Alteration Mapping:\n")
                    f.write("--------------------\n")
                
                alteration_dir = self.processed_dir / 'alteration'
                alteration_dir.mkdir(exist_ok=True)
                
                from enums import AlterationIndices
                
                # Ensure geological_mapper is initialized with base_processor
                geological_mapper = ASTER_Geological_Mapper(base_processor=processor)
                
                for alteration in AlterationIndices:
                    try:
                        index_map, confidence_map = geological_mapper.save_alteration_map(alteration, alteration_dir)
                        if index_map is None or confidence_map is None:
                            logger.warning(f"Skipping {alteration.value} - not supported or failed to calculate")
                            with open(report_file, 'a') as f:
                                f.write(f"  - {alteration.value}: Skipped - not supported or failed\n")
                            continue
                        logger.info(f"Processed {alteration.value} map")
                        self.scene_logger.info(f"Processed {alteration.value} map")
                        with open(report_file, 'a') as f:
                            f.write(f"  - {alteration.value}: Processed successfully\n")
                    except Exception as e:
                        logger.error(f"Error processing {alteration.value} map: {str(e)}")
                        logger.error(f"Stack trace: {traceback.format_exc()}")
                        self.scene_logger.error(f"Error processing {alteration.value} map: {str(e)}")
                        self.scene_logger.error(f"Stack trace: {traceback.format_exc()}")
                        with open(report_file, 'a') as f:
                            f.write(f"  - {alteration.value}: Error - {str(e)}\n")
            
            # Process gold pathfinder maps if requested
            if options.get('process_gold_pathfinders', True):
                self.update_status(
                    ProcessingStatus.PROCESSING, 
                    60, 
                    ProcessingStages.GOLD_PATHFINDER,
                    "Processing gold pathfinder maps"
                )
                with open(report_file, 'a') as f:
                    f.write("\n4. Gold Pathfinder Mapping:\n")
                    f.write("--------------------------\n")
                
                gold_dir = self.processed_dir / 'gold_pathfinders'
                gold_dir.mkdir(exist_ok=True)
                
                from enums import GoldPathfinderIndices
                
                for pathfinder in GoldPathfinderIndices:
                    try:
                        processor.save_mineral_map(pathfinder, gold_dir)
                        logger.info(f"Processed {pathfinder.value} map")
                        self.scene_logger.info(f"Processed {pathfinder.value} map")
                        with open(report_file, 'a') as f:
                            f.write(f"  - {pathfinder.value}: Processed successfully\n")
                    except Exception as e:
                        logger.error(f"Error processing {pathfinder.value} map: {str(e)}")
                        logger.error(f"Stack trace: {traceback.format_exc()}")
                        self.scene_logger.error(f"Error processing {pathfinder.value} map: {str(e)}")
                        self.scene_logger.error(f"Stack trace: {traceback.format_exc()}")
                        with open(report_file, 'a') as f:
                            f.write(f"  - {pathfinder.value}: Error - {str(e)}\n")
            
            # Process RGB composites and band combinations
            if options.get('enhanced_visualization', True):
                self.update_status(
                    ProcessingStatus.PROCESSING, 
                    80, 
                    ProcessingStages.GEOLOGICAL_MAPPING,
                    "Creating RGB composites"
                )
                with open(report_file, 'a') as f:
                    f.write("\n5. RGB Composites:\n")
                    f.write("-----------------\n")
                
                rgb_dir = self.processed_dir / 'rgb_composites'
                rgb_dir.mkdir(exist_ok=True)
                
                # Check write permissions
                if not os.access(rgb_dir, os.W_OK):
                    logger.error(f"No write permission for {rgb_dir}")
                    self.scene_logger.error(f"No write permission for {rgb_dir}")
                    self.update_status(
                        ProcessingStatus.FAILED,
                        0,
                        None,
                        f"No write permission for {rgb_dir}"
                    )
                    with open(report_file, 'a') as f:
                        f.write(f"Status: Failed - No write permission for {rgb_dir}\n")
                    return False
                
                from enums import BandCombinations
                
                # Initialize the geological mapper if not already done elsewhere
                from .aster_geological_mapper import ASTER_Geological_Mapper
                geological_mapper = ASTER_Geological_Mapper(processor)
                
                for combo in BandCombinations:
                    try:
                        # Get required bands from the mapper
                        required_bands = geological_mapper.get_required_bands(combo)
                        valid_combo = True
                        
                        for band in required_bands:
                            band_data = processor.get_band_data(band)
                            if band_data is None:
                                logger.warning(f"Skipping {combo.value}: Band {band} not found")
                                self.scene_logger.warning(f"Skipping {combo.value}: Band {band} not found")
                                with open(report_file, 'a') as f:
                                    f.write(f"  - {combo.value}: Skipped - Band {band} not found\n")
                                valid_combo = False
                                break  # Exit the inner loop
                                
                            # Check if band data has enough valid values
                            valid_mask = ~np.isnan(band_data)
                            valid_pixel_count = np.sum(valid_mask)
                            total_pixels = band_data.size
                            valid_percentage = (valid_pixel_count / total_pixels) * 100
                            
                            if valid_pixel_count < (total_pixels * 0.05):  # Require at least 5% valid pixels
                                logger.warning(f"Skipping {combo.value}: Insufficient valid data in band {band} (only {valid_percentage:.2f}% valid)")
                                self.scene_logger.warning(f"Skipping {combo.value}: Insufficient valid data in band {band} (only {valid_percentage:.2f}% valid)")
                                with open(report_file, 'a') as f:
                                    f.write(f"  - {combo.value}: Skipped - Insufficient valid data in band {band} (only {valid_percentage:.2f}% valid)\n")
                                valid_combo = False
                                break  # Exit the inner loop
                        
                        # Only proceed if all bands are valid
                        if valid_combo:
                            logger.info(f"Creating band combination: {combo.value}")
                            self.scene_logger.info(f"Creating band combination: {combo.value}")
                            geological_mapper.create_band_combination_map(combo, rgb_dir)
                            logger.info(f"Created {combo.value} band combination")
                            self.scene_logger.info(f"Created {combo.value} band combination")
                            with open(report_file, 'a') as f:
                                f.write(f"  - {combo.value}: Processed successfully\n")
                        else:
                            logger.info(f"Skipped {combo.value} due to invalid or missing band data")
                            self.scene_logger.info(f"Skipped {combo.value} due to invalid or missing band data")
                    except Exception as e:
                        logger.error(f"Error creating {combo.value} band combination: {str(e)}")
                        logger.error(f"Stack trace: {traceback.format_exc()}")
                        self.scene_logger.error(f"Error creating {combo.value} band combination: {str(e)}")
                        self.scene_logger.error(f"Stack trace: {traceback.format_exc()}")
                        with open(report_file, 'a') as f:
                            f.write(f"  - {combo.value}: Error - {str(e)}\n")
            
            # Perform advanced analysis if requested
            if options.get('process_advanced_analysis', True):
                self.update_status(
                    ProcessingStatus.PROCESSING, 
                    90, 
                    ProcessingStages.ADVANCED_ANALYSIS,
                    "Performing advanced analysis"
                )
                with open(report_file, 'a') as f:
                    f.write("\n6. Advanced Analysis:\n")
                    f.write("--------------------\n")
                
                analysis_dir = self.processed_dir / 'analysis'
                analysis_dir.mkdir(exist_ok=True)
                
                from .aster_advanced_analysis import ASTER_Advanced_Analysis
                
                advanced = ASTER_Advanced_Analysis(processor, geological_mapper)
                
                try:
                    advanced.create_band_ratio_matrix(analysis_dir)
                    logger.info("Created band ratio matrix")
                    self.scene_logger.info("Created band ratio matrix")
                    with open(report_file, 'a') as f:
                        f.write("  - Band ratio matrix: Processed successfully\n")
                except Exception as e:
                    logger.error(f"Error creating band ratio matrix: {str(e)}")
                    logger.error(f"Stack trace: {traceback.format_exc()}")
                    self.scene_logger.error(f"Error creating band ratio matrix: {str(e)}")
                    self.scene_logger.error(f"Stack trace: {traceback.format_exc()}")
                    with open(report_file, 'a') as f:
                        f.write(f"  - Band ratio matrix: Error - {str(e)}\n")
                
                try:
                    advanced.extract_geological_features(analysis_dir)
                    logger.info("Extracted geological features")
                    self.scene_logger.info("Extracted geological features")
                    with open(report_file, 'a') as f:
                        f.write("  - Geological features: Processed successfully\n")
                except Exception as e:
                    logger.error(f"Error extracting geological features: {str(e)}")
                    logger.error(f"Stack trace: {traceback.format_exc()}")
                    self.scene_logger.error(f"Error extracting geological features: {str(e)}")
                    self.scene_logger.error(f"Stack trace: {traceback.format_exc()}")
                    with open(report_file, 'a') as f:
                        f.write(f"  - Geological features: Error - {str(e)}\n")
                
                try:
                    advanced.generate_mineral_potential_map(analysis_dir)
                    logger.info("Generated mineral potential map")
                    self.scene_logger.info("Generated mineral potential map")
                    with open(report_file, 'a') as f:
                        f.write("  - Mineral potential map: Processed successfully\n")
                except Exception as e:
                    logger.error(f"Error generating mineral potential map: {str(e)}")
                    logger.error(f"Stack trace: {traceback.format_exc()}")
                    self.scene_logger.error(f"Error generating mineral potential map: {str(e)}")
                    self.scene_logger.error(f"Stack trace: {traceback.format_exc()}")
                    with open(report_file, 'a') as f:
                        f.write(f"  - Mineral potential map: Error - {str(e)}\n")
            
            # Copy metadata to processed directory
            metadata_file = self.raw_dir / 'metadata.json'
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                with open(self.processed_dir / 'metadata.json', 'w') as f:
                    json.dump({
                        'scene_id': self.scene_id,
                        'name': metadata.get('original_filename', self.scene_id),
                        'date': metadata.get('upload_date', datetime.now().isoformat()),
                        'bounds': {
                            'west': scene_metadata.bounds.west,
                            'east': scene_metadata.bounds.east,
                            'south': scene_metadata.bounds.south,
                            'north': scene_metadata.bounds.north
                        },
                        'cloud_cover': scene_metadata.cloud_cover,
                        'solar_azimuth': scene_metadata.solar_azimuth,
                        'solar_elevation': scene_metadata.solar_elevation,
                        'acquisition_date': scene_metadata.acquisition_date
                    }, f, indent=2)
            
            # Finalize processing report
            with open(report_file, 'a') as f:
                f.write("\n\nProcessing Summary:\n")
                f.write("-----------------\n")
                f.write(f"Processing completed at: {datetime.now().isoformat()}\n")
                f.write(f"Output directory: {self.processed_dir}\n")
            
            # Update status to completed
            self.update_status(
                ProcessingStatus.COMPLETED, 
                100, 
                None,
                "Processing completed"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing scene {self.scene_id}: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            self.scene_logger.error(f"Error processing scene {self.scene_id}: {str(e)}")
            self.scene_logger.error(f"Stack trace: {traceback.format_exc()}")
            
            self.update_status(
                ProcessingStatus.FAILED, 
                0, 
                None,
                str(e)
            )
            with open(report_file, 'a') as f:
                f.write(f"Status: Failed - {str(e)}\n")
            
            return False


    def get_band_data(self, band_number: int) -> Optional[np.ndarray]:
        try:
            if band_number in self.reflectance_data:
                data = self.reflectance_data[band_number]
                if np.any(data):
                    return data
                logger.warning(f"Band {band_number} contains no valid data")
                return None
            logger.warning(f"Band {band_number} not found in reflectance data")
            return None
        except Exception as e:
            logger.error(f"Error retrieving band {band_number}: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            return None


    def _process_vnir_only(self, vnir_file: Optional[Path], options: Dict, report_file: Path) -> bool:
        """
        Process ASTER data in VNIR-only mode
        
        Parameters:
        -----------
        vnir_file : Optional[Path]
            Path to the VNIR file
        options : Dict
            Processing options
        report_file : Path
            Path to the processing report file
            
        Returns:
        --------
        bool
            True if processing was successful
        """
        if not vnir_file:
            self.update_status(
                ProcessingStatus.FAILED, 
                0, 
                None,
                "No VNIR file found for VNIR-only processing"
            )
            with open(report_file, 'a') as f:
                f.write("Status: Failed - No VNIR file found for VNIR-only processing\n")
            return False
        
        try:
            self.update_status(
                ProcessingStatus.PROCESSING, 
                10, 
                ProcessingStages.EXTRACT,
                "Initializing VNIR-only processor"
            )
            with open(report_file, 'a') as f:
                f.write("\nVNIR-Only Processing:\n")
                f.write("--------------------\n")
            
            from processors.vnir_processor import VNIRProcessor
            from processors.aster_l2_processor import SceneMetadata, GeographicBounds
            
            metadata_bounds = None
            metadata_file = Path(str(vnir_file) + '.met')
            if metadata_file.exists():
                try:
                    coords = self._extract_coordinates_from_met(metadata_file)
                    if coords:
                        metadata_bounds = GeographicBounds(
                            west=coords['west'],
                            east=coords['east'],
                            south=coords['south'],
                            north=coords['north']
                        )
                except Exception as e:
                    logger.error(f"Error reading metadata file: {str(e)}")
                    self.scene_logger.error(f"Error reading metadata file: {str(e)}")
            
            if not metadata_bounds:
                metadata_bounds = GeographicBounds(
                    west=-2.67, east=-1.99, south=8.76, north=9.43
                )
            
            scene_metadata = SceneMetadata(
                bounds=metadata_bounds,
                solar_azimuth=152.9561090000,
                solar_elevation=53.5193190000,
                cloud_cover=5.0,
                acquisition_date=datetime.now().strftime('%Y-%m-%d')
            )
            
            vnir_processor = VNIRProcessor(
                vnir_file=str(vnir_file),
                metadata=scene_metadata
            )
            
            self.update_status(
                ProcessingStatus.PROCESSING, 
                20, 
                ProcessingStages.MINERAL_MAPPING,
                "Processing VNIR bands"
            )
            with open(report_file, 'a') as f:
                f.write("\n1. VNIR Band Processing:\n")
                f.write("-----------------------\n")
            
            rgb_dir = self.processed_dir / 'rgb_composites'
            rgb_dir.mkdir(exist_ok=True)
            
            minerals_dir = self.processed_dir / 'minerals'
            minerals_dir.mkdir(exist_ok=True)
            
            if options.get('process_minerals', True):
                try:
                    vnir_processor.save_band_as_geotiff(1, minerals_dir / "band1_blue.tif")
                    vnir_processor.save_band_as_geotiff(2, minerals_dir / "band2_green.tif")
                    vnir_processor.save_band_as_geotiff(3, minerals_dir / "band3_nir.tif")
                    logger.info("Saved VNIR bands")
                    self.scene_logger.info("Saved VNIR bands")
                    with open(report_file, 'a') as f:
                        f.write("  - VNIR bands: Processed successfully\n")
                except Exception as e:
                    logger.error(f"Error saving VNIR bands: {str(e)}")
                    logger.error(f"Stack trace: {traceback.format_exc()}")
                    self.scene_logger.error(f"Error saving VNIR bands: {str(e)}")
                    self.scene_logger.error(f"Stack trace: {traceback.format_exc()}")
                    with open(report_file, 'a') as f:
                        f.write(f"  - VNIR bands: Error - {str(e)}\n")
            
            self.update_status(
                ProcessingStatus.PROCESSING, 
                40, 
                ProcessingStages.GEOLOGICAL_MAPPING,
                "Creating band composites"
            )
            with open(report_file, 'a') as f:
                f.write("\n2. Band Composites:\n")
                f.write("------------------\n")
            
            if options.get('enhanced_visualization', True):
                try:
                    vnir_processor.create_true_color_composite(rgb_dir / "true_color_composite.tif")
                    vnir_processor.create_false_color_composite(rgb_dir / "false_color_composite.tif")
                    logger.info("Created VNIR composites")
                    self.scene_logger.info("Created VNIR composites")
                    with open(report_file, 'a') as f:
                        f.write("  - VNIR composites: Processed successfully\n")
                except Exception as e:
                    logger.error(f"Error creating composites: {str(e)}")
                    logger.error(f"Stack trace: {traceback.format_exc()}")
                    self.scene_logger.error(f"Error creating composites: {str(e)}")
                    self.scene_logger.error(f"Stack trace: {traceback.format_exc()}")
                    with open(report_file, 'a') as f:
                        f.write(f"  - VNIR composites: Error - {str(e)}\n")
            
            self.update_status(
                ProcessingStatus.PROCESSING, 
                60, 
                ProcessingStages.MINERAL_MAPPING,
                "Creating vegetation indices"
            )
            with open(report_file, 'a') as f:
                f.write("\n3. Vegetation Indices:\n")
                f.write("--------------------\n")
            
            if options.get('process_minerals', True):
                try:
                    vnir_processor.create_ndvi_map(minerals_dir / "ndvi_map.tif")
                    logger.info("Created NDVI map")
                    self.scene_logger.info("Created NDVI map")
                    with open(report_file, 'a') as f:
                        f.write("  - NDVI map: Processed successfully\n")
                except Exception as e:
                    logger.error(f"Error creating NDVI map: {str(e)}")
                    logger.error(f"Stack trace: {traceback.format_exc()}")
                    self.scene_logger.error(f"Error creating NDVI map: {str(e)}")
                    self.scene_logger.error(f"Stack trace: {traceback.format_exc()}")
                    with open(report_file, 'a') as f:
                        f.write(f"  - NDVI map: Error - {str(e)}\n")
            
            self.update_status(
                ProcessingStatus.PROCESSING, 
                80, 
                ProcessingStages.EXTRACT,
                "Updating metadata"
            )
            with open(report_file, 'a') as f:
                f.write("\n4. Metadata Update:\n")
                f.write("------------------\n")
            
            metadata_path = self.processed_dir / 'metadata.json'
            metadata = {
                'scene_id': self.scene_id,
                'processing_mode': 'VNIR-only',
                'bounds': {
                    'west': scene_metadata.bounds.west,
                    'east': scene_metadata.bounds.east,
                    'south': scene_metadata.bounds.south,
                    'north': scene_metadata.bounds.north
                },
                'acquisition_date': scene_metadata.acquisition_date,
                'solar_azimuth': scene_metadata.solar_azimuth,
                'solar_elevation': scene_metadata.solar_elevation,
                'cloud_cover': scene_metadata.cloud_cover
            }
            
            raw_metadata_path = self.raw_dir / 'metadata.json'
            if raw_metadata_path.exists():
                try:
                    with open(raw_metadata_path, 'r') as f:
                        raw_metadata = json.load(f)
                        metadata['name'] = raw_metadata.get('original_filename', self.scene_id)
                        metadata['upload_date'] = raw_metadata.get('upload_date', datetime.now().isoformat())
                except Exception as e:
                    logger.error(f"Error reading raw metadata: {str(e)}")
                    self.scene_logger.error(f"Error reading raw metadata: {str(e)}")
                    metadata['name'] = self.scene_id
                    metadata['upload_date'] = datetime.now().isoformat()
            else:
                metadata['name'] = self.scene_id
                metadata['upload_date'] = datetime.now().isoformat()
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            with open(report_file, 'a') as f:
                f.write("  - Metadata: Saved successfully\n")
            
            try:
                vnir_processor.create_thumbnail(self.processed_dir / 'thumbnail.png')
                logger.info("Created thumbnail")
                self.scene_logger.info("Created thumbnail")
                with open(report_file, 'a') as f:
                    f.write("  - Thumbnail: Created successfully\n")
            except Exception as e:
                logger.error(f"Error creating thumbnail: {str(e)}")
                logger.error(f"Stack trace: {traceback.format_exc()}")
                self.scene_logger.error(f"Error creating thumbnail: {str(e)}")
                self.scene_logger.error(f"Stack trace: {traceback.format_exc()}")
                with open(report_file, 'a') as f:
                    f.write(f"  - Thumbnail: Error - {str(e)}\n")
            
            self.update_status(
                ProcessingStatus.COMPLETED, 
                100, 
                None,
                "VNIR-only processing completed"
            )
            with open(report_file, 'a') as f:
                f.write("\nProcessing Summary:\n")
                f.write("-----------------\n")
                f.write(f"VNIR-only processing completed at: {datetime.now().isoformat()}\n")
                f.write(f"Output directory: {self.processed_dir}\n")
            
            logger.info(f"VNIR-only processing completed for scene {self.scene_id}")
            self.scene_logger.info(f"VNIR-only processing completed for scene {self.scene_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error in VNIR-only processing: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            self.scene_logger.error(f"Error in VNIR-only processing: {str(e)}")
            self.scene_logger.error(f"Stack trace: {traceback.format_exc()}")
            self.update_status(
                ProcessingStatus.FAILED, 
                0, 
                None,
                f"Error in VNIR-only processing: {str(e)}"
            )
            with open(report_file, 'a') as f:
                f.write(f"Status: Failed - {str(e)}\n")
            return False
    
    def generate_prospectivity_map(self, options: Dict = None) -> str:
        """
        Generate a gold prospectivity map
        
        Parameters:
        -----------
        options : Dict, optional
            Prospectivity generation options
            
        Returns:
        --------
        str
            Path to the generated prospectivity map
        """
        options = options or {}
        
        report_file = self.processed_dir / f"{self.scene_id}_processing_report.txt"
        with open(report_file, 'a') as f:
            f.write("\n7. Gold Prospectivity Mapping:\n")
            f.write("----------------------------\n")
        
        threshold = options.get('threshold', 0.7)
        pathfinders = options.get('pathfinders', [
            'gold_alteration', 'pyrite', 'arsenopyrite', 'advanced_argillic_gold'
        ])
        alterations = options.get('alterations', [
            'advanced_argillic', 'phyllic', 'silicification'
        ])
        
        prospectivity_dir = self.processed_dir / 'prospectivity'
        prospectivity_dir.mkdir(exist_ok=True)
        
        from .gold_prospectivity_mapper import GoldProspectivityMapper
        
        mapper = GoldProspectivityMapper(output_directory=str(prospectivity_dir))
        
        gold_dir = self.processed_dir / 'gold_pathfinders'
        if gold_dir.exists():
            for pathfinder in pathfinders:
                pathfinder_map = gold_dir / f"{pathfinder}_map.tif"
                if pathfinder_map.exists():
                    mapper.add_mineral_map(pathfinder, str(pathfinder_map), weight=0.8)
                    logger.info(f"Added {pathfinder} map to prospectivity")
                    self.scene_logger.info(f"Added {pathfinder} map to prospectivity")
                    with open(report_file, 'a') as f:
                        f.write(f"  - {pathfinder}: Added to prospectivity map\n")
        
        alteration_dir = self.processed_dir / 'alteration'
        if alteration_dir.exists():
            for alteration in alterations:
                alteration_map = alteration_dir / f"{alteration}_map.tif"
                if alteration_map.exists():
                    mapper.add_mineral_map(alteration, str(alteration_map), weight=0.6)
                    logger.info(f"Added {alteration} map to prospectivity")
                    self.scene_logger.info(f"Added {alteration} map to prospectivity")
                    with open(report_file, 'a') as f:
                        f.write(f"  - {alteration}: Added to prospectivity map\n")
        
        try:
            prospectivity_map = mapper.generate_prospectivity_map("gold_prospectivity.tif")
            logger.info(f"Generated prospectivity map: {prospectivity_map}")
            self.scene_logger.info(f"Generated prospectivity map: {prospectivity_map}")
            with open(report_file, 'a') as f:
                f.write(f"  - Prospectivity map: Generated successfully at {prospectivity_map}\n")
            
            visualization = mapper.visualize_map(
                "gold_prospectivity_visualization.png", 
                colormap="prospectivity", 
                title="Gold Prospectivity Map"
            )
            logger.info(f"Generated visualization: {visualization}")
            self.scene_logger.info(f"Generated visualization: {visualization}")
            with open(report_file, 'a') as f:
                f.write(f"  - Visualization: Generated successfully at {visualization}\n")
            
            high_areas = mapper.get_high_prospectivity_areas(threshold, "high_prospectivity.tif")
            logger.info(f"Extracted high prospectivity areas: {high_areas}")
            self.scene_logger.info(f"Extracted high prospectivity areas: {high_areas}")
            with open(report_file, 'a') as f:
                f.write(f"  - High prospectivity areas: Extracted successfully at {high_areas}\n")
            
            geojson = mapper.export_to_geojson(threshold, "high_prospectivity.geojson")
            logger.info(f"Exported GeoJSON: {geojson}")
            self.scene_logger.info(f"Exported GeoJSON: {geojson}")
            with open(report_file, 'a') as f:
                f.write(f"  - GeoJSON: Exported successfully at {geojson}\n")
            
            return prospectivity_map
        except Exception as e:
            logger.error(f"Error generating prospectivity map: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            self.scene_logger.error(f"Error generating prospectivity map: {str(e)}")
            self.scene_logger.error(f"Stack trace: {traceback.format_exc()}")
            with open(report_file, 'a') as f:
                f.write(f"  - Prospectivity map: Error - {str(e)}\n")
            return ""
    
    def generate_report(self, report_type: str, options: Dict = None) -> str:
        """
        Generate an analysis report
        
        Parameters:
        -----------
        report_type : str
            Type of report to generate
        options : Dict, optional
            Report generation options
            
        Returns:
        --------
        str
            Path to the generated report
        """
        options = options or {}
        
        reports_dir = self.processed_dir / 'reports'
        reports_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = reports_dir / f"{self.scene_id}_{report_type}_{timestamp}.pdf"
        
        try:
            with open(report_file, 'w') as f:
                f.write(f"ASTER Analysis Report\n")
                f.write(f"Scene ID: {self.scene_id}\n")
                f.write(f"Report Type: {report_type}\n")
                f.write(f"Generated: {timestamp}\n\n")
                f.write("Options:\n")
                for key, value in options.items():
                    f.write(f"  {key}: {value}\n")
            
            logger.info(f"Generated report: {report_file}")
            self.scene_logger.info(f"Generated report: {report_file}")
            return str(report_file)
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            self.scene_logger.error(f"Error generating report: {str(e)}")
            self.scene_logger.error(f"Stack trace: {traceback.format_exc()}")
            return ""