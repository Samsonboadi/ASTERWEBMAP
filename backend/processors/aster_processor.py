# backend/processors/aster_processor.py
"""
ASTER Processor Module
Primary integration point for ASTER data processing
"""

import os
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union

from enums import ProcessingStatus, ProcessingStages

logger = logging.getLogger(__name__)

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
        
        # Ensure directories exist
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.extracted_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized ASTER processor for scene {scene_id}")
    
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
    
    def extract_data(self) -> bool:
        """
        Extract ASTER data from zip/archive files
        
        Returns:
        --------
        bool
            True if extraction was successful
        """
        # Look for zip files in raw directory
        zip_files = list(self.raw_dir.glob('*.zip'))
        
        if not zip_files:
            logger.info(f"No zip files found in {self.raw_dir}")
            return False
        
        try:
            import zipfile
            
            for zip_file in zip_files:
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(self.extracted_dir)
                
                logger.info(f"Extracted {zip_file}")
            
            return True
        except Exception as e:
            logger.error(f"Error extracting data: {str(e)}")
            return False
    
    def find_aster_files(self) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Find VNIR and SWIR files in the extracted directory
        
        Returns:
        --------
        Tuple[Optional[Path], Optional[Path]]
            Paths to VNIR and SWIR files
        """
        # Look for HDF files in extracted directory
        vnir_file = None
        swir_file = None
        hdf_files = list(self.extracted_dir.glob('**/*.hdf'))
        
        logger.info(f"Found {len(hdf_files)} HDF files in {self.extracted_dir}")
        
        # First, try to identify files by checking file contents
        for file_path in hdf_files:
            try:
                with Dataset(file_path, 'r') as ds:
                    variables = list(ds.variables.keys())
                    
                    # Check for VNIR bands (1-3)
                    has_vnir_bands = any(f'Band{i}' in variables for i in range(1, 4))
                    # Check for SWIR bands (4-9)
                    has_swir_bands = any(f'Band{i}' in variables for i in range(4, 10))
                    
                    if has_vnir_bands and not has_swir_bands:
                        vnir_file = file_path
                        logger.info(f"Identified VNIR file based on band content: {vnir_file}")
                    elif has_swir_bands:
                        swir_file = file_path
                        logger.info(f"Identified SWIR file based on band content: {swir_file}")
            except Exception as e:
                logger.warning(f"Error examining file {file_path}: {str(e)}")
        
        # If we haven't identified both files by content, try filename patterns
        if not (vnir_file and swir_file):
            # Look for files with VNIR or SWIR in the name
            for file_path in hdf_files:
                lower_name = file_path.name.lower()
                if 'vnir' in lower_name:
                    vnir_file = file_path
                    logger.info(f"Identified VNIR file based on filename: {vnir_file}")
                elif 'swir' in lower_name:
                    swir_file = file_path
                    logger.info(f"Identified SWIR file based on filename: {swir_file}")
        
        # If we've found VNIR but not SWIR, try to find a matching SWIR file
        if vnir_file and not swir_file:
            swir_file = self.find_matching_swir_file(vnir_file)
        
        # If we still haven't found the files, try picking by file size (VNIR typically larger)
        if not (vnir_file and swir_file) and len(hdf_files) >= 2:
            hdf_files.sort(key=lambda x: x.stat().st_size, reverse=True)
            vnir_file = hdf_files[0]
            swir_file = hdf_files[1]
            logger.info(f"Using largest file as VNIR: {vnir_file}")
            logger.info(f"Using second largest file as SWIR: {swir_file}")
        
        # Verify the files by checking their content
        if vnir_file and swir_file:
            # Verify VNIR file actually has VNIR bands
            try:
                with Dataset(vnir_file, 'r') as ds:
                    variables = list(ds.variables.keys())
                    has_vnir_bands = any(f'Band{i}' in variables for i in range(1, 4))
                    if not has_vnir_bands:
                        logger.warning(f"Supposed VNIR file {vnir_file} doesn't contain VNIR bands")
                        vnir_file = None
            except Exception as e:
                logger.warning(f"Error verifying VNIR file {vnir_file}: {str(e)}")
            
            # Verify SWIR file actually has SWIR bands
            try:
                with Dataset(swir_file, 'r') as ds:
                    variables = list(ds.variables.keys())
                    has_swir_bands = any(f'Band{i}' in variables for i in range(4, 10))
                    if not has_swir_bands:
                        logger.warning(f"Supposed SWIR file {swir_file} doesn't contain SWIR bands")
                        # If the file doesn't have SWIR bands but has VNIR bands, it might be misidentified
                        has_vnir_bands = any(f'Band{i}' in variables for i in range(1, 4))
                        if has_vnir_bands:
                            logger.warning(f"The file {swir_file} actually contains VNIR bands")
                            # Swap the files if the "SWIR" file actually has VNIR bands and we don't have a VNIR file
                            if not vnir_file:
                                vnir_file = swir_file
                        swir_file = None
            except Exception as e:
                logger.warning(f"Error verifying SWIR file {swir_file}: {str(e)}")
        
        # Log what we found
        if vnir_file:
            logger.info(f"Selected VNIR file: {vnir_file}")
        else:
            logger.warning("No VNIR file found")
        
        if swir_file:
            logger.info(f"Selected SWIR file: {swir_file}")
        else:
            logger.warning("No SWIR file found")
        
        return vnir_file, swir_file
    



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
        # Get the directory containing the VNIR file
        dir_path = vnir_file.parent
        
        # Get the VNIR filename components
        vnir_stem = vnir_file.stem
        
        # Look for SWIR file using different matching patterns
        
        # Pattern 1: Check for file with timestamp one second later
        # Example: AST_09XT_00307032001075305_20250515100232_1324576.hdf → AST_09XT_00307032001075305_20250515100233_1324576.hdf
        if "_" in vnir_stem:
            parts = vnir_stem.split("_")
            if len(parts) >= 2:
                # Check if the second-to-last part looks like a timestamp
                timestamp_part_index = -2
                timestamp_part = parts[timestamp_part_index]
                if len(timestamp_part) == 14 and timestamp_part.isdigit():
                    # Try to increment the timestamp by 1 second
                    timestamp_seconds = int(timestamp_part[-6:])
                    new_seconds = (timestamp_seconds + 1) % 1000000  # Handle rollover
                    new_timestamp = timestamp_part[:-6] + f"{new_seconds:06d}"
                    
                    # Construct the potential SWIR filename
                    parts[timestamp_part_index] = new_timestamp
                    swir_stem = "_".join(parts)
                    swir_file = dir_path / f"{swir_stem}{vnir_file.suffix}"
                    
                    if swir_file.exists():
                        logger.info(f"Found matching SWIR file using timestamp+1 pattern: {swir_file}")
                        return swir_file
        
        # Pattern 2: Check for file with 'SWIR' in the name
        for file_path in dir_path.glob(f"*SWIR*{vnir_file.suffix}"):
            logger.info(f"Found potential SWIR file with 'SWIR' in name: {file_path}")
            return file_path
        
        # Pattern 3: Look for files with similar base name but different metadata tags
        base_parts = []
        # Extract the base parts of the filename (product ID, date, etc.)
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
                    return file_path
        
        # No matching SWIR file found
        logger.warning(f"No matching SWIR file found for {vnir_file}")
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
                
                success = self.extract_data()
                if not success:
                    self.update_status(
                        ProcessingStatus.FAILED, 
                        0, 
                        None,
                        "Data extraction failed"
                    )
                    return False
            
            # Find ASTER files
            vnir_file, swir_file = self.find_aster_files()
            
            # Check if we should use VNIR-only mode
            if not swir_file and not vnir_only_mode:
                logger.info("SWIR file not found, switching to VNIR-only mode")
                vnir_only_mode = True
            
            # Process in VNIR-only mode if specified or if SWIR is missing
            if vnir_only_mode:
                return self._process_vnir_only(vnir_file, options)
            
            # Check if both files are available for full processing
            if not (vnir_file and swir_file):
                self.update_status(
                    ProcessingStatus.FAILED, 
                    0, 
                    None,
                    "Could not find VNIR and SWIR files"
                )
                return False
            
            # Initialize L2 processor
            from .aster_l2_processor import ASTER_L2_Processor, SceneMetadata, GeographicBounds
            
            # For demo purposes, create dummy metadata
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
            
            self.update_status(
                ProcessingStatus.PROCESSING, 
                10, 
                ProcessingStages.EXTRACT,
                "Initializing processor"
            )
            
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
                
                minerals_dir = self.processed_dir / 'minerals'
                minerals_dir.mkdir(exist_ok=True)
                
                # Process mineral maps
                from .aster_l2_processor import MineralIndices
                
                for mineral in MineralIndices:
                    try:
                        processor.save_mineral_map(mineral, minerals_dir)
                    except Exception as e:
                        logger.error(f"Error processing {mineral.value} map: {str(e)}")
            
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
                
                alteration_dir = self.processed_dir / 'alteration'
                alteration_dir.mkdir(exist_ok=True)
                
                # Process alteration maps
                from enums import AlterationIndices
                
                for alteration in AlterationIndices:
                    try:
                        geological_mapper.save_alteration_map(alteration, alteration_dir)
                    except Exception as e:
                        logger.error(f"Error processing {alteration.value} map: {str(e)}")
            
            # Process gold pathfinder maps if requested
            if options.get('process_gold_pathfinders', True):
                self.update_status(
                    ProcessingStatus.PROCESSING, 
                    60, 
                    ProcessingStages.GOLD_PATHFINDER,
                    "Processing gold pathfinder maps"
                )
                
                gold_dir = self.processed_dir / 'gold_pathfinders'
                gold_dir.mkdir(exist_ok=True)
                
                # Process gold pathfinder maps
                from enums import GoldPathfinderIndices
                
                for pathfinder in GoldPathfinderIndices:
                    try:
                        # In a real system, this would use the actual gold pathfinder processor
                        processor.save_mineral_map(pathfinder, gold_dir)
                    except Exception as e:
                        logger.error(f"Error processing {pathfinder.value} map: {str(e)}")
            
            # Process RGB composites and band combinations
            if options.get('enhanced_visualization', True):
                self.update_status(
                    ProcessingStatus.PROCESSING, 
                    80, 
                    ProcessingStages.GEOLOGICAL_MAPPING,
                    "Creating RGB composites"
                )
                
                rgb_dir = self.processed_dir / 'rgb_composites'
                rgb_dir.mkdir(exist_ok=True)
                
                # Create RGB composites
                from enums import BandCombinations
                
                for combo in BandCombinations:
                    try:
                        geological_mapper.create_band_combination_map(combo, rgb_dir)
                    except Exception as e:
                        logger.error(f"Error creating {combo.value} band combination: {str(e)}")
            
            # Perform advanced analysis if requested
            if options.get('process_advanced_analysis', True):
                self.update_status(
                    ProcessingStatus.PROCESSING, 
                    90, 
                    ProcessingStages.ADVANCED_ANALYSIS,
                    "Performing advanced analysis"
                )
                
                analysis_dir = self.processed_dir / 'analysis'
                analysis_dir.mkdir(exist_ok=True)
                
                from .aster_advanced_analysis import ASTER_Advanced_Analysis
                
                advanced = ASTER_Advanced_Analysis(processor, geological_mapper)
                advanced.create_band_ratio_matrix(analysis_dir)
                advanced.extract_geological_features(analysis_dir)
                advanced.generate_mineral_potential_map(analysis_dir)
            
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
            
            # Update status to failed
            self.update_status(
                ProcessingStatus.FAILED, 
                0, 
                None,
                str(e)
            )
            
            return False



    def _process_vnir_only(self, vnir_file: Optional[Path], options: Dict) -> bool:
        """
        Process ASTER data in VNIR-only mode
        
        Parameters:
        -----------
        vnir_file : Optional[Path]
            Path to the VNIR file
        options : Dict
            Processing options
            
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
            return False
        
        try:
            # Update status
            self.update_status(
                ProcessingStatus.PROCESSING, 
                10, 
                ProcessingStages.EXTRACT,
                "Initializing VNIR-only processor"
            )
            
            # Import VNIR processor
            from processors.vnir_processor import VNIRProcessor
            
            # Create scene metadata
            from processors.aster_l2_processor import SceneMetadata, GeographicBounds
            
            # Try to read metadata from MET file
            metadata_bounds = None
            metadata_file = Path(str(vnir_file) + '.met')
            if metadata_file.exists():
                try:
                    # Extract coordinates from MET file
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
            
            # Use default bounds if metadata extraction failed
            if not metadata_bounds:
                metadata_bounds = GeographicBounds(
                    west=-2.67, east=-1.99, south=8.76, north=9.43
                )
            
            # Create metadata object
            scene_metadata = SceneMetadata(
                bounds=metadata_bounds,
                solar_azimuth=152.9561090000,
                solar_elevation=53.5193190000,
                cloud_cover=5.0,
                acquisition_date=datetime.datetime.now().strftime('%Y-%m-%d')
            )
            
            # Initialize VNIR processor
            vnir_processor = VNIRProcessor(
                vnir_file=str(vnir_file),
                metadata=scene_metadata
            )
            
            # Update progress
            self.update_status(
                ProcessingStatus.PROCESSING, 
                20, 
                ProcessingStages.MINERAL_MAPPING,
                "Processing VNIR bands"
            )
            
            # Create output directories
            rgb_dir = self.processed_dir / 'rgb_composites'
            rgb_dir.mkdir(exist_ok=True)
            
            minerals_dir = self.processed_dir / 'minerals'
            minerals_dir.mkdir(exist_ok=True)
            
            # Save individual bands if requested
            if options.get('process_minerals', True):
                try:
                    vnir_processor.save_band_as_geotiff(1, minerals_dir / "band1_blue.tif")
                    vnir_processor.save_band_as_geotiff(2, minerals_dir / "band2_green.tif")
                    vnir_processor.save_band_as_geotiff(3, minerals_dir / "band3_nir.tif")
                except Exception as e:
                    logger.error(f"Error saving VNIR bands: {str(e)}")
            
            # Update progress
            self.update_status(
                ProcessingStatus.PROCESSING, 
                40, 
                ProcessingStages.GEOLOGICAL_MAPPING,
                "Creating band composites"
            )
            
            # Create RGB composites if requested
            if options.get('enhanced_visualization', True):
                try:
                    vnir_processor.create_true_color_composite(rgb_dir / "true_color_composite.tif")
                    vnir_processor.create_false_color_composite(rgb_dir / "false_color_composite.tif")
                except Exception as e:
                    logger.error(f"Error creating composites: {str(e)}")
            
            # Update progress
            self.update_status(
                ProcessingStatus.PROCESSING, 
                60, 
                ProcessingStages.MINERAL_MAPPING,
                "Creating vegetation indices"
            )
            
            # Create NDVI map if requested
            if options.get('process_minerals', True):
                try:
                    vnir_processor.create_ndvi_map(minerals_dir / "ndvi_map.tif")
                except Exception as e:
                    logger.error(f"Error creating NDVI map: {str(e)}")
            
            # Update progress
            self.update_status(
                ProcessingStatus.PROCESSING, 
                80, 
                ProcessingStages.EXTRACT,
                "Updating metadata"
            )
            
            # Save metadata to processed directory
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
            
            # Add original filename if available
            raw_metadata_path = self.raw_dir / 'metadata.json'
            if raw_metadata_path.exists():
                try:
                    with open(raw_metadata_path, 'r') as f:
                        raw_metadata = json.load(f)
                        metadata['name'] = raw_metadata.get('original_filename', self.scene_id)
                        metadata['upload_date'] = raw_metadata.get('upload_date', datetime.datetime.now().isoformat())
                except Exception as e:
                    logger.error(f"Error reading raw metadata: {str(e)}")
                    metadata['name'] = self.scene_id
                    metadata['upload_date'] = datetime.datetime.now().isoformat()
            else:
                metadata['name'] = self.scene_id
                metadata['upload_date'] = datetime.datetime.now().isoformat()
            
            # Write metadata to file
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Create a thumbnail
            try:
                vnir_processor.create_thumbnail(self.processed_dir / 'thumbnail.png')
            except Exception as e:
                logger.error(f"Error creating thumbnail: {str(e)}")
            
            # Update status to completed
            self.update_status(
                ProcessingStatus.COMPLETED, 
                100, 
                None,
                "VNIR-only processing completed"
            )
            
            logger.info(f"VNIR-only processing completed for scene {self.scene_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error in VNIR-only processing: {str(e)}")
            self.update_status(
                ProcessingStatus.FAILED, 
                0, 
                None,
                f"Error in VNIR-only processing: {str(e)}"
            )
            return False



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
                
            # Look for the GPOLYGON section and coordinates
            gpolygon_pattern = r'GROUP\s+=\s+GPOLYGON(.*?)END_GROUP\s+=\s+GPOLYGON'
            gpolygon_match = re.search(gpolygon_pattern, content, re.DOTALL)
            
            if gpolygon_match:
                logger.info("Found GPOLYGON section")
                gpolygon_content = gpolygon_match.group(1)
                
                # Extract longitude values
                lon_pattern = r'GRINGPOINTLONGITUDE.*?VALUE\s+=\s+\((.*?)\)'
                lon_match = re.search(lon_pattern, gpolygon_content, re.DOTALL)
                lons = []
                if lon_match:
                    lon_str = lon_match.group(1)
                    try:
                        lons = [float(x.strip()) for x in lon_str.split(',')]
                    except Exception as e:
                        logger.error(f"Error parsing longitudes: {str(e)}")
                
                # Extract latitude values
                lat_pattern = r'GRINGPOINTLATITUDE.*?VALUE\s+=\s+\((.*?)\)'
                lat_match = re.search(lat_pattern, gpolygon_content, re.DOTALL)
                lats = []
                if lat_match:
                    lat_str = lat_match.group(1)
                    try:
                        lats = [float(x.strip()) for x in lat_str.split(',')]
                    except Exception as e:
                        logger.error(f"Error parsing latitudes: {str(e)}")
                
                # Calculate bounds
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
            return None 


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
        
        # Get options
        threshold = options.get('threshold', 0.7)
        pathfinders = options.get('pathfinders', [
            'gold_alteration', 'pyrite', 'arsenopyrite', 'advanced_argillic_gold'
        ])
        alterations = options.get('alterations', [
            'advanced_argillic', 'phyllic', 'silicification'
        ])
        
        # Initialize gold prospectivity mapper
        prospectivity_dir = self.processed_dir / 'prospectivity'
        prospectivity_dir.mkdir(exist_ok=True)
        
        from .gold_prospectivity_mapper import GoldProspectivityMapper
        
        mapper = GoldProspectivityMapper(output_directory=str(prospectivity_dir))
        
        # Add pathfinder maps
        gold_dir = self.processed_dir / 'gold_pathfinders'
        if gold_dir.exists():
            for pathfinder in pathfinders:
                pathfinder_map = gold_dir / f"{pathfinder}_map.tif"
                if pathfinder_map.exists():
                    mapper.add_mineral_map(pathfinder, str(pathfinder_map), weight=0.8)
        
        # Add alteration maps
        alteration_dir = self.processed_dir / 'alteration'
        if alteration_dir.exists():
            for alteration in alterations:
                alteration_map = alteration_dir / f"{alteration}_map.tif"
                if alteration_map.exists():
                    mapper.add_mineral_map(alteration, str(alteration_map), weight=0.6)
        
        # Generate prospectivity map
        prospectivity_map = mapper.generate_prospectivity_map("gold_prospectivity.tif")
        
        # Create visualization
        visualization = mapper.visualize_map("gold_prospectivity_visualization.png", 
                                          colormap="prospectivity", 
                                          title="Gold Prospectivity Map")
        
        # Extract high prospectivity areas
        high_areas = mapper.get_high_prospectivity_areas(threshold, "high_prospectivity.tif")
        
        # Export to GeoJSON for web visualization
        geojson = mapper.export_to_geojson(threshold, "high_prospectivity.geojson")
        
        return prospectivity_map
    
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
        
        # Create reports directory
        reports_dir = self.processed_dir / 'reports'
        reports_dir.mkdir(exist_ok=True)
        
        # Generate report filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = reports_dir / f"{self.scene_id}_{report_type}_{timestamp}.pdf"
        
        # In a real system, this would generate an actual report
        # For this demo, we'll just create a dummy file
        with open(report_file, 'w') as f:
            f.write(f"ASTER Analysis Report\n")
            f.write(f"Scene ID: {self.scene_id}\n")
            f.write(f"Report Type: {report_type}\n")
            f.write(f"Generated: {timestamp}\n")
            
            for key, value in options.items():
                f.write(f"{key}: {value}\n")
        
        return str(report_file)