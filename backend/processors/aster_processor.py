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
        
        # First, look for files with VNIR/SWIR in the name
        for file_path in self.extracted_dir.glob('**/*.hdf'):
            if 'vnir' in file_path.name.lower():
                vnir_file = file_path
            elif 'swir' in file_path.name.lower():
                swir_file = file_path
        
        # If not found, try to identify by file size (VNIR typically larger)
        if not (vnir_file and swir_file):
            hdf_files = list(self.extracted_dir.glob('**/*.hdf'))
            if len(hdf_files) >= 2:
                hdf_files.sort(key=lambda x: x.stat().st_size, reverse=True)
                vnir_file = hdf_files[0]
                swir_file = hdf_files[1]
        
        # For demo purposes, if still not found, use sample data or a placeholder
        if not (vnir_file and swir_file):
            logger.warning("Could not find VNIR and SWIR files in the extracted data. Using placeholders.")
            # Use the first HDF file for both
            hdf_files = list(self.extracted_dir.glob('**/*.hdf'))
            if hdf_files:
                vnir_file = hdf_files[0]
                swir_file = hdf_files[0]
        
        return vnir_file, swir_file
    
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
                
                self.extract_data()
            
            # Find ASTER files
            vnir_file, swir_file = self.find_aster_files()
            
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