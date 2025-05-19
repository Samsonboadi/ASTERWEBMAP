# aster_ai_integration.py
"""
ASTER AI Integration Module
===========================
This module integrates ASTER processing with AI-powered analysis to generate
comprehensive geological reports, combining mineral mapping with intelligent
interpretation.

Author: GIS Remote Sensing AI Team
"""

import os
import sys
import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from enum import Enum
import yaml

from .aster_vision_analyzer import ASTERVisionAnalyzer, VisionAnalysisType
from .aster_band_math import MapCombiner, TargetedMapGenerator, CombinationMethod
from utils.enhanced_logging import setup_logging, log_processing_operation, log_exception

# Set up logging
logger = logging.getLogger(__name__)

class ReportType(Enum):
    """Types of geological reports"""
    MINERAL_DISTRIBUTION = "mineral_distribution"
    ALTERATION_ANALYSIS = "alteration_analysis"
    STRUCTURAL_ANALYSIS = "structural_analysis"
    EXPLORATION_POTENTIAL = "exploration_potential"
    COMPREHENSIVE = "comprehensive"
class AIIntegrationConfig:
    """Configuration for AI integration"""
    
    def __init__(self, config_file: Optional[Union[str, Path]] = None):
        """
        Initialize configuration
        
        Parameters:
        -----------
        config_file : Optional[Union[str, Path]]
            Path to configuration file (YAML or JSON)
        """
        # Default configuration
        self.config = {
            "api": {
                "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
                "max_retries": 3,
                "retry_delay": 5,
                "timeout": 60
            },
            "processing": {
                "generate_enhanced_visualizations": True,
                "auto_generate_composite_maps": True,
                "auto_generate_reports": True,
                "save_intermediates": False
            },
            "report": {
                "default_type": "comprehensive",
                "include_recommendations": True,
                "generate_pdf": True,
                "include_histograms": True,
                "include_spatial_stats": True
            },
            "visualization": {
                "color_schemes": {
                    "mineral": "viridis",
                    "alteration": "plasma",
                    "composite": "inferno"
                },
                "dpi": 300,
                "add_annotations": True,
                "add_contours": True
            }
        }
        
        # Load configuration from file if provided
        if config_file:
            self.load_config(config_file)
    
    def load_config(self, config_file: Union[str, Path]) -> None:
        """
        Load configuration from file
        
        Parameters:
        -----------
        config_file : Union[str, Path]
            Path to configuration file (YAML or JSON)
        """
        config_file = Path(config_file)
        
        if not config_file.exists():
            logger.warning(f"Configuration file not found: {config_file}")
            return
        
        try:
            if config_file.suffix.lower() in ['.yaml', '.yml']:
                with open(config_file, 'r') as f:
                    file_config = yaml.safe_load(f)
            elif config_file.suffix.lower() == '.json':
                with open(config_file, 'r') as f:
                    file_config = json.load(f)
            else:
                logger.warning(f"Unsupported configuration file format: {config_file.suffix}")
                return
            
            # Update configuration with values from file
            self._update_dict(self.config, file_config)
            
            logger.info(f"Loaded configuration from {config_file}")
            
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            logger.exception("Detailed error:")
    
    def _update_dict(self, target: Dict, source: Dict) -> None:
        """
        Recursively update dictionary
        
        Parameters:
        -----------
        target : Dict
            Target dictionary to update
        source : Dict
            Source dictionary with new values
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._update_dict(target[key], value)
            else:
                target[key] = value
    
    def save_config(self, config_file: Union[str, Path]) -> None:
        """
        Save configuration to file
        
        Parameters:
        -----------
        config_file : Union[str, Path]
            Path to save configuration file
        """
        config_file = Path(config_file)
        
        try:
            # Create parent directory if it doesn't exist
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            if config_file.suffix.lower() in ['.yaml', '.yml']:
                with open(config_file, 'w') as f:
                    yaml.dump(self.config, f, default_flow_style=False)
            elif config_file.suffix.lower() == '.json':
                with open(config_file, 'w') as f:
                    json.dump(self.config, f, indent=2)
            else:
                logger.warning(f"Unsupported configuration file format: {config_file.suffix}")
                return
            
            logger.info(f"Saved configuration to {config_file}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
            logger.exception("Detailed error:")
    
    def get(self, section: str, key: str, default: Any = None) -> Any:
        """
        Get configuration value
        
        Parameters:
        -----------
        section : str
            Configuration section
        key : str
            Configuration key
        default : Any
            Default value if not found
            
        Returns:
        --------
        Any
            Configuration value
        """
        return self.config.get(section, {}).get(key, default)
    
    def set(self, section: str, key: str, value: Any) -> None:
        """
        Set configuration value
        
        Parameters:
        -----------
        section : str
            Configuration section
        key : str
            Configuration key
        value : Any
            Value to set
        """
        if section not in self.config:
            self.config[section] = {}
        
        self.config[section][key] = value
class ASTERAIProcessor:
    """
    Class to integrate ASTER processing with AI-powered analysis
    """
    
    def __init__(self, config: Optional[AIIntegrationConfig] = None):
        """
        Initialize the AI processor
        
        Parameters:
        -----------
        config : Optional[AIIntegrationConfig]
            Configuration for AI integration
        """
        self.config = config or AIIntegrationConfig()
        
        # Initialize AI analysis components
        api_key = self.config.get('api', 'openai_api_key')
        self.vision_analyzer = ASTERVisionAnalyzer(api_key=api_key)
        
        # Initialize band math components
        self.map_combiner = MapCombiner()
        self.targeted_map_generator = TargetedMapGenerator(self.map_combiner)
        
        logger.info("Initialized ASTER AI Processor")
    
    def process_scene(self, 
                     scene_dir: Union[str, Path],
                     output_dir: Union[str, Path],
                     report_type: ReportType = ReportType.COMPREHENSIVE) -> Dict:
        """
        Process an ASTER scene with AI analysis
        
        Parameters:
        -----------
        scene_dir : Union[str, Path]
            Directory containing processed ASTER scene
        output_dir : Union[str, Path]
            Directory to save AI analysis results
        report_type : ReportType
            Type of report to generate
            
        Returns:
        --------
        Dict
            Processing results and report information
        """
        scene_dir = Path(scene_dir)
        output_dir = Path(output_dir)
        
        # Get scene ID
        scene_id = scene_dir.name
        
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Processing scene {scene_id} with AI analysis")
        operation_start = log_processing_operation(
            logger, 
            "ai_scene_processing", 
            scene_id=scene_id,
            parameters={"report_type": report_type.value}
        )
        
        try:
            # Create AI analysis subdirectory
            ai_dir = output_dir / "ai_analysis"
            ai_dir.mkdir(exist_ok=True)
            
            # Step 1: Generate enhanced visualizations for analysis
            if self.config.get('processing', 'generate_enhanced_visualizations'):
                enhanced_maps = self._generate_enhanced_visualizations(scene_dir, ai_dir)
                logger.info(f"Generated {len(enhanced_maps)} enhanced visualizations")
            else:
                # Use existing visualizations if available
                enhanced_maps = self._find_existing_visualizations(scene_dir)
                logger.info(f"Found {len(enhanced_maps)} existing visualizations")
            
            # Step 2: Generate composite maps if enabled
            if self.config.get('processing', 'auto_generate_composite_maps'):
                composite_maps = self._generate_composite_maps(scene_dir, ai_dir)
                logger.info(f"Generated {len(composite_maps)} composite maps")
            else:
                composite_maps = []
            
            # Step 3: Perform AI analysis based on report type
            ai_analysis_results = self._perform_ai_analysis(
                scene_dir, 
                ai_dir, 
                enhanced_maps, 
                composite_maps,
                report_type
            )
            
            # Step 4: Generate comprehensive report
            report_path = self._generate_report(
                scene_dir,
                ai_dir,
                ai_analysis_results,
                enhanced_maps,
                composite_maps,
                report_type
            )
            
            # Record processing results
            results = {
                "scene_id": scene_id,
                "output_directory": str(ai_dir),
                "report_type": report_type.value,
                "enhanced_maps": enhanced_maps,
                "composite_maps": composite_maps,
                "ai_analysis_results": ai_analysis_results,
                "report_path": str(report_path)
            }
            
            # Save processing results as JSON
            results_path = ai_dir / f"{scene_id}_ai_processing_results.json"
            with open(results_path, 'w') as f:
                # Convert paths to strings for JSON serialization
                serializable_results = self._make_serializable(results)
                json.dump(serializable_results, f, indent=2)
            
            # Log completion
            log_processing_operation(
                logger, 
                "ai_scene_processing", 
                scene_id=scene_id, 
                start_time=operation_start
            )
            
            return results
            
        except Exception as e:
            log_exception(logger, "ai_scene_processing", e, scene_id=scene_id)
            raise RuntimeError(f"Error processing scene {scene_id}: {str(e)}")
    def _make_serializable(self, obj: Any) -> Any:
            """
            Make an object JSON serializable by converting paths to strings
            
            Parameters:
            -----------
            obj : Any
                Object to make serializable
                
            Returns:
            --------
            Any
                Serializable object
            """
            if isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: self._make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [self._make_serializable(item) for item in obj]
            elif hasattr(obj, 'value'):  # For Enum objects
                return obj.value
            else:
                return obj
        
        # Implementation of helper methods would go here
        # (These methods would handle the individual processing steps)
    
    def _generate_enhanced_visualizations(self, scene_dir: Path, output_dir: Path) -> Dict[str, Path]:
        """Generate enhanced visualizations for AI analysis"""
        # Implementation details omitted for brevity
        enhanced_maps = {}
        # Find and enhance mineral, alteration, and analysis maps
        return enhanced_maps
    
    def _find_existing_visualizations(self, scene_dir: Path) -> Dict[str, Path]:
        """Find existing visualizations in the scene directory"""
        # Implementation details omitted for brevity
        existing_maps = {}
        # Look for visualizations in minerals, alteration, and RGB directories
        return existing_maps
    
    def _generate_composite_maps(self, scene_dir: Path, output_dir: Path) -> Dict[str, Path]:
        """Generate composite maps for specific exploration targets"""
        # Implementation details omitted for brevity
        composite_maps = {}
        # Generate gold potential, base metal potential maps, etc.
        return composite_maps
    
    def _perform_ai_analysis(self, scene_dir: Path, output_dir: Path, 
                          enhanced_maps: Dict[str, Path], composite_maps: Dict[str, Path],
                          report_type: ReportType) -> Dict:
        """Perform AI analysis on maps"""
        # Implementation details omitted for brevity
        analysis_results = {}
        # Analyze individual maps and create combined analysis
        return analysis_results
    
    def _generate_report(self, scene_dir: Path, output_dir: Path, 
                    analysis_results: Dict, enhanced_maps: Dict[str, Path], 
                    composite_maps: Dict[str, Path], report_type: ReportType) -> Path:
        """Generate a comprehensive geological report"""
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        report_file = output_dir / f"{scene_dir.name}_{report_type.value}_report.html"
        
        try:
            # Extract scene metadata
            metadata_path = scene_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {"name": scene_dir.name}
            
            # Start creating HTML report
            html_content = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>{report_type.value.replace('_', ' ').title()} Report - {metadata.get('name', 'ASTER Scene')}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; color: #333; }}
                    h1, h2, h3 {{ color: #2c3e50; }}
                    .container {{ max-width: 1200px; margin: 0 auto; }}
                    .header {{ background-color: #3498db; color: white; padding: 20px; border-radius: 5px; }}
                    .section {{ margin: 20px 0; padding: 20px; background-color: #f9f9f9; border-radius: 5px; }}
                    .flex-container {{ display: flex; flex-wrap: wrap; gap: 20px; }}
                    .image-container {{ flex: 1; min-width: 300px; }}
                    img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }}
                    table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                    th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #ddd; }}
                    th {{ background-color: #f2f2f2; }}
                    .footer {{ font-size: 0.8em; color: #7f8c8d; margin-top: 50px; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>{report_type.value.replace('_', ' ').title()} Analysis Report</h1>
                        <p>Scene ID: {scene_dir.name}</p>
                        <p>Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    </div>
            """
            
            # Add scene information section
            html_content += f"""
            <div class="section">
                <h2>Scene Information</h2>
                <table>
                    <tr>
                        <th>Scene Name</th>
                        <td>{metadata.get('name', scene_dir.name)}</td>
                    </tr>
                    <tr>
                        <th>Acquisition Date</th>
                        <td>{metadata.get('acquisition_date', 'Unknown')}</td>
                    </tr>
                    <tr>
                        <th>Cloud Cover</th>
                        <td>{metadata.get('cloud_cover', 'Unknown')}%</td>
                    </tr>
                    <tr>
                        <th>Solar Azimuth</th>
                        <td>{metadata.get('solar_azimuth', 'Unknown')}</td>
                    </tr>
                    <tr>
                        <th>Solar Elevation</th>
                        <td>{metadata.get('solar_elevation', 'Unknown')}</td>
                    </tr>
                </table>
            </div>
            """
            
            # Add analysis results based on report type
            if report_type == ReportType.MINERAL_DISTRIBUTION:
                # Add mineral distribution section
                html_content += self._generate_mineral_section(scene_dir, output_dir, analysis_results, enhanced_maps)
            
            elif report_type == ReportType.ALTERATION_ANALYSIS:
                # Add alteration analysis section
                html_content += self._generate_alteration_section(scene_dir, output_dir, analysis_results, enhanced_maps)
            
            elif report_type == ReportType.STRUCTURAL_ANALYSIS:
                # Add structural analysis section
                html_content += self._generate_structural_section(scene_dir, output_dir, analysis_results, enhanced_maps)
            
            elif report_type == ReportType.EXPLORATION_POTENTIAL:
                # Add exploration potential section
                html_content += self._generate_exploration_section(scene_dir, output_dir, analysis_results, composite_maps)
            
            else:  # Comprehensive
                # Add all sections
                html_content += self._generate_mineral_section(scene_dir, output_dir, analysis_results, enhanced_maps)
                html_content += self._generate_alteration_section(scene_dir, output_dir, analysis_results, enhanced_maps)
                html_content += self._generate_structural_section(scene_dir, output_dir, analysis_results, enhanced_maps)
                html_content += self._generate_exploration_section(scene_dir, output_dir, analysis_results, composite_maps)
            
            # Add footer
            html_content += f"""
                    <div class="footer">
                        <p>Report generated by ASTER Web Explorer AI Analysis Module</p>
                        <p>© {datetime.datetime.now().year} ASTER Explorer</p>
                    </div>
                </div>
            </body>
            </html>
            """
            
            # Write HTML to file
            with open(report_file, 'w') as f:
                f.write(html_content)
            
            logger.info(f"Generated {report_type.value} report at {report_file}")
            return report_file
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            logger.exception("Detailed exception:")
            
            # Create a simple error report in case of failure
            with open(report_file, 'w') as f:
                f.write(f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Error Report</title>
                </head>
                <body>
                    <h1>Error Generating Report</h1>
                    <p>Scene ID: {scene_dir.name}</p>
                    <p>Report Type: {report_type.value}</p>
                    <p>Error: {str(e)}</p>
                </body>
                </html>
                """)
            
            return report_file

def main():
    """Main entry point for the ASTER AI Integration module"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ASTER AI Integration")
    parser.add_argument("--scene_dir", type=str, required=True, help="Directory containing processed ASTER scene")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save AI analysis results")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--report_type", type=str, default="comprehensive", 
                      choices=[t.value for t in ReportType], 
                      help="Type of report to generate")
    parser.add_argument("--api_key", type=str, help="OpenAI API key (overrides config file and env var)")
    
    args = parser.parse_args()
    
    # Setup logging
    log_dir = Path(args.output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(log_dir=log_dir, module_name="aster_ai_integration")
    
    # Create configuration
    config = AIIntegrationConfig(args.config)
    
    # Override API key if provided
    if args.api_key:
        config.set('api', 'openai_api_key', args.api_key)
    
    # Create AI processor
    ai_processor = ASTERAIProcessor(config)
    
    # Process scene
    try:
        ai_processor.process_scene(
            args.scene_dir,
            args.output_dir,
            ReportType(args.report_type)
        )
        
        logger.info(f"Completed AI analysis for {Path(args.scene_dir).name}")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        logger.exception("Detailed error:")
        sys.exit(1)

if __name__ == "__main__":
    main()
