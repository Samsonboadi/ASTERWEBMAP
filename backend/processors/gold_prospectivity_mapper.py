# backend/processors/gold_prospectivity_mapper.py

import os
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
import rasterio
from rasterio.features import shapes
from rasterio.transform import from_bounds
import geopandas as gpd
from shapely.geometry import shape, mapping
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import json

# Set up logging
logger = logging.getLogger(__name__)

class GoldProspectivityMapper:
    """
    Class for generating gold prospectivity maps by combining 
    multiple mineral and alteration indicators
    """
    
    def __init__(self, output_directory: Union[str, Path]):
        """
        Initialize gold prospectivity mapper
        
        Parameters:
        -----------
        output_directory : Union[str, Path]
            Directory to save outputs
        """
        self.output_dir = Path(output_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize maps and weights containers
        self.mineral_maps = {}
        self.mineral_weights = {}
        self.transform = None
        self.crs = None
        self.shape = None
        
        # Define the default colormap for visualizations
        self.colormaps = {
            'prospectivity': LinearSegmentedColormap.from_list(
                'prospectivity', 
                [(0, 0, 0.5), (0, 0, 1), (0, 1, 1), (0, 1, 0), (1, 1, 0), (1, 0, 0)], 
                N=256
            ),
            'high': LinearSegmentedColormap.from_list(
                'high_prospectivity', 
                [(1, 0.8, 0.8), (1, 0, 0)], 
                N=128
            )
        }
        
        logger.info("Initialized GoldProspectivityMapper")
    
    def add_mineral_map(self, 
                       name: str, 
                       map_file: Union[str, Path],
                       weight: float = 1.0,
                       band_index: int = 1) -> None:
        """
        Add a mineral or alteration map to the prospectivity model
        
        Parameters:
        -----------
        name : str
            Name of the mineral or alteration indicator
        map_file : Union[str, Path]
            Path to the GeoTIFF file
        weight : float
            Weight for this indicator (0-1)
        band_index : int
            Band to use from the file (typically 1)
        """
        try:
            map_file = Path(map_file)
            
            if not map_file.exists():
                raise FileNotFoundError(f"Map file not found: {map_file}")
            
            with rasterio.open(map_file) as src:
                # Check if the band exists
                if band_index > src.count:
                    logger.warning(f"Band {band_index} not available in {map_file.name}, using band 1")
                    band_index = 1
                
                # Read data
                data = src.read(band_index)
                
                # Store the CRS and transform if not already set
                if self.transform is None:
                    self.transform = src.transform
                    self.crs = src.crs
                
                # Store the shape if not already set
                if self.shape is None:
                    self.shape = data.shape
                elif data.shape != self.shape:
                    # Resample if shapes don't match
                    logger.warning(f"Shape mismatch: {data.shape} vs {self.shape}, resampling required")
                    # For simplicity, we'll skip resampling here but it would be needed in production
                
                # Replace no data values with NaN
                if src.nodata is not None:
                    data = data.astype(np.float32)
                    data[data == src.nodata] = np.nan
                
                # Normalize to 0-1 if needed
                if np.nanmax(data) > 1:
                    data = (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))
            
            # Store the map and weight
            self.mineral_maps[name] = data
            self.mineral_weights[name] = weight
            
            logger.info(f"Added mineral map: {name} from {map_file.name} with weight {weight}")
            
        except Exception as e:
            logger.error(f"Error adding mineral map {name}: {str(e)}")
            raise
    
    def generate_prospectivity_map(self, output_filename: str) -> str:
        """
        Generate the gold prospectivity map
        
        Parameters:
        -----------
        output_filename : str
            Filename for the output prospectivity map
            
        Returns:
        --------
        str
            Path to the generated prospectivity map
        """
        try:
            if not self.mineral_maps:
                raise ValueError("No mineral maps have been added")
            
            logger.info(f"Generating prospectivity map with {len(self.mineral_maps)} indicators")
            
            # Calculate weights
            total_weight = sum(self.mineral_weights.values())
            normalized_weights = {name: weight/total_weight 
                               for name, weight in self.mineral_weights.items()}
            
            # Initialize result map
            result = np.zeros(self.shape, dtype=np.float32)
            
            # Apply weighted sum
            for name, data in self.mineral_maps.items():
                weight = normalized_weights[name]
                # Handle NaN values
                valid_mask = ~np.isnan(data)
                result[valid_mask] += data[valid_mask] * weight
            
            # Scale result to 0-1
            if np.any(result > 0):
                max_val = np.nanmax(result)
                if max_val > 0:
                    result = result / max_val
            
            # Save the result
            output_file = self.output_dir / output_filename
            
            with rasterio.open(
                output_file,
                'w',
                driver='GTiff',
                height=self.shape[0],
                width=self.shape[1],
                count=1,
                dtype=rasterio.float32,
                crs=self.crs,
                transform=self.transform,
                nodata=np.nan
            ) as dst:
                dst.write(result, 1)
                dst.update_tags(
                    indicators=','.join(self.mineral_maps.keys()),
                    weights=','.join([f"{name}:{weight}" for name, weight 
                                    in self.mineral_weights.items()]),
                    description='Gold Prospectivity Map',
                    creation_date=datetime.datetime.now().isoformat()
                )
            
            logger.info(f"Saved prospectivity map to {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Error generating prospectivity map: {str(e)}")
            raise
    
    def visualize_map(self, 
                    output_filename: str,
                    colormap: str = 'prospectivity',
                    title: str = 'Gold Prospectivity Map') -> str:
        """
        Create a visualization of the prospectivity map
        
        Parameters:
        -----------
        output_filename : str
            Filename for the output visualization
        colormap : str
            Name of the colormap to use
        title : str
            Title for the visualization
            
        Returns:
        --------
        str
            Path to the generated visualization
        """
        try:
            # Find the most recent prospectivity map
            prospectivity_files = list(self.output_dir.glob('*prospectivity*.tif'))
            if not prospectivity_files:
                raise FileNotFoundError("No prospectivity map found to visualize")
            
            # Sort by modification time, newest first
            prospectivity_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            prospectivity_file = prospectivity_files[0]
            
            with rasterio.open(prospectivity_file) as src:
                prospectivity_data = src.read(1)
                
                # Create the visualization
                plt.figure(figsize=(12, 8))
                
                # Get colormap
                if colormap not in self.colormaps:
                    colormap = 'prospectivity'
                cmap = self.colormaps[colormap]
                
                # Create visualization
                plt.imshow(prospectivity_data, cmap=cmap)
                plt.colorbar(label='Prospectivity Index')
                plt.title(title, fontsize=16)
                
                # Add indicators as legend
                if self.mineral_maps:
                    legend_text = "Indicators:\n" + "\n".join(
                        [f"- {name} (w={self.mineral_weights[name]:.2f})" 
                         for name in self.mineral_maps.keys()]
                    )
                    plt.figtext(0.02, 0.02, legend_text, fontsize=10, 
                                bbox=dict(facecolor='white', alpha=0.8))
                
                # Save the figure
                output_file = self.output_dir / output_filename
                plt.tight_layout()
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"Saved visualization to {output_file}")
                return str(output_file)
                
        except Exception as e:
            logger.error(f"Error visualizing prospectivity map: {str(e)}")
            raise
    
    def get_high_prospectivity_areas(self, 
                                   threshold: float = 0.7,
                                   output_filename: str = "high_prospectivity.tif") -> str:
        """
        Extract high prospectivity areas based on a threshold
        
        Parameters:
        -----------
        threshold : float
            Threshold value (0-1) for high prospectivity
        output_filename : str
            Filename for the output high prospectivity map
            
        Returns:
        --------
        str
            Path to the generated high prospectivity map
        """
        try:
            # Find the most recent prospectivity map
            prospectivity_files = list(self.output_dir.glob('*prospectivity*.tif'))
            if not prospectivity_files:
                raise FileNotFoundError("No prospectivity map found")
            
            # Sort by modification time, newest first
            prospectivity_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            prospectivity_file = prospectivity_files[0]
            
            with rasterio.open(prospectivity_file) as src:
                prospectivity_data = src.read(1)
                
                # Apply threshold
                high_areas = prospectivity_data >= threshold
                
                # Create binary mask
                mask = np.zeros_like(prospectivity_data)
                mask[high_areas] = 1
                
                # Save as GeoTIFF
                output_file = self.output_dir / output_filename
                
                with rasterio.open(
                    output_file,
                    'w',
                    driver='GTiff',
                    height=mask.shape[0],
                    width=mask.shape[1],
                    count=1,
                    dtype=rasterio.uint8,
                    crs=src.crs,
                    transform=src.transform,
                    nodata=0
                ) as dst:
                    dst.write(mask.astype(rasterio.uint8), 1)
                    dst.update_tags(
                        description=f'High Prospectivity Areas (threshold={threshold})',
                        threshold=str(threshold)
                    )
                
                logger.info(f"Saved high prospectivity areas to {output_file}")
                return str(output_file)
                
        except Exception as e:
            logger.error(f"Error extracting high prospectivity areas: {str(e)}")
            raise
    
    def export_to_geojson(self, 
                        threshold: float = 0.7,
                        output_filename: str = "high_prospectivity.geojson") -> str:
        """
        Export high prospectivity areas as GeoJSON
        
        Parameters:
        -----------
        threshold : float
            Threshold value (0-1) for high prospectivity
        output_filename : str
            Filename for the output GeoJSON
            
        Returns:
        --------
        str
            Path to the generated GeoJSON file
        """
        try:
            # Find the most recent prospectivity map
            prospectivity_files = list(self.output_dir.glob('*prospectivity*.tif'))
            if not prospectivity_files:
                raise FileNotFoundError("No prospectivity map found")
            
            # Sort by modification time, newest first
            prospectivity_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            prospectivity_file = prospectivity_files[0]
            
            with rasterio.open(prospectivity_file) as src:
                prospectivity_data = src.read(1)
                
                # Apply thresholds for different categories
                high_mask = prospectivity_data >= threshold
                medium_mask = (prospectivity_data >= threshold/2) & (prospectivity_data < threshold)
                low_mask = (prospectivity_data >= threshold/4) & (prospectivity_data < threshold/2)
                
                # Create GeoJSON features for each category
                results = {
                    'high': [],
                    'medium': [],
                    'low': []
                }
                
                # Process high areas
                high_mask = high_mask.astype(rasterio.uint8)
                for geom, value in shapes(high_mask, mask=high_mask.astype(bool), transform=src.transform):
                    feature = {
                        'type': 'Feature',
                        'geometry': geom,
                        'properties': {
                            'category': 'high',
                            'confidence': float(value),
                            'value': float(threshold)
                        }
                    }
                    results['high'].append(feature)
                
                # Process medium areas
                medium_mask = medium_mask.astype(rasterio.uint8)
                for geom, value in shapes(medium_mask, mask=medium_mask.astype(bool), transform=src.transform):
                    feature = {
                        'type': 'Feature',
                        'geometry': geom,
                        'properties': {
                            'category': 'medium',
                            'confidence': float(value),
                            'value': float(threshold/2)
                        }
                    }
                    results['medium'].append(feature)
                
                # Process low areas
                low_mask = low_mask.astype(rasterio.uint8)
                for geom, value in shapes(low_mask, mask=low_mask.astype(bool), transform=src.transform):
                    feature = {
                        'type': 'Feature',
                        'geometry': geom,
                        'properties': {
                            'category': 'low',
                            'confidence': float(value),
                            'value': float(threshold/4)
                        }
                    }
                    results['low'].append(feature)
                
                # Create GeoJSON collection for each category
                for category in results:
                    if results[category]:
                        results[category] = {
                            'type': 'FeatureCollection',
                            'features': results[category]
                        }
                
                # Save as GeoJSON
                output_file = self.output_dir / output_filename
                with open(output_file, 'w') as f:
                    json.dump(results, f)
                
                logger.info(f"Saved prospectivity areas as GeoJSON to {output_file}")
                return str(output_file)
                
        except Exception as e:
            logger.error(f"Error exporting to GeoJSON: {str(e)}")
            raise