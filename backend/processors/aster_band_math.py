# aster_band_math.py
"""
ASTER Band Math Module
======================
This module provides functionality for combining mineral and alteration maps
using band math operations to generate composite maps highlighting areas where
multiple mineralogical or alteration indicators overlap.

Author: GIS Remote Sensing
"""

import logging
import numpy as np
import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Callable
from enum import Enum
import rasterio
from rasterio.transform import Affine
from rasterio.crs import CRS
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Set up logging
logger = logging.getLogger(__name__)

class CombinationMethod(Enum):
    """Methods for combining mineral maps"""
    ADD = "add"                   # Simple addition
    MULTIPLY = "multiply"         # Multiplication (AND-like)
    WEIGHTED_ADD = "weighted_add" # Weighted addition
    MAX = "max"                   # Maximum value (OR-like)
    MIN = "min"                   # Minimum value
    THRESHOLD = "threshold"       # Binary threshold then add
    FUZZY_OR = "fuzzy_or"         # Fuzzy OR operation
    FUZZY_AND = "fuzzy_and"       # Fuzzy AND operation
    CUSTOM = "custom"             # Custom function

class MapCombiner:
    """
    Class for combining multiple mineral and alteration maps
    to generate composite maps highlighting areas of interest
    """
    
    def __init__(self, no_data_value: float = -9999):
        """
        Initialize the map combiner
        
        Parameters:
        -----------
        no_data_value : float
            Value to use for no data pixels
        """
        self.no_data_value = no_data_value
        logger.info(f"Initialized MapCombiner with no_data_value={no_data_value}")
    
    def combine_maps(self, 
                    map_files: List[Union[str, Path]],
                    output_file: Union[str, Path],
                    method: CombinationMethod = CombinationMethod.ADD,
                    weights: Optional[List[float]] = None,
                    threshold: float = 0.5,
                    band_index: int = 1,
                    custom_function: Optional[Callable] = None,
                    create_visualization: bool = True,
                    labels: Optional[List[str]] = None) -> Dict:
        """
        Combine multiple mineral or alteration maps into a composite map
        
        Parameters:
        -----------
        map_files : List[Union[str, Path]]
            List of GeoTIFF files containing mineral or alteration maps
        output_file : Union[str, Path]
            Path to save the output composite map
        method : CombinationMethod
            Method for combining the maps
        weights : List[float], optional
            Weights for each input map (for WEIGHTED_ADD method)
        threshold : float
            Threshold value for the THRESHOLD method
        band_index : int
            Band index to use from each input map (typically 1 for mineral index)
        custom_function : Callable, optional
            Custom function for CUSTOM method that takes a list of arrays and returns a combined array
        create_visualization : bool
            Whether to create a PNG visualization of the composite map
        labels : List[str], optional
            Labels for each input map (for visualization)
            
        Returns:
        --------
        Dict
            Statistics and metadata about the combined map
        """
        map_files = [Path(f) for f in map_files]
        output_file = Path(output_file)
        
        # Validate inputs
        if not map_files:
            logger.error("No input maps provided")
            raise ValueError("No input maps provided")
        
        if method == CombinationMethod.WEIGHTED_ADD and (weights is None or len(weights) != len(map_files)):
            logger.error(f"WEIGHTED_ADD method requires weights for each map (got {len(weights) if weights else 0}, expected {len(map_files)})")
            raise ValueError(f"WEIGHTED_ADD method requires weights for each map")
        
        if method == CombinationMethod.CUSTOM and custom_function is None:
            logger.error("CUSTOM method requires a custom_function")
            raise ValueError("CUSTOM method requires a custom_function")
        
        # Read input maps
        logger.info(f"Reading {len(map_files)} input maps...")
        
        map_data = []
        reference_profile = None
        map_names = []
        
        for i, map_file in enumerate(map_files):
            try:
                with rasterio.open(map_file) as src:
                    # Check if the requested band exists
                    if band_index > src.count:
                        logger.warning(f"Band {band_index} not available in {map_file.name}, using band 1 instead")
                        band_idx = 1
                    else:
                        band_idx = band_index
                    
                    # Read data
                    data = src.read(band_idx)
                    
                    # Save the reference profile from the first valid file
                    if reference_profile is None:
                        reference_profile = src.profile.copy()
                        reference_profile.update(count=1)  # Output will be single band
                    
                    # Mask no data values
                    mask = data == src.nodata if src.nodata is not None else np.isnan(data)
                    data = np.ma.masked_array(data, mask=mask)
                    
                    # Record map information
                    map_name = map_file.stem.split('_')[0] if '_' in map_file.stem else map_file.stem
                    map_names.append(map_name)
                    
                    # Store data
                    map_data.append(data)
                    
                    logger.info(f"Read map {i+1}/{len(map_files)}: {map_file.name} (shape: {data.shape})")
                    
            except Exception as e:
                logger.error(f"Error reading {map_file.name}: {str(e)}")
                raise RuntimeError(f"Error reading {map_file.name}: {str(e)}")
        # Check if all maps have the same shape
        shapes = [data.shape for data in map_data]
        if len(set(shapes)) > 1:
            logger.error(f"Input maps have different shapes: {shapes}")
            raise ValueError(f"Input maps have different shapes")
        
        # Ensure we have a reference profile for output
        if reference_profile is None:
            logger.error("Could not determine reference profile from input maps")
            raise RuntimeError("Could not determine reference profile from input maps")
        
        # Combine maps based on the selected method
        logger.info(f"Combining maps using method: {method.value}")
        
        try:
            if method == CombinationMethod.ADD:
                combined = self._add_maps(map_data)
            elif method == CombinationMethod.MULTIPLY:
                combined = self._multiply_maps(map_data)
            elif method == CombinationMethod.WEIGHTED_ADD:
                combined = self._weighted_add_maps(map_data, weights)
            elif method == CombinationMethod.MAX:
                combined = self._max_maps(map_data)
            elif method == CombinationMethod.MIN:
                combined = self._min_maps(map_data)
            elif method == CombinationMethod.THRESHOLD:
                combined = self._threshold_maps(map_data, threshold)
            elif method == CombinationMethod.FUZZY_OR:
                combined = self._fuzzy_or_maps(map_data)
            elif method == CombinationMethod.FUZZY_AND:
                combined = self._fuzzy_and_maps(map_data)
            elif method == CombinationMethod.CUSTOM:
                combined = custom_function(map_data)
            else:
                logger.error(f"Unsupported combination method: {method}")
                raise ValueError(f"Unsupported combination method: {method}")
                
            # Calculate statistics
            valid_mask = ~np.ma.getmaskarray(combined)
            if np.any(valid_mask):
                stats = {
                    'min': float(np.min(combined[valid_mask])),
                    'max': float(np.max(combined[valid_mask])),
                    'mean': float(np.mean(combined[valid_mask])),
                    'std': float(np.std(combined[valid_mask])),
                    'valid_pixels': int(np.sum(valid_mask)),
                    'total_pixels': int(valid_mask.size)
                }
            else:
                stats = {
                    'min': None,
                    'max': None,
                    'mean': None,
                    'std': None,
                    'valid_pixels': 0,
                    'total_pixels': int(valid_mask.size)
                }
            
            # Fill masked values with no_data for saving
            combined_data = combined.filled(self.no_data_value)
            
            # Save combined map
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            reference_profile.update(
                dtype=rasterio.float32,
                nodata=self.no_data_value
            )
            
            with rasterio.open(output_file, 'w', **reference_profile) as dst:
                dst.write(combined_data.astype(rasterio.float32), 1)
                
                # Add metadata
                dst.update_tags(
                    combination_method=method.value,
                    input_maps=','.join(map_names),
                    creation_timestamp=datetime.datetime.now().isoformat()
                )
                
                if method == CombinationMethod.WEIGHTED_ADD and weights:
                    dst.update_tags(weights=','.join([str(w) for w in weights]))
                
                if method == CombinationMethod.THRESHOLD:
                    dst.update_tags(threshold=str(threshold))
            
            logger.info(f"Saved combined map to {output_file}")
            # Create visualization if requested
            if create_visualization:
                viz_file = output_file.with_suffix('.png')
                self._create_visualization(
                    combined, 
                    viz_file, 
                    title=f"Composite Map ({method.value})", 
                    input_labels=labels if labels else map_names
                )
                logger.info(f"Created visualization at {viz_file}")
            
            # Return statistics and metadata
            result = {
                'output_file': str(output_file),
                'method': method.value,
                'input_maps': list(map(str, map_files)),
                'input_map_names': map_names,
                'statistics': stats
            }
            
            if method == CombinationMethod.WEIGHTED_ADD and weights:
                result['weights'] = weights
            
            if method == CombinationMethod.THRESHOLD:
                result['threshold'] = threshold
            
            if create_visualization:
                result['visualization'] = str(viz_file)
            
            return result
            
        except Exception as e:
            logger.error(f"Error combining maps: {str(e)}")
            logger.exception("Detailed error:")
            raise RuntimeError(f"Error combining maps: {str(e)}")
    
    def _add_maps(self, map_data: List[np.ma.MaskedArray]) -> np.ma.MaskedArray:
        """
        Add maps together
        
        Parameters:
        -----------
        map_data : List[np.ma.MaskedArray]
            List of map data arrays
            
        Returns:
        --------
        np.ma.MaskedArray
            Combined map
        """
        # Start with zeros
        result = np.ma.zeros_like(map_data[0])
        
        # Add each map
        for data in map_data:
            result += data
        
        return result
    
    def _multiply_maps(self, map_data: List[np.ma.MaskedArray]) -> np.ma.MaskedArray:
        """
        Multiply maps together (AND-like operation)
        
        Parameters:
        -----------
        map_data : List[np.ma.MaskedArray]
            List of map data arrays
            
        Returns:
        --------
        np.ma.MaskedArray
            Combined map
        """
        # Start with ones
        result = np.ma.ones_like(map_data[0])
        
        # Multiply each map
        for data in map_data:
            result *= data
        
        return result
    def _weighted_add_maps(self, 
                            map_data: List[np.ma.MaskedArray], 
                            weights: List[float]) -> np.ma.MaskedArray:
            """
            Add maps with weights
            
            Parameters:
            -----------
            map_data : List[np.ma.MaskedArray]
                List of map data arrays
            weights : List[float]
                Weights for each map
                
            Returns:
            --------
            np.ma.MaskedArray
                Combined map
            """
            # Start with zeros
            result = np.ma.zeros_like(map_data[0])
            
            # Add each map with its weight
            for data, weight in zip(map_data, weights):
                result += data * weight
            
            # Normalize by sum of weights
            result /= sum(weights)
            
            return result
    
    def _max_maps(self, map_data: List[np.ma.MaskedArray]) -> np.ma.MaskedArray:
        """
        Take maximum value at each pixel (OR-like operation)
        
        Parameters:
        -----------
        map_data : List[np.ma.MaskedArray]
            List of map data arrays
            
        Returns:
        --------
        np.ma.MaskedArray
            Combined map
        """
        # Stack arrays along a new axis
        stacked = np.ma.stack(map_data, axis=0)
        
        # Take maximum along the stack axis
        result = np.ma.max(stacked, axis=0)
        
        return result
    
    def _min_maps(self, map_data: List[np.ma.MaskedArray]) -> np.ma.MaskedArray:
        """
        Take minimum value at each pixel
        
        Parameters:
        -----------
        map_data : List[np.ma.MaskedArray]
            List of map data arrays
            
        Returns:
        --------
        np.ma.MaskedArray
            Combined map
        """
        # Stack arrays along a new axis
        stacked = np.ma.stack(map_data, axis=0)
        
        # Take minimum along the stack axis
        result = np.ma.min(stacked, axis=0)
        
        return result
    def _threshold_maps(self, 
                        map_data: List[np.ma.MaskedArray], 
                        threshold: float) -> np.ma.MaskedArray:
            """
            Apply threshold to each map and then add
            
            Parameters:
            -----------
            map_data : List[np.ma.MaskedArray]
                List of map data arrays
            threshold : float
                Threshold value
                
            Returns:
            --------
            np.ma.MaskedArray
                Combined map
            """
            # Start with zeros
            result = np.ma.zeros_like(map_data[0])
            
            # Threshold each map and add
            for data in map_data:
                binary = np.ma.masked_array(
                    (data > threshold).astype(np.float32),
                    mask=data.mask
                )
                result += binary
            
            return result
    
    def _fuzzy_or_maps(self, map_data: List[np.ma.MaskedArray]) -> np.ma.MaskedArray:
        """
        Apply fuzzy OR operation (probabilistic sum)
        
        Parameters:
        -----------
        map_data : List[np.ma.MaskedArray]
            List of map data arrays
            
        Returns:
        --------
        np.ma.MaskedArray
            Combined map
        """
        # Start with zeros
        result = np.ma.zeros_like(map_data[0])
        
        # Apply fuzzy OR: 1 - product(1 - x)
        complement = np.ma.ones_like(result)
        
        for data in map_data:
            # Ensure data is in [0, 1] range
            normalized_data = np.ma.clip(data, 0, 1)
            complement *= (1 - normalized_data)
        
        result = 1 - complement
        
        return result
    
    def _fuzzy_and_maps(self, map_data: List[np.ma.MaskedArray]) -> np.ma.MaskedArray:
        """
        Apply fuzzy AND operation (minimum)
        
        Parameters:
        -----------
        map_data : List[np.ma.MaskedArray]
            List of map data arrays
            
        Returns:
        --------
        np.ma.MaskedArray
            Combined map
        """
        # Ensure data is in [0, 1] range
        normalized_data = [np.ma.clip(data, 0, 1) for data in map_data]
        
        # Stack arrays along a new axis
        stacked = np.ma.stack(normalized_data, axis=0)
        
        # Take minimum along the stack axis (fuzzy AND)
        result = np.ma.min(stacked, axis=0)
        
        return result



    def _create_visualization(self, 
                           data: np.ma.MaskedArray, 
                           output_file: Path,
                           title: str = "Composite Map",
                           input_labels: List[str] = None) -> None:
        """
        Create visualization of the combined map
        
        Parameters:
        -----------
        data : np.ma.MaskedArray
            Combined map data
        output_file : Path
            Path to save the visualization
        title : str
            Title for the visualization
        input_labels : List[str], optional
            Labels for input maps for the legend
        """
        try:
            # Create figure
            plt.figure(figsize=(10, 8))
            
            # Create custom colormap from white to red
            colors = [(1, 1, 1), (1, 0.8, 0), (1, 0, 0)]
            cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)
            
            # Plot the data
            masked_data = np.ma.masked_where(data.mask, data)
            plt.imshow(masked_data, cmap=cmap)
            cbar = plt.colorbar(label="Composite Score")
            
            # Add title and labels
            plt.title(title, fontsize=16)
            plt.xlabel("X (pixels)", fontsize=12)
            plt.ylabel("Y (pixels)", fontsize=12)
            
            # Add legend for input maps if provided
            if input_labels:
                legend_text = "Input Maps:\n" + "\n".join([f"- {label}" for label in input_labels])
                plt.figtext(0.02, 0.02, legend_text, fontsize=10, 
                            bbox=dict(facecolor='white', alpha=0.8))
            
            # Save figure
            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
            logger.exception("Detailed error:")


class TargetedMapGenerator:
    """
    Generate targeted maps for specific exploration objectives
    by combining multiple mineral and alteration indicators
    """
    
    def __init__(self, map_combiner: Optional[MapCombiner] = None):
        """
        Initialize the targeted map generator
        
        Parameters:
        -----------
        map_combiner : MapCombiner, optional
            MapCombiner instance to use (creates a new one if None)
        """
        self.map_combiner = map_combiner or MapCombiner()
        logger.info("Initialized TargetedMapGenerator")
        
    def generate_gold_potential_map(self,
                                  base_dir: Union[str, Path],
                                  output_file: Union[str, Path],
                                  scene_id: Optional[str] = None) -> Dict:
        """
        Generate a gold potential map by combining multiple indicators
        
        Parameters:
        -----------
        base_dir : Union[str, Path]
            Base directory containing processed ASTER scene data
        output_file : Union[str, Path]
            Path to save the output gold potential map
        scene_id : str, optional
            Scene ID (if not provided, will try to determine from base_dir)
            
        Returns:
        --------
        Dict
            Statistics and metadata about the generated map
        """
        base_dir = Path(base_dir)
        output_file = Path(output_file)
        
        if scene_id is None:
            scene_id = base_dir.name
        
        logger.info(f"Generating gold potential map for scene {scene_id}")
        
        try:
            # Define indicators for gold potential
            mineral_indicators = [
                "alunite",
                "kaolinite",
                "sericite",
                "illite",
                "pyrite",
                "iron_oxide"
            ]
            
            alteration_indicators = [
                "advanced_argillic",
                "argillic",
                "phyllic"
            ]
            
            pathfinder_indicators = [
                "gold_alteration",
                "quartz_adularia",
                "arsenopyrite",
                "silica"
            ]
            
            # Find available maps
            mineral_maps = []
            for indicator in mineral_indicators:
                potential_paths = list(base_dir.glob(f"minerals/{indicator}_map.tif"))
                if potential_paths:
                    mineral_maps.append(potential_paths[0])
            
            alteration_maps = []
            for indicator in alteration_indicators:
                potential_paths = list(base_dir.glob(f"alteration/{indicator}_map.tif"))
                if potential_paths:
                    alteration_maps.append(potential_paths[0])
            
            pathfinder_maps = []
            for indicator in pathfinder_indicators:
                potential_paths = list(base_dir.glob(f"gold_pathfinders/{indicator}_map.tif"))
                if potential_paths:
                    pathfinder_maps.append(potential_paths[0])
            
            # Check if we have enough maps
            all_maps = mineral_maps + alteration_maps + pathfinder_maps
            
            if len(all_maps) < 2:
                logger.error(f"Not enough indicator maps found for gold potential (found {len(all_maps)})")
                raise ValueError(f"Not enough indicator maps found for gold potential")
            
            logger.info(f"Found {len(mineral_maps)} mineral maps, {len(alteration_maps)} alteration maps, "
                        f"and {len(pathfinder_maps)} pathfinder maps")
            
            # Define weights based on map type
            weights = []
            for map_path in all_maps:
                if "gold_pathfinders" in str(map_path):
                    weights.append(0.4)  # Higher weight for direct gold pathfinders
                elif "alteration" in str(map_path):
                    weights.append(0.3)  # Medium weight for alteration
                else:
                    weights.append(0.2)  # Lower weight for general minerals
            
            # Normalize weights
            sum_weights = sum(weights)
            weights = [w / sum_weights for w in weights]
            
            # Define labels for visualization
            labels = [path.stem.split('_')[0] for path in all_maps]
            
            # Create output directory if needed
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Combine maps
            result = self.map_combiner.combine_maps(
                map_files=all_maps,
                output_file=output_file,
                method=CombinationMethod.WEIGHTED_ADD,
                weights=weights,
                create_visualization=True,
                labels=labels
            )
            
            logger.info(f"Generated gold potential map: {output_file}")
            return result
            
        except Exception as e:
            logger.error(f"Error generating gold potential map: {str(e)}")
            logger.exception("Detailed error:")
            raise RuntimeError(f"Error generating gold potential map: {str(e)}")
    
    def generate_base_metal_potential_map(self,
                                       base_dir: Union[str, Path],
                                       output_file: Union[str, Path],
                                       scene_id: Optional[str] = None) -> Dict:
        """
        Generate a base metal potential map by combining multiple indicators
        
        Parameters:
        -----------
        base_dir : Union[str, Path]
            Base directory containing processed ASTER scene data
        output_file : Union[str, Path]
            Path to save the output base metal potential map
        scene_id : str, optional
            Scene ID (if not provided, will try to determine from base_dir)
            
        Returns:
        --------
        Dict
            Statistics and metadata about the generated map
        """
        base_dir = Path(base_dir)
        output_file = Path(output_file)
        
        if scene_id is None:
            scene_id = base_dir.name
        
        logger.info(f"Generating base metal potential map for scene {scene_id}")
        
        try:
            # Define indicators for base metal potential
            mineral_indicators = [
                "chlorite",
                "epidote",
                "pyrite",
                "chalcopyrite",
                "sphalerite",
                "galena"
            ]
            
            alteration_indicators = [
                "propylitic",
                "phyllic",
                "chlorite_epidote",
                "sulfide_alteration"
            ]
            
            # Find available maps
            mineral_maps = []
            for indicator in mineral_indicators:
                potential_paths = list(base_dir.glob(f"minerals/{indicator}_map.tif"))
                if potential_paths:
                    mineral_maps.append(potential_paths[0])
            
            alteration_maps = []
            for indicator in alteration_indicators:
                potential_paths = list(base_dir.glob(f"alteration/{indicator}_map.tif"))
                if potential_paths:
                    alteration_maps.append(potential_paths[0])
            
            # Check if we have enough maps
            all_maps = mineral_maps + alteration_maps
            
            if len(all_maps) < 2:
                logger.error(f"Not enough indicator maps found for base metal potential (found {len(all_maps)})")
                raise ValueError(f"Not enough indicator maps found for base metal potential")
            
            logger.info(f"Found {len(mineral_maps)} mineral maps and {len(alteration_maps)} alteration maps")
            
            # Define weights based on map type
            weights = []
            for map_path in all_maps:
                map_name = map_path.stem.split('_')[0].lower()
                
                # Higher weights for specific indicators
                if map_name in ['chalcopyrite', 'sphalerite', 'galena', 'pyrite']:
                    weights.append(0.4)
                elif map_name in ['propylitic', 'phyllic', 'sulfide_alteration']:
                    weights.append(0.3)
                else:
                    weights.append(0.2)
            
            # Normalize weights
            sum_weights = sum(weights)
            weights = [w / sum_weights for w in weights]
            
            # Define labels for visualization
            labels = [path.stem.split('_')[0] for path in all_maps]
            
            # Create output directory if needed
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Combine maps
            result = self.map_combiner.combine_maps(
                map_files=all_maps,
                output_file=output_file,
                method=CombinationMethod.WEIGHTED_ADD,
                weights=weights,
                create_visualization=True,
                labels=labels
            )
            
            logger.info(f"Generated base metal potential map: {output_file}")
            return result
            
        except Exception as e:
            logger.error(f"Error generating base metal potential map: {str(e)}")
            logger.exception("Detailed error:")
            raise RuntimeError(f"Error generating base metal potential map: {str(e)}")