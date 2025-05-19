import os
import logging
import numpy as np
from pathlib import Path
import h5py
from typing import Dict, Tuple, List, Optional, Union
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import datetime

from processors.aster_l2_processor import SceneMetadata, GeographicBounds

# Set up logging
logger = logging.getLogger(__name__)

class VNIRProcessor:
    """
    Class for processing ASTER data using only VNIR bands
    This is a simplified version for when SWIR data is not available
    """
    
    def __init__(self, vnir_file: Union[str, Path], metadata: Optional[SceneMetadata] = None):
        """
        Initialize VNIR processor
        
        Parameters:
        -----------
        vnir_file : Union[str, Path]
            Path to the VNIR file
        metadata : Optional[SceneMetadata]
            Scene metadata
        """
        self.vnir_file = Path(vnir_file)
        self.metadata = metadata
        
        logger.info(f"Initializing VNIR-only processor with VNIR: {self.vnir_file}")
        
        if not self.vnir_file.exists():
            raise FileNotFoundError(f"VNIR file not found: {self.vnir_file}")
        
        # Define band names based on common patterns
        self.band_names = {
            1: 'Band1',  # Blue (0.52-0.60 μm)
            2: 'Band2',  # Red (0.63-0.69 μm)
            3: 'Band3N'  # Near Infrared (0.78-0.86 μm)
        }
        
        # Initialize band data
        self.band_data = {}
        
        # Load the data
        self.load_data()
        
        # Set CRS and transform from metadata or defaults
        if self.metadata and hasattr(self.metadata, 'bounds'):
            bounds = self.metadata.bounds
            
            # Get first band shape for transform
            first_band_shape = next(iter(self.band_data.values())).shape
            
            self.transform = from_bounds(
                bounds.west, bounds.south,
                bounds.east, bounds.north,
                first_band_shape[1], first_band_shape[0]
            )
            
            # Set UTM zone based on center longitude
            center_lon = (bounds.east + bounds.west) / 2
            center_lat = (bounds.north + bounds.south) / 2
            utm_zone = int((center_lon + 180) / 6) + 1
            
            # Adjust EPSG code based on hemisphere
            if center_lat >= 0:
                epsg = 32600 + utm_zone  # Northern hemisphere
            else:
                epsg = 32700 + utm_zone  # Southern hemisphere
                
            self.crs = CRS.from_epsg(epsg)
        else:
            # Default to WGS84
            self.crs = CRS.from_epsg(4326)
            
            # Use a simple transform (identity transform)
            first_band_shape = next(iter(self.band_data.values())).shape
            self.transform = from_bounds(
                0, 0, first_band_shape[1], first_band_shape[0],
                first_band_shape[1], first_band_shape[0]
            )
    
    def load_data(self):
        """Load reflectance data from VNIR file"""
        try:
            with Dataset(self.vnir_file, 'r') as vnir:
                available_vars = list(vnir.variables.keys())
                logger.info(f"Available VNIR variables: {available_vars}")
                
                # Check which bands are available
                for band_num, band_name in self.band_names.items():
                    if band_name in vnir.variables:
                        try:
                            data = vnir.variables[band_name][:]
                            if data is None:
                                logger.warning(f"Loaded null data for VNIR band {band_num}")
                                continue
                            
                            data = data.astype(np.float32)
                            data = data * 0.001  # Scale factor - adjust as needed
                            data[data < 0] = 0
                            data[data > 1] = 1
                            
                            self.band_data[band_num] = data
                            logger.info(f"Loaded VNIR band {band_num} with shape {data.shape}")
                        except Exception as e:
                            logger.error(f"Error loading VNIR band {band_num}: {str(e)}")
                    else:
                        logger.warning(f"Band {band_name} not found in VNIR file")
                
                # Check if we found any bands
                if not self.band_data:
                    # Try using a direct index approach if named bands didn't work
                    for i, var_name in enumerate(available_vars):
                        if 'Band' in var_name:
                            try:
                                data = vnir.variables[var_name][:]
                                if data is None:
                                    continue
                                
                                data = data.astype(np.float32)
                                data = data * 0.001  # Scale factor
                                data[data < 0] = 0
                                data[data > 1] = 1
                                
                                # Map to band numbers 1, 2, 3 as best guess
                                band_num = i + 1
                                if band_num <= 3:
                                    self.band_data[band_num] = data
                                    logger.info(f"Loaded VNIR band {band_num} with shape {data.shape}")
                            except Exception as e:
                                logger.error(f"Error loading variable {var_name}: {str(e)}")
        except Exception as e:
            logger.error(f"Error loading data from {self.vnir_file}: {str(e)}")
            raise
        
        if not self.band_data:
            raise RuntimeError(f"No band data could be loaded from {self.vnir_file}")
        
        logger.info(f"Successfully loaded {len(self.band_data)} bands")
    
    def save_band_as_geotiff(self, band_num: int, output_path: Union[str, Path]) -> str:
        """
        Save a single band as GeoTIFF
        
        Parameters:
        -----------
        band_num : int
            Band number to save
        output_path : Union[str, Path]
            Path to save the GeoTIFF
            
        Returns:
        --------
        str
            Path to the saved file
        """
        output_path = Path(output_path)
        
        if band_num not in self.band_data:
            raise ValueError(f"Band {band_num} not available")
        
        # Get the band data
        band_data = self.band_data[band_num]
        
        # Create GeoTIFF
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=band_data.shape[0],
            width=band_data.shape[1],
            count=1,
            dtype=band_data.dtype,
            crs=self.crs,
            transform=self.transform,
            nodata=np.nan
        ) as dst:
            dst.write(band_data, 1)
            dst.set_band_description(1, f"ASTER VNIR Band {band_num}")
            
            # Add metadata
            if self.metadata:
                dst.update_tags(
                    acquisition_date=self.metadata.acquisition_date,
                    solar_azimuth=str(self.metadata.solar_azimuth),
                    solar_elevation=str(self.metadata.solar_elevation),
                    cloud_cover=str(self.metadata.cloud_cover),
                    band_number=str(band_num),
                    creation_date=datetime.datetime.now().isoformat()
                )
        
        logger.info(f"Saved band {band_num} to {output_path}")
        return str(output_path)
    
    def create_true_color_composite(self, output_path: Union[str, Path]) -> str:
        """
        Create a true color composite using bands 3, 2, 1 (NIR, Red, Green)
        
        Parameters:
        -----------
        output_path : Union[str, Path]
            Path to save the GeoTIFF
            
        Returns:
        --------
        str
            Path to the saved file
        """
        output_path = Path(output_path)
        
        # Check if we have all required bands
        required_bands = [3, 2, 1]
        missing_bands = [b for b in required_bands if b not in self.band_data]
        
        if missing_bands:
            # Try to create with available bands
            if len(self.band_data) >= 3:
                # Use the first 3 available bands
                available_bands = list(self.band_data.keys())[:3]
                logger.warning(f"Missing bands {missing_bands} for true color composite. Using bands {available_bands} instead.")
                required_bands = available_bands
            else:
                raise ValueError(f"Missing required bands for true color composite: {missing_bands}")
        
        # Get band data and normalize
        rgb_bands = []
        for band in required_bands:
            band_data = self.band_data[band]
            
            # Enhance contrast
            valid_mask = ~np.isnan(band_data)
            p2, p98 = np.percentile(band_data[valid_mask], [2, 98])
            band_data = np.clip((band_data - p2) / (p98 - p2), 0, 1)
            
            rgb_bands.append(band_data)
        
        # Stack bands
        rgb = np.dstack(rgb_bands)
        
        # Create visualization
        plt.figure(figsize=(10, 10))
        plt.imshow(rgb)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save as multi-band GeoTIFF
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=rgb.shape[0],
            width=rgb.shape[1],
            count=3,
            dtype=rgb.dtype,
            crs=self.crs,
            transform=self.transform,
            nodata=np.nan
        ) as dst:
            for i, band in enumerate(required_bands):
                dst.write(self.band_data[band], i+1)
                dst.set_band_description(i+1, f"ASTER VNIR Band {band}")
            
            # Add metadata
            dst.update_tags(
                composite_type="true_color",
                bands_used=",".join(str(b) for b in required_bands),
                creation_date=datetime.datetime.now().isoformat()
            )
            
            if self.metadata:
                dst.update_tags(
                    acquisition_date=self.metadata.acquisition_date,
                    solar_azimuth=str(self.metadata.solar_azimuth),
                    solar_elevation=str(self.metadata.solar_elevation),
                    cloud_cover=str(self.metadata.cloud_cover)
                )
        
        logger.info(f"Saved true color composite to {output_path}")
        return str(output_path)
    
    def create_false_color_composite(self, output_path: Union[str, Path]) -> str:
        """
        Create a false color composite (NIR, Red, Green)
        
        Parameters:
        -----------
        output_path : Union[str, Path]
            Path to save the GeoTIFF
            
        Returns:
        --------
        str
            Path to the saved file
        """
        output_path = Path(output_path)
        
        # Check if we have all required bands
        required_bands = [3, 2, 1]  # NIR, Red, Green
        missing_bands = [b for b in required_bands if b not in self.band_data]
        
        if missing_bands:
            # Try to create with available bands
            if len(self.band_data) >= 3:
                # Use the first 3 available bands
                available_bands = list(self.band_data.keys())[:3]
                logger.warning(f"Missing bands {missing_bands} for false color composite. Using bands {available_bands} instead.")
                required_bands = available_bands
            else:
                raise ValueError(f"Missing required bands for false color composite: {missing_bands}")
        
        # Get band data and normalize
        rgb_bands = []
        for band in required_bands:
            band_data = self.band_data[band]
            
            # Enhance contrast
            valid_mask = ~np.isnan(band_data)
            p2, p98 = np.percentile(band_data[valid_mask], [2, 98])
            band_data = np.clip((band_data - p2) / (p98 - p2), 0, 1)
            
            rgb_bands.append(band_data)
        
        # Stack bands
        rgb = np.dstack(rgb_bands)
        
        # Create visualization
        plt.figure(figsize=(10, 10))
        plt.imshow(rgb)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save as multi-band GeoTIFF
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=rgb.shape[0],
            width=rgb.shape[1],
            count=3,
            dtype=rgb.dtype,
            crs=self.crs,
            transform=self.transform,
            nodata=np.nan
        ) as dst:
            for i, band in enumerate(required_bands):
                dst.write(self.band_data[band], i+1)
                dst.set_band_description(i+1, f"ASTER VNIR Band {band}")
            
            # Add metadata
            dst.update_tags(
                composite_type="false_color",
                bands_used=",".join(str(b) for b in required_bands),
                creation_date=datetime.datetime.now().isoformat()
            )
            
            if self.metadata:
                dst.update_tags(
                    acquisition_date=self.metadata.acquisition_date,
                    solar_azimuth=str(self.metadata.solar_azimuth),
                    solar_elevation=str(self.metadata.solar_elevation),
                    cloud_cover=str(self.metadata.cloud_cover)
                )
        
        logger.info(f"Saved false color composite to {output_path}")
        return str(output_path)
    
    def create_ndvi_map(self, output_path: Union[str, Path]) -> str:
        """
        Create an NDVI (Normalized Difference Vegetation Index) map
        
        Parameters:
        -----------
        output_path : Union[str, Path]
            Path to save the GeoTIFF
            
        Returns:
        --------
        str
            Path to the saved file
        """
        output_path = Path(output_path)
        
        # Check if we have the required bands
        if 3 not in self.band_data or 2 not in self.band_data:
            raise ValueError("Bands 3 (NIR) and 2 (Red) are required for NDVI calculation")
        
        # Get band data
        nir = self.band_data[3]
        red = self.band_data[2]
        
        # Calculate NDVI
        # NDVI = (NIR - Red) / (NIR + Red)
        denominator = nir + red
        ndvi = np.zeros_like(nir)
        valid_mask = (denominator > 0) & ~np.isnan(denominator)
        ndvi[valid_mask] = (nir[valid_mask] - red[valid_mask]) / denominator[valid_mask]
        
        # NDVI values should be between -1 and 1
        ndvi = np.clip(ndvi, -1, 1)
        
        # Create visualization
        plt.figure(figsize=(10, 10))
        plt.imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
        plt.colorbar(label='NDVI')
        plt.title('Normalized Difference Vegetation Index (NDVI)')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save as GeoTIFF
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=ndvi.shape[0],
            width=ndvi.shape[1],
            count=1,
            dtype=ndvi.dtype,
            crs=self.crs,
            transform=self.transform,
            nodata=np.nan
        ) as dst:
            dst.write(ndvi, 1)
            dst.set_band_description(1, "NDVI")
            
            # Add metadata
            dst.update_tags(
                index_type="ndvi",
                description="Normalized Difference Vegetation Index",
                formula="(NIR - Red) / (NIR + Red)",
                nir_band="3",
                red_band="2",
                creation_date=datetime.datetime.now().isoformat()
            )
            
            if self.metadata:
                dst.update_tags(
                    acquisition_date=self.metadata.acquisition_date,
                    solar_azimuth=str(self.metadata.solar_azimuth),
                    solar_elevation=str(self.metadata.solar_elevation),
                    cloud_cover=str(self.metadata.cloud_cover)
                )
        
        logger.info(f"Saved NDVI map to {output_path}")
        return str(output_path)
    
    def create_thumbnail(self, output_path: Union[str, Path]) -> str:
        """
        Create a thumbnail image for the scene
        
        Parameters:
        -----------
        output_path : Union[str, Path]
            Path to save the thumbnail
            
        Returns:
        --------
        str
            Path to the saved file
        """
        output_path = Path(output_path)
        
        # Use all available bands to create a composite
        if len(self.band_data) >= 3:
            # Use the first 3 bands
            bands = list(self.band_data.keys())[:3]
        else:
            # Use the available bands and repeat the last one if needed
            bands = list(self.band_data.keys())
            while len(bands) < 3:
                bands.append(bands[-1])
        
        # Get band data and normalize
        rgb_bands = []
        for band in bands:
            band_data = self.band_data[band]
            
            # Enhance contrast
            valid_mask = ~np.isnan(band_data)
            p2, p98 = np.percentile(band_data[valid_mask], [2, 98])
            band_data = np.clip((band_data - p2) / (p98 - p2), 0, 1)
            
            rgb_bands.append(band_data)
        
        # Stack bands
        rgb = np.dstack(rgb_bands)
        
        # Create thumbnail
        plt.figure(figsize=(6, 6))
        plt.imshow(rgb)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved thumbnail to {output_path}")
        return str(output_path)