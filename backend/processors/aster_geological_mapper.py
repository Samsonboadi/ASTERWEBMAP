#aster_geological_mapper.py
import numpy as np
from pathlib import Path
import logging
from enum import Enum
from typing import Dict, Tuple, List, Optional, Union
import matplotlib.pyplot as plt
from scipy import ndimage
from sklearn.decomposition import PCA
from skimage import exposure, morphology
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from skimage import feature, filters, morphology, segmentation, draw
from .aster_l2_processor import ASTER_L2_Processor, MineralIndices

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AlterationIndices(Enum):
    """Alteration mapping indices"""
    ADVANCED_ARGILLIC = "advanced_argillic"
    ARGILLIC = "argillic"
    PROPYLITIC = "propylitic"
    PHYLLIC = "phyllic"
    SILICIFICATION = "silicification"
    IRON_ALTERATION = "iron_alteration"
    GOSSAN = "gossan"
    
    # New alteration types
    SULFIDE_ALTERATION = "sulfide_alteration"
    CHLORITE_EPIDOTE = "chlorite_epidote"
    SERICITE_PYRITE = "sericite_pyrite"
    CARBONATE_ALTERATION = "carbonate_alteration"

class GeologicalFeatures(Enum):
    """Geological feature types"""
    LINEAMENTS = "lineaments"
    FAULTS = "faults"
    CONTACTS = "contacts"
    FRACTURES = "fractures"
    FOLDS = "folds"

class BandCombinations(Enum):
    """Enhanced band combinations for geological mapping"""
    LITHOLOGICAL = "lithological"
    GENERAL_ALTERATION = "general_alteration"
    IRON_OXIDE = "iron_oxide"
    ALOH_MINERALS = "aloh_minerals"
    MGOH_CARBONATE = "mgoh_carbonate"
    CROSTA = "crosta"
    SULFIDE = "sulfide"  
    CHLORITE_ALTERATION = "chlorite_alteration"  

class BandRatioMaps(Enum):
    """Specific band ratio maps"""
    FERRIC_IRON = "ferric_iron"
    ALOH_CONTENT = "aloh_content"
    MGOH_CONTENT = "mgoh_content"

class ASTER_Geological_Mapper:
    def __init__(self, base_processor):
        """
        Initialize the geological mapper extension
        
        Parameters:
        -----------
        base_processor : ASTER_L2_Processor
            Base ASTER processor instance
        """
        self.processor = base_processor
        self.reflectance_data = base_processor.reflectance_data
        self.metadata = base_processor.metadata
        
        # Define alteration indices
        self.alteration_indices = {
            AlterationIndices.ADVANCED_ARGILLIC: {
                'ratios': [(4, 6), (4, 5)],
                'threshold': 0.50,
                'description': 'Advanced argillic alteration (alunite, kaolinite)',
                'band_combinations': [(4, 6), (4, 5), (5, 6)]
            },
            AlterationIndices.ARGILLIC: {
                'ratios': [(5, 7), (5, 6), (4, 5)],
                'threshold': 0.50,
                'description': 'Argillic alteration (kaolinite, illite, smectite)',
                'band_combinations': [(5, 7), (5, 6), (4, 5)]
            },
            AlterationIndices.PROPYLITIC: {
                'ratios': [(7, 8), (9, 8)],
                'threshold': 0.50,
                'description': 'Propylitic alteration (chlorite, epidote)',
                'band_combinations': [(7, 8), (9, 8), (6, 8)]
            },
            AlterationIndices.PHYLLIC: {
                'ratios': [(4, 6), (5, 6)],
                'threshold': 0.50,
                'description': 'Phyllic alteration (sericite)',
                'band_combinations': [(4, 6), (5, 6), (7, 6)]
            },
            AlterationIndices.SILICIFICATION: {
                'ratios': [(7, 6), (9, 8)],
                'threshold': 0.50,
                'description': 'Silicification',
                'band_combinations': [(7, 6), (9, 8), (5, 6)]
            },
            AlterationIndices.SULFIDE_ALTERATION: {
                'ratios': [(2, 1), (3, 4), (5, 6)],
                'threshold': 0.45,
                'description': 'Sulfide alteration assemblage',
                'band_combinations': [(2, 1), (3, 4), (5, 6)]
            },
            AlterationIndices.CHLORITE_EPIDOTE: {
                'ratios': [(7, 8), (6, 9), (5, 6)],
                'threshold': 0.50,
                'description': 'Chlorite-epidote alteration',
                'band_combinations': [(7, 8), (6, 9), (5, 6)]
            },
            AlterationIndices.SERICITE_PYRITE: {
                'ratios': [(4, 6), (5, 6), (2, 1)],
                'threshold': 0.50,
                'description': 'Sericite-pyrite alteration',
                'band_combinations': [(4, 6), (5, 6), (2, 1)]
            },
            AlterationIndices.CARBONATE_ALTERATION: {
                'ratios': [(8, 9), (6, 8), (7, 9)],
                'threshold': 0.50,
                'description': 'Carbonate alteration',
                'band_combinations': [(8, 9), (6, 8), (7, 9)]
            },
            AlterationIndices.IRON_ALTERATION: {
                'ratios': [(2, 1), (3, 2)],
                'threshold': 0.45,
                'description': 'Iron alteration/oxidation',
                'band_combinations': [(2, 1), (3, 2), (4, 2)]
            },
            AlterationIndices.GOSSAN: {
                'ratios': [(2, 1), (3, 4), (2, 3)],
                'threshold': 0.50,
                'description': 'Gossan (iron oxide cap)',
                'band_combinations': [(2, 1), (3, 4), (2, 3)]
            }
        }

        # Define band combinations at class level
        self.combinations = {
            BandCombinations.LITHOLOGICAL: {
                'bands': [(4, 7), (3, 4), (2, 1)],  # Ratios: 4/7, 3/4, 2/1
                'description': 'Lithological boundaries RGB composite',
                'is_ratio': True
            },
            BandCombinations.GENERAL_ALTERATION: {
                'bands': [4, 6, 8],
                'description': 'General alteration false-color composite',
                'is_ratio': False
            },
            BandCombinations.IRON_OXIDE: {
                'bands': [2, 1, 3],
                'description': 'Iron oxide mapping',
                'is_ratio': False
            },
            BandCombinations.ALOH_MINERALS: {
                'bands': [4, 5, 6],
                'description': 'Al-OH minerals mapping',
                'is_ratio': False
            },
            BandCombinations.MGOH_CARBONATE: {
                'bands': [4, 7, 9],
                'description': 'Mg-OH and carbonate minerals mapping',
                'is_ratio': False
            },
            BandCombinations.CROSTA: {
                'bands': [(4, 6), (5, 7), (7, 8)],  # Ratios: 4/6, 5/7, 7/8
                'description': 'Crosta technique composite',
                'is_ratio': True
            },
            BandCombinations.SULFIDE: {
                'bands': [2, 1, 4],
                'description': 'Sulfide minerals mapping',
                'is_ratio': False
            },
            BandCombinations.CHLORITE_ALTERATION: {
                'bands': [7, 8, 6],
                'description': 'Chlorite and epidote alteration mapping',
                'is_ratio': False
            }
        }

    def save_alteration_map(self, alteration_type: AlterationIndices, output_dir: Path):
        """Save alteration map to file"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate alteration index
        index_map, confidence_map = self.calculate_alteration_index(alteration_type)
        
        # Prepare transform and CRS
        if self.processor.metadata and self.processor.metadata.bounds:
            bounds = self.processor.metadata.bounds
            transform = from_bounds(
                bounds.west, bounds.south,
                bounds.east, bounds.north,
                index_map.shape[1], index_map.shape[0]
            )
            utm_zone = self.processor.metadata.utm_zone if self.processor.metadata.utm_zone else int((bounds.east + bounds.west) / 12) + 31
            crs = CRS.from_dict({'proj': 'utm', 'zone': utm_zone, 'datum': 'WGS84'})
        else:
            transform = from_bounds(0, 0, index_map.shape[1], index_map.shape[0],
                                index_map.shape[1], index_map.shape[0])
            crs = CRS.from_epsg(4326)
        
        # Save as GeoTIFF
        output_file = output_dir / f"{alteration_type.value}_map.tif"
        
        with rasterio.open(
            output_file,
            'w',
            driver='GTiff',
            height=index_map.shape[0],
            width=index_map.shape[1],
            count=2,
            dtype=index_map.dtype,
            crs=crs,
            transform=transform,
            nodata=np.nan
        ) as dst:
            dst.write(index_map, 1)
            dst.write(confidence_map, 2)
            dst.set_band_description(1, f"{alteration_type.value} index")
            dst.set_band_description(2, "Confidence map")
            
            if self.processor.metadata:
                dst.update_tags(
                    acquisition_date=self.processor.metadata.acquisition_date,
                    solar_azimuth=str(self.processor.metadata.solar_azimuth),
                    solar_elevation=str(self.processor.metadata.solar_elevation),
                    cloud_cover=str(self.processor.metadata.cloud_cover)
                )
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        
        plt.subplot(121)
        plt.imshow(index_map, cmap='viridis')
        plt.colorbar(label='Index Value')
        plt.title(f"{alteration_type.value} Distribution")
        
        plt.subplot(122)
        plt.imshow(confidence_map, cmap='RdYlGn')
        plt.colorbar(label='Confidence')
        plt.title('Confidence Map')
        
        plt.tight_layout()
        plt.savefig(output_dir / f"{alteration_type.value}_visualization.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved {alteration_type.value} map to {output_file}")

    def calculate_alteration_index(self, alteration_type: AlterationIndices,
                                 enhanced_processing: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate alteration index with enhanced processing
        
        Parameters:
        -----------
        alteration_type : AlterationIndices
            Type of alteration to map
        enhanced_processing : bool
            Whether to apply enhanced processing algorithms
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Alteration index array and confidence map
        """
        if alteration_type not in self.alteration_indices:
            raise ValueError(f"Unsupported alteration type: {alteration_type}")
            
        alteration_info = self.alteration_indices[alteration_type]
        ratios = alteration_info['ratios']
        threshold = alteration_info['threshold']
        
        # Initialize output arrays
        first_band = next(iter(self.reflectance_data.values()))
        result = np.zeros_like(first_band, dtype=np.float32)
        confidence = np.zeros_like(first_band, dtype=np.float32)
        
        # Calculate band ratios
        ratio_results = []
        for band1, band2 in ratios:
            if band1 not in self.reflectance_data or band2 not in self.reflectance_data:
                raise ValueError(f"Missing required bands {band1} or {band2}")
                
            b1 = self.reflectance_data[band1]
            b2 = self.reflectance_data[band2]
            
            # Check for invalid data
            if b1 is None or b2 is None or not np.any(b1) or not np.any(b2) or np.all(np.isnan(b1)) or np.all(np.isnan(b2)):
                logger.warning(f"Invalid data in bands {band1} or {band2}")
                return np.zeros_like(b1), np.zeros_like(b1)
            
            # Calculate ratio where both bands have valid data
            valid_mask = (b1 > 0) & (b2 > 0) & ~np.isnan(b1) & ~np.isnan(b2)
            ratio = np.zeros_like(b1)
            ratio[valid_mask] = b1[valid_mask] / b2[valid_mask]
            
            if enhanced_processing:
                # Apply noise reduction
                ratio = ndimage.gaussian_filter(ratio, sigma=1)
                
                # Enhance contrast
                valid_ratio = ratio[valid_mask]
                if len(valid_ratio) > 0:
                    p2, p98 = np.percentile(valid_ratio, [2, 98])
                    ratio[valid_mask] = exposure.rescale_intensity(
                        ratio[valid_mask], in_range=(p2, p98)
                    )
            
            ratio_results.append(ratio)
        
        # Combine ratios with PCA if enhanced processing
        if enhanced_processing and len(ratio_results) > 1:
            # Reshape for PCA
            data = np.stack(ratio_results, axis=-1)
            valid_mask = ~np.isnan(data).any(axis=-1)
            shape = data.shape
            
            # Apply PCA
            data_2d = data[valid_mask].reshape(-1, len(ratio_results))
            pca = PCA(n_components=1)
            transformed = pca.fit_transform(data_2d)
            
            # Reshape back
            result = np.zeros(shape[:-1])
            result[valid_mask] = transformed.ravel()
            
            # Calculate confidence based on explained variance
            confidence[valid_mask] = pca.explained_variance_ratio_[0]
        else:
            # Simple averaging if not using enhanced processing
            for ratio in ratio_results:
                result += ratio
            result /= len(ratios)
            
            # Calculate confidence based on ratio consistency
            confidence = 1 - np.std(ratio_results, axis=0)
        
        # Apply threshold
        result[confidence < threshold] = 0
        
        return result, confidence

    def detect_geological_features(self, feature_type: GeologicalFeatures,
                                    band_number: int = 4) -> np.ndarray:
        """
        Detect geological features using edge detection and morphological operations
        
        Parameters:
        -----------
        feature_type : GeologicalFeatures
            Type of geological feature to detect
        band_number : int
            Band number to use for feature detection
            
        Returns:
        --------
        np.ndarray
            Binary mask of detected features
        """
        from skimage import feature, filters, morphology, segmentation
        import numpy as np
        
        # Get band data
        if band_number not in self.processor.reflectance_data:
            raise ValueError(f"Band {band_number} not available")
            
        img = self.processor.reflectance_data[band_number]
        
        # Apply preprocessing
        img = filters.gaussian(img, sigma=1)
        
        if feature_type == GeologicalFeatures.LINEAMENTS:
            # Detect edges
            edges = feature.canny(img, sigma=2)
            
            # Use skeletonization for lineament detection
            skeleton = morphology.skeletonize(edges)
            
            return skeleton.astype(np.float32)
            
        elif feature_type in [GeologicalFeatures.FAULTS, GeologicalFeatures.FRACTURES]:
            # Use edge detection with different parameters
            edges = feature.canny(
                img,
                sigma=2.0,
                low_threshold=0.55,
                high_threshold=0.8
            )
            # Clean up noise
            result = morphology.remove_small_objects(edges, min_size=20)
            
        elif feature_type == GeologicalFeatures.CONTACTS:
            # Use watershed segmentation for contacts
            gradient = filters.sobel(img)
            markers = np.zeros_like(img, dtype=int)
            markers[img < img.mean()] = 1
            markers[img > img.mean()] = 2
            result = segmentation.watershed(gradient, markers) == 2
            
        else:
            raise ValueError(f"Unsupported feature type: {feature_type}")
            
        return result.astype(np.float32)

    def create_composite_map(self, output_dir: Path):
        """Create enhanced composite geological map combining multiple features"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process all band combinations
        for combo in BandCombinations:
            try:
                self.create_band_combination_map(combo, output_dir)
            except Exception as e:
                logger.error(f"Error creating {combo.value} map: {str(e)}")
        
        # Create individual ratio maps
        for ratio in BandRatioMaps:
            try:
                self.create_ratio_map(ratio, output_dir)
            except Exception as e:
                logger.error(f"Error creating {ratio.value} map: {str(e)}")

    def create_ratio_map(self, ratio_type: BandRatioMaps, output_dir: Path):
        """Create and save ratio maps as GeoTIFFs with correct georeferencing."""
        ratios = {
            BandRatioMaps.FERRIC_IRON: {
                'bands': (2, 1),
                'description': 'Ferric Iron Content (Band2/Band1)',
                'threshold': 0.50
            },
            BandRatioMaps.ALOH_CONTENT: {
                'bands': (5, 6),
                'description': 'Al-OH Content (Band5/Band6)',
                'threshold': 0.50
            },
            BandRatioMaps.MGOH_CONTENT: {
                'bands': (7, 8),
                'description': 'Mg-OH Content (Band7/Band8)',
                'threshold': 0.50
            }
        }
        
        ratio_info = ratios[ratio_type]
        output_file = output_dir / f"{ratio_type.value}.tif"
        
        try:
            # Calculate ratio
            num_band = ratio_info['bands'][0]
            den_band = ratio_info['bands'][1]
            
            numerator = self.processor.reflectance_data[num_band]
            denominator = self.processor.reflectance_data[den_band]
            
            valid_mask = (numerator > 0) & (denominator > 0)
            ratio = np.zeros_like(numerator, dtype=np.float32)
            ratio[valid_mask] = numerator[valid_mask] / denominator[valid_mask]
            
            # Calculate confidence
            confidence = np.ones_like(ratio, dtype=np.float32)
            confidence[~valid_mask] = 0
            
            # Normalize ratio to 0-1 range
            if np.any(valid_mask):
                p2, p98 = np.percentile(ratio[valid_mask], [2, 98])
                ratio = np.clip((ratio - p2) / (p98 - p2), 0, 1)
            
            # Prepare transform and CRS to match composites
            if self.metadata and hasattr(self.metadata, 'bounds'):
                bounds = self.metadata.bounds
                transform = from_bounds(
                    bounds.west, bounds.south,
                    bounds.east, bounds.north,
                    ratio.shape[1], ratio.shape[0]
                )
                crs = CRS.from_epsg(4326)
            else:
                transform = from_bounds(0, 0, ratio.shape[1], ratio.shape[0], ratio.shape[1], ratio.shape[0])
                crs = CRS.from_epsg(4326)
            
            # Save as GeoTIFF
            nodata = -9999
            ratio = np.nan_to_num(ratio, nan=nodata)
            confidence = np.nan_to_num(confidence, nan=nodata)
            
            with rasterio.open(
                output_file,
                'w',
                driver='GTiff',
                height=ratio.shape[0],
                width=ratio.shape[1],
                count=2,
                dtype=rasterio.float32,
                crs=crs,
                transform=transform,
                nodata=nodata,
                compress='LZW'
            ) as dst:
                dst.write(ratio, 1)
                dst.write(confidence, 2)
                dst.set_band_description(1, ratio_info['description'])
                dst.set_band_description(2, "Confidence map")
                
                if self.metadata:
                    dst.update_tags(
                        ratio_type=ratio_type.value,
                        description=ratio_info['description'],
                        bands_used=f"{num_band}/{den_band}",
                        acquisition_date=self.metadata.acquisition_date
                    )
            
            logger.info(f"Saved {ratio_type.value}.tif with correct georeferencing to {output_file}")
            
        except Exception as e:
            logger.error(f"Error creating {ratio_type.value} map: {str(e)}")
            raise

    def create_band_combination_map(self, combo_type: BandCombinations, output_dir: Path):
        """Create band combination maps and save specific minerals as GeoTIFFs with correct georeferencing."""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Log at the start to see if we're getting here
        logger.info(f"Creating band combination map for {combo_type.value}")
        
        # Verify metadata and bounds
        if not hasattr(self, 'metadata') or not self.metadata:
            logger.error(f"Missing metadata for {combo_type.value}")
            raise ValueError(f"Missing metadata for {combo_type.value}")
            
        if not hasattr(self.metadata, 'bounds') or not self.metadata.bounds:
            logger.error(f"Missing bounds in metadata for {combo_type.value}")
            raise ValueError(f"Missing bounds in metadata for {combo_type.value}")
            
        # Print the existing bounds for debugging
        logger.info(f"Bounds for {combo_type.value}: {self.metadata.bounds.__dict__ if hasattr(self.metadata.bounds, '__dict__') else self.metadata.bounds}")
        
        combo_info = self.combinations[combo_type]
        output_file = output_dir / f"{combo_type.value}_composite.tif"
        
        try:
            # Prepare transform and CRS
            bounds = self.metadata.bounds
            crs = CRS.from_epsg(4326)
            
            rgb = np.zeros((self.reflectance_data[4].shape[0], 
                          self.reflectance_data[4].shape[1], 3), dtype=np.float32)
            nodata = -9999
            
            if combo_info['is_ratio']:
                # Handle ratio-based combinations (e.g., LITHOLOGICAL, CROSTA)
                ratio_pairs = combo_info['bands']  # List of (num, den) tuples
                if len(ratio_pairs) != 3:
                    raise ValueError(f"Expected 3 ratio pairs for {combo_type.value}, got {len(ratio_pairs)}")
                
                for i, (num_band, den_band) in enumerate(ratio_pairs):
                    if num_band not in self.reflectance_data or den_band not in self.reflectance_data:
                        logger.warning(f"Missing band data for {num_band} or {den_band}")
                        return
                    
                    numerator = self.reflectance_data[num_band]
                    denominator = self.reflectance_data[den_band]
                    
                    valid_mask = (numerator > 0) & (denominator > 0) & ~np.isnan(numerator) & ~np.isnan(denominator)
                    if not np.any(valid_mask):
                        logger.warning(f"Invalid data in bands {num_band} or {den_band} for {combo_type.value}")
                        return
                    
                    ratio = np.zeros_like(numerator, dtype=np.float32)
                    ratio[valid_mask] = numerator[valid_mask] / denominator[valid_mask]
                    
                    if np.any(valid_mask):
                        p2, p98 = np.percentile(ratio[valid_mask], [2, 98])
                        ratio = np.clip((ratio - p2) / (p98 - p2), 0, 1)
                    
                    rgb[:, :, i] = ratio
            else:
                # Handle direct band combinations (e.g., GENERAL_ALTERATION, IRON_OXIDE)
                bands = combo_info['bands']
                if len(bands) != 3:
                    raise ValueError(f"Expected 3 bands for {combo_type.value}, got {len(bands)}")
                
                for i, band in enumerate(bands):
                    if band not in self.reflectance_data:
                        logger.warning(f"Missing band data for {band}")
                        return
                    
                    band_data = self.reflectance_data[band]
                    valid_mask = ~np.isnan(band_data)
                    if not np.any(valid_mask):
                        logger.warning(f"Invalid data in band {band} for {combo_type.value}")
                        return
                    
                    if np.any(valid_mask):
                        p2, p98 = np.percentile(band_data[valid_mask], [2, 98])
                        band_data = np.clip((band_data - p2) / (p98 - p2), 0, 1)
                    
                    rgb[:, :, i] = band_data
                    
                # Save iron_oxide and mgoh_carbonate as individual maps
                if combo_type == BandCombinations.IRON_OXIDE:
                    iron_file = output_dir / "iron_oxide.tif"
                    iron_data = rgb[:, :, 0]  # Band 2 as primary indicator
                    transform = from_bounds(
                        bounds.west, bounds.south,
                        bounds.east, bounds.north,
                        iron_data.shape[1], iron_data.shape[0]
                    )
                    
                    with rasterio.open(
                        iron_file,
                        'w',
                        driver='GTiff',
                        height=iron_data.shape[0],
                        width=iron_data.shape[1],
                        count=1,
                        dtype=iron_data.dtype,
                        crs=crs,
                        transform=transform,
                        nodata=nodata,
                        compress='LZW'
                    ) as dst:
                        dst.write(iron_data, 1)
                        dst.set_band_description(1, "Iron Oxide Index (Band 2)")
                        if self.metadata:
                            dst.update_tags(
                                description="Iron oxide individual map",
                                acquisition_date=self.metadata.acquisition_date
                            )
                    logger.info(f"Saved iron_oxide.tif with correct georeferencing to {iron_file}")
                
                elif combo_type == BandCombinations.MGOH_CARBONATE:
                    mgoh_file = output_dir / "mgoh_carbonate.tif"
                    mgoh_data = rgb[:, :, 1]  # Band 7 as primary indicator
                    transform = from_bounds(
                        bounds.west, bounds.south,
                        bounds.east, bounds.north,
                        mgoh_data.shape[1], mgoh_data.shape[0]
                    )
                    
                    with rasterio.open(
                        mgoh_file,
                        'w',
                        driver='GTiff',
                        height=mgoh_data.shape[0],
                        width=mgoh_data.shape[1],
                        count=1,
                        dtype=mgoh_data.dtype,
                        crs=crs,
                        transform=transform,
                        nodata=nodata,
                        compress='LZW'
                    ) as dst:
                        dst.write(mgoh_data, 1)
                        dst.set_band_description(1, "Mg-OH/Carbonate Index (Band 7)")
                        if self.metadata:
                            dst.update_tags(
                                description="Mg-OH and carbonate individual map",
                                acquisition_date=self.metadata.acquisition_date
                            )
                    logger.info(f"Saved mgoh_carbonate.tif with correct georeferencing to {mgoh_file}")
            
            # Save composite map
            transform = from_bounds(
                bounds.west, bounds.south,
                bounds.east, bounds.north,
                rgb.shape[1], rgb.shape[0]
            )
            
            with rasterio.open(
                output_file,
                'w',
                driver='GTiff',
                height=rgb.shape[0],
                width=rgb.shape[1],
                count=3,
                dtype=rgb.dtype,
                crs=crs,
                transform=transform,
                nodata=nodata,
                compress='LZW'
            ) as dst:
                for i in range(3):
                    dst.write(rgb[:, :, i], i + 1)
                dst.descriptions = [combo_info['description']] * 3
                
                if self.metadata:
                    dst.update_tags(
                        band_combination=combo_type.value,
                        description=combo_info['description'],
                        acquisition_date=self.metadata.acquisition_date
                    )
            
            logger.info(f"Saved {combo_type.value}_composite.tif with correct georeferencing to {output_file}")
            
        except Exception as e:
            logger.error(f"Error creating {combo_type.value} composite and individual maps: {str(e)}")
            raise

    def create_enhanced_sulfide_map(self, output_dir: Path, threshold: float = 0.65):
        """
        Create an enhanced sulfide alteration map by combining multiple indicators
        
        Parameters:
        -----------
        output_dir : Path
            Output directory for saving results
        threshold : float
            Threshold value for sulfide detection (higher values = more conservative)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Generating enhanced sulfide alteration map...")
        
        # Collect input indicators
        indicators = {}
        
        # Get sulfide-related mineral indices from processor
        for mineral in [MineralIndices.PYRITE, MineralIndices.CHALCOPYRITE, 
                       MineralIndices.IRON_OXIDE, MineralIndices.FERROUS_IRON]:
            try:
                index_map, confidence = self.processor.calculate_mineral_index(mineral)
                indicators[f"{mineral.value}"] = index_map * confidence
                logger.info(f"Added {mineral.value} index to sulfide indicators")
            except Exception as e:
                logger.error(f"Error calculating {mineral.value} index: {str(e)}")
        
        # Get alteration indices
        for alteration in [AlterationIndices.SULFIDE_ALTERATION, 
                          AlterationIndices.SERICITE_PYRITE,
                          AlterationIndices.GOSSAN]:
            try:
                index_map, confidence = self.calculate_alteration_index(alteration)
                indicators[f"{alteration.value}"] = index_map * confidence
                logger.info(f"Added {alteration.value} index to sulfide indicators")
            except Exception as e:
                logger.error(f"Error calculating {alteration.value} index: {str(e)}")
        
        # Calculate direct band ratios for iron sulfide detection
        band2 = self.reflectance_data[2]  # Iron-sensitive
        band1 = self.reflectance_data[1]  # Iron-sensitive
        band3 = self.reflectance_data[3]  # Used for normalization
        
        valid_mask = (band2 > 0) & (band1 > 0) & (band3 > 0)
        iron_ratio = np.zeros_like(band2)
        iron_ratio[valid_mask] = band2[valid_mask] / band1[valid_mask]
        
        # Normalize
        if np.any(valid_mask):
            p2, p98 = np.percentile(iron_ratio[valid_mask], [2, 98])
            iron_ratio = np.clip((iron_ratio - p2) / (p98 - p2), 0, 1)
        
        indicators["iron_ratio"] = iron_ratio
        logger.info("Added direct iron band ratio to sulfide indicators")
        
        # Combine all indicators with PCA
        try:
            # Stack indicators and handle missing values
            indicator_stack = []
            indicator_names = []
            for name, indicator in indicators.items():
                if indicator is not None and not np.all(np.isnan(indicator)):
                    indicator_stack.append(indicator)
                    indicator_names.append(name)
            
            # Create 3D array for PCA
            data = np.dstack(indicator_stack)
            valid_mask = ~np.isnan(data).any(axis=2)
            
            if np.sum(valid_mask) < 100:
                logger.warning("Too few valid pixels for PCA analysis. Using simple average instead.")
                # Fall back to simple averaging
                sulfide_map = np.zeros_like(list(indicators.values())[0])
                for indicator in indicators.values():
                    if indicator is not None:
                        sulfide_map += np.nan_to_num(indicator, 0)
                sulfide_map /= len(indicators)
            else:
                # Reshape for PCA
                data_2d = data[valid_mask].reshape(-1, len(indicator_stack))
                
                # Run PCA
                pca = PCA(n_components=1)
                transformed = pca.fit_transform(data_2d)
                
                # Reshape back to 2D map
                sulfide_map = np.zeros(valid_mask.shape)
                sulfide_map[valid_mask] = transformed.ravel()
                
                # Normalize to 0-1 range
                if np.any(valid_mask):
                    sulfide_map = (sulfide_map - np.min(sulfide_map[valid_mask])) / \
                                (np.max(sulfide_map[valid_mask]) - np.min(sulfide_map[valid_mask]))
            
            # Apply threshold to create binary mask
            sulfide_mask = sulfide_map > threshold
            
            # Apply spatial filtering to remove noise
            sulfide_mask = ndimage.binary_opening(sulfide_mask, structure=np.ones((3, 3)))
            sulfide_mask = ndimage.binary_closing(sulfide_mask, structure=np.ones((3, 3)))
            
            # Calculate confidence based on indicator agreement
            indicator_count = np.zeros_like(sulfide_map)
            for indicator in indicators.values():
                if indicator is not None:
                    indicator_binary = indicator > np.nanpercentile(indicator, 75)
                    indicator_count += indicator_binary
            
            confidence_map = indicator_count / len(indicators)
            
            # Save the result as a GeoTIFF
            output_file = output_dir / "enhanced_sulfide_map.tif"
            
            # Prepare transform and CRS
            if self.metadata and hasattr(self.metadata, 'bounds'):
                bounds = self.metadata.bounds
                transform = from_bounds(
                    bounds.west, bounds.south,
                    bounds.east, bounds.north,
                    sulfide_map.shape[1], sulfide_map.shape[0]
                )
                crs = CRS.from_epsg(4326)
            else:
                logger.warning("No metadata bounds available. Using fallback.")
                transform = from_bounds(0, 0, sulfide_map.shape[1], sulfide_map.shape[0],
                                      sulfide_map.shape[1], sulfide_map.shape[0])
                crs = CRS.from_epsg(4326)
            
            # Handle NaN values
            nodata = -9999
            sulfide_map = np.nan_to_num(sulfide_map, nan=nodata)
            confidence_map = np.nan_to_num(confidence_map, nan=nodata)
            
            with rasterio.open(
                output_file,
                'w',
                driver='GTiff',
                height=sulfide_map.shape[0],
                width=sulfide_map.shape[1],
                count=2,
                dtype=rasterio.float32,
                crs=crs,
                transform=transform,
                nodata=nodata,
                compress='LZW'
            ) as dst:
                dst.write(sulfide_map.astype(rasterio.float32), 1)
                dst.write(confidence_map.astype(rasterio.float32), 2)
                dst.set_band_description(1, "Enhanced Sulfide Index")
                dst.set_band_description(2, "Confidence Map")
                
                if self.metadata:
                    dst.update_tags(
                        description="Enhanced sulfide alteration map",
                        indicators_used=",".join(indicator_names),
                        threshold=str(threshold),
                        acquisition_date=self.metadata.acquisition_date
                    )
            
            # Create visualization
            plt.figure(figsize=(12, 8))
            
            plt.subplot(121)
            plt.imshow(sulfide_map, cmap='viridis')
            plt.colorbar(label='Sulfide Index')
            plt.title('Enhanced Sulfide Distribution')
            
            plt.subplot(122)
            plt.imshow(confidence_map, cmap='RdYlGn')
            plt.colorbar(label='Confidence')
            plt.title('Confidence Map')
            
            plt.tight_layout()
            plt.savefig(output_dir / "enhanced_sulfide_map.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved enhanced sulfide map to {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Error creating enhanced sulfide map: {str(e)}")
            logger.error("Stack trace:", exc_info=True)
            raise

    def generate_alteration_minerals_report(self, output_dir: Path):
        """
        Generate comprehensive alteration minerals analysis and report
        focusing on chlorite, sericite, carbonates and their relationships
        
        Parameters:
        -----------
        output_dir : Path
            Output directory for saving results
        
        Returns:
        --------
        Path
            Path to the generated report file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report_file = output_dir / 'alteration_minerals_report.txt'
        csv_file = output_dir / 'alteration_minerals_data.csv'
        
        logger.info("Generating alteration minerals analysis...")
        
        # Collect all the mineral maps
        mineral_maps = {}
        mineral_confidence = {}
        
        # Get key alteration minerals
        target_minerals = [
            MineralIndices.CHLORITE,
            MineralIndices.SERICITE, 
            MineralIndices.CARBONATE,
            MineralIndices.EPIDOTE,
            MineralIndices.KAOLINITE,
            MineralIndices.ILLITE,
            MineralIndices.ALUNITE
        ]
        
        for mineral in target_minerals:
            try:
                index_map, confidence = self.processor.calculate_mineral_index(mineral)
                mineral_maps[mineral.value] = index_map
                mineral_confidence[mineral.value] = confidence
                logger.info(f"Calculated {mineral.value} distribution")
            except Exception as e:
                logger.error(f"Error calculating {mineral.value} index: {str(e)}")
        
        # Get alteration assemblages
        target_assemblages = [
            AlterationIndices.PROPYLITIC,
            AlterationIndices.PHYLLIC,
            AlterationIndices.ARGILLIC,
            AlterationIndices.CHLORITE_EPIDOTE,
            AlterationIndices.SERICITE_PYRITE,
            AlterationIndices.CARBONATE_ALTERATION
        ]
        
        assemblage_maps = {}
        assemblage_confidence = {}
        
        for assemblage in target_assemblages:
            try:
                index_map, confidence = self.calculate_alteration_index(assemblage)
                assemblage_maps[assemblage.value] = index_map
                assemblage_confidence[assemblage.value] = confidence
                logger.info(f"Calculated {assemblage.value} distribution")
            except Exception as e:
                logger.error(f"Error calculating {assemblage.value} index: {str(e)}")
        
        # Create RGB composite showing chlorite-sericite-carbonate
        try:
            # Create 3-band composite
            if all(m in mineral_maps for m in ['chlorite', 'sericite', 'carbonate']):
                rgb = np.zeros((
                    mineral_maps['chlorite'].shape[0],
                    mineral_maps['chlorite'].shape[1],
                    3
                ), dtype=np.float32)
                
                # Normalize each mineral map to 0-1 range
                for i, mineral in enumerate(['chlorite', 'sericite', 'carbonate']):
                    m = mineral_maps[mineral]
                    valid_mask = ~np.isnan(m)
                    if np.any(valid_mask):
                        p2, p98 = np.percentile(m[valid_mask], [2, 98])
                        rgb[:, :, i] = np.clip((m - p2) / (p98 - p2), 0, 1)
                
                # Save as GeoTIFF
                rgb_file = output_dir / 'chlorite_sericite_carbonate_rgb.tif'
                
                # Prepare transform and CRS
                if self.metadata and hasattr(self.metadata, 'bounds'):
                    bounds = self.metadata.bounds
                    transform = from_bounds(
                        bounds.west, bounds.south,
                        bounds.east, bounds.north,
                        rgb.shape[1], rgb.shape[0]
                    )
                    crs = CRS.from_epsg(4326)
                else:
                    transform = from_bounds(
                        0, 0, rgb.shape[1], rgb.shape[0],
                        rgb.shape[1], rgb.shape[0]
                    )
                    crs = CRS.from_epsg(4326)
                
                with rasterio.open(
                    rgb_file,
                    'w',
                    driver='GTiff',
                    height=rgb.shape[0],
                    width=rgb.shape[1],
                    count=3,
                    dtype=rgb.dtype,
                    crs=crs,
                    transform=transform,
                    nodata=np.nan,
                    compress='LZW'
                ) as dst:
                    for i in range(3):
                        dst.write(rgb[:, :, i], i + 1)
                    dst.set_band_description(1, "Chlorite (R)")
                    dst.set_band_description(2, "Sericite (G)")
                    dst.set_band_description(3, "Carbonate (B)")
                    
                    if self.metadata:
                        dst.update_tags(
                            description="Chlorite-Sericite-Carbonate RGB composite",
                            r_band="chlorite",
                            g_band="sericite",
                            b_band="carbonate",
                            acquisition_date=self.metadata.acquisition_date
                        )
                
                # Create visualization
                plt.figure(figsize=(10, 8))
                plt.imshow(rgb)
                plt.title('Chlorite (R) - Sericite (G) - Carbonate (B) Composite')
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(output_dir / 'chlorite_sericite_carbonate_rgb.png', 
                        dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info("Created chlorite-sericite-carbonate RGB composite")
            else:
                logger.warning("Could not create RGB composite - missing mineral maps")
                
        except Exception as e:
            logger.error(f"Error creating RGB composite: {str(e)}")
        
        # Calculate alteration ratios
        try:
            # Calculate CSCB (Chlorite-Sericite-Carbonate Balance) index
            if all(m in mineral_maps for m in ['chlorite', 'sericite', 'carbonate']):
                chlorite = np.nan_to_num(mineral_maps['chlorite'], 0)
                sericite = np.nan_to_num(mineral_maps['sericite'], 0)
                carbonate = np.nan_to_num(mineral_maps['carbonate'], 0)
                
                # Simple ratio of chlorite to sericite (high values = propylitic, low values = phyllic)
                chl_ser_ratio = np.zeros_like(chlorite)
                valid_mask = (chlorite > 0) & (sericite > 0)
                chl_ser_ratio[valid_mask] = chlorite[valid_mask] / sericite[valid_mask]
                
                # Ratio of carbonate to total alteration (high values = carbonate alteration)
                carb_alt_ratio = carbonate / (chlorite + sericite + carbonate + 0.001)
                
                # Save these indicators
                indicator_file = output_dir / 'alteration_indicators.tif'
                
                # Prepare transform and CRS
                if self.metadata and hasattr(self.metadata, 'bounds'):
                    bounds = self.metadata.bounds
                    transform = from_bounds(
                        bounds.west, bounds.south,
                        bounds.east, bounds.north,
                        chl_ser_ratio.shape[1], chl_ser_ratio.shape[0]
                    )
                    crs = CRS.from_epsg(4326)
                else:
                    transform = from_bounds(
                        0, 0, chl_ser_ratio.shape[1], chl_ser_ratio.shape[0],
                        chl_ser_ratio.shape[1], chl_ser_ratio.shape[0]
                    )
                    crs = CRS.from_epsg(4326)
                
                with rasterio.open(
                    indicator_file,
                    'w',
                    driver='GTiff',
                    height=chl_ser_ratio.shape[0],
                    width=chl_ser_ratio.shape[1],
                    count=2,
                    dtype=rasterio.float32,
                    crs=crs,
                    transform=transform,
                    nodata=np.nan
                ) as dst:
                    dst.write(np.nan_to_num(chl_ser_ratio, np.nan), 1)
                    dst.write(np.nan_to_num(carb_alt_ratio, np.nan), 2)
                    dst.set_band_description(1, "Chlorite/Sericite Ratio")
                    dst.set_band_description(2, "Carbonate/Total Alteration Ratio")
                
                logger.info("Calculated and saved alteration ratio indicators")
            else:
                logger.warning("Could not calculate alteration ratios - missing mineral maps")
                
        except Exception as e:
            logger.error(f"Error calculating alteration ratios: {str(e)}")
        
        # Generate statistics and write report
        with open(report_file, 'w') as f:
            f.write("ASTER Alteration Minerals Analysis Report\n")
            f.write("=======================================\n\n")
            
            f.write("Scene Information:\n")
            f.write("-----------------\n")
            if self.metadata:
                f.write(f"Acquisition Date: {self.metadata.acquisition_date}\n")
                f.write(f"Solar Azimuth: {self.metadata.solar_azimuth}°\n")
                f.write(f"Solar Elevation: {self.metadata.solar_elevation}°\n")
                f.write(f"Cloud Cover: {self.metadata.cloud_cover}%\n\n")
            
            f.write("Alteration Minerals Analysis:\n")
            f.write("---------------------------\n")
            
            # Save statistics for each mineral
            csv_data = []
            csv_header = ['mineral', 'coverage_percent', 'mean_intensity', 'mean_confidence', 'max_intensity']
            
            for mineral, mineral_map in mineral_maps.items():
                valid_pixels = mineral_map[~np.isnan(mineral_map)]
                
                if len(valid_pixels) > 0:
                    coverage = np.sum(valid_pixels > 0) / len(valid_pixels) * 100
                    mean_intensity = np.mean(valid_pixels[valid_pixels > 0])
                    max_intensity = np.max(valid_pixels)
                    mean_confidence = np.mean(mineral_confidence[mineral][~np.isnan(mineral_confidence[mineral])])
                    
                    f.write(f"\n{mineral.title()}:\n")
                    f.write(f"Coverage: {coverage:.2f}%\n")
                    f.write(f"Mean Intensity: {mean_intensity:.3f}\n")
                    f.write(f"Max Intensity: {max_intensity:.3f}\n")
                    f.write(f"Mean Confidence: {mean_confidence:.3f}\n")
                    
                    # Create visualization for this mineral
                    plt.figure(figsize=(12, 4))
                    
                    plt.subplot(121)
                    plt.imshow(mineral_map, cmap='viridis')
                    plt.colorbar(label='Index Value')
                    plt.title(f'{mineral.title()} Distribution')
                    
                    plt.subplot(122)
                    plt.imshow(mineral_confidence[mineral], cmap='RdYlGn')
                    plt.colorbar(label='Confidence')
                    plt.title('Confidence Map')
                    
                    plt.tight_layout()
                    plt.savefig(output_dir / f'{mineral}_distribution.png', dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    # Add data for CSV
                    csv_data.append([
                        mineral,
                        f"{coverage:.2f}",
                        f"{mean_intensity:.3f}",
                        f"{mean_confidence:.3f}",
                        f"{max_intensity:.3f}"
                    ])
            
            # Write alteration assemblage statistics
            f.write("\n\nAlteration Assemblages Analysis:\n")
            f.write("--------------------------------\n")
            
            for assemblage, assemblage_map in assemblage_maps.items():
                valid_pixels = assemblage_map[~np.isnan(assemblage_map)]
                
                if len(valid_pixels) > 0:
                    coverage = np.sum(valid_pixels > 0) / len(valid_pixels) * 100
                    mean_intensity = np.mean(valid_pixels[valid_pixels > 0])
                    max_intensity = np.max(valid_pixels)
                    mean_confidence = np.mean(assemblage_confidence[assemblage][~np.isnan(assemblage_confidence[assemblage])])
                    
                    f.write(f"\n{assemblage.replace('_', ' ').title()}:\n")
                    f.write(f"Coverage: {coverage:.2f}%\n")
                    f.write(f"Mean Intensity: {mean_intensity:.3f}\n")
                    f.write(f"Mean Confidence: {mean_confidence:.3f}\n")
                    
                    # Create visualization for this assemblage as well
                    plt.figure(figsize=(12, 4))
                    
                    plt.subplot(121)
                    plt.imshow(assemblage_map, cmap='plasma')
                    plt.colorbar(label='Index Value')
                    plt.title(f'{assemblage.replace("_", " ").title()} Distribution')
                    
                    plt.subplot(122)
                    plt.imshow(assemblage_confidence[assemblage], cmap='RdYlGn')
                    plt.colorbar(label='Confidence')
                    plt.title('Confidence Map')
                    
                    plt.tight_layout()
                    plt.savefig(output_dir / f'{assemblage}_distribution.png', dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    # Add to CSV data
                    csv_data.append([
                        f"assemblage_{assemblage}",
                        f"{coverage:.2f}",
                        f"{mean_intensity:.3f}",
                        f"{mean_confidence:.3f}",
                        f"{max_intensity:.3f}"
                    ])
            
            # Write alteration zoning analysis
            f.write("\n\nAlteration Zoning Analysis:\n")
            f.write("--------------------------\n")
            if all(m in mineral_maps for m in ['chlorite', 'sericite', 'carbonate']):
                # Calculate zoning statistics
                ch_sr_ratio_valid = chl_ser_ratio[~np.isnan(chl_ser_ratio)]
                if len(ch_sr_ratio_valid) > 0:
                    high_propylitic = np.sum(ch_sr_ratio_valid > 2) / len(ch_sr_ratio_valid) * 100
                    high_phyllic = np.sum(ch_sr_ratio_valid < 0.5) / len(ch_sr_ratio_valid) * 100
                    balanced = np.sum((ch_sr_ratio_valid >= 0.5) & (ch_sr_ratio_valid <= 2)) / len(ch_sr_ratio_valid) * 100
                    
                    f.write("\nChlorite-Sericite Ratio Analysis:\n")
                    f.write(f"Propylitic-dominant zones (Chl/Ser > 2): {high_propylitic:.2f}%\n")
                    f.write(f"Phyllic-dominant zones (Chl/Ser < 0.5): {high_phyllic:.2f}%\n")
                    f.write(f"Mixed alteration zones (0.5 ≤ Chl/Ser ≤ 2): {balanced:.2f}%\n")
                
                # Calculate carbonate alteration statistics
                carb_ratio_valid = carb_alt_ratio[~np.isnan(carb_alt_ratio)]
                if len(carb_ratio_valid) > 0:
                    high_carbonate = np.sum(carb_ratio_valid > 0.4) / len(carb_ratio_valid) * 100
                    medium_carbonate = np.sum((carb_ratio_valid > 0.2) & (carb_ratio_valid <= 0.4)) / len(carb_ratio_valid) * 100
                    low_carbonate = np.sum(carb_ratio_valid <= 0.2) / len(carb_ratio_valid) * 100
                    
                    f.write("\nCarbonate Alteration Analysis:\n")
                    f.write(f"High carbonate zones (Carb/Total > 0.4): {high_carbonate:.2f}%\n")
                    f.write(f"Medium carbonate zones (0.2 < Carb/Total ≤ 0.4): {medium_carbonate:.2f}%\n")
                    f.write(f"Low carbonate zones (Carb/Total ≤ 0.2): {low_carbonate:.2f}%\n")
            else:
                f.write("Could not calculate alteration zoning - missing required mineral maps.\n")
            
            # Write mineral correlations
            f.write("\n\nMineral Correlations:\n")
            f.write("--------------------\n")
            if len(mineral_maps) >= 2:
                correlations = {}
                for i, (mineral1, map1) in enumerate(mineral_maps.items()):
                    m1_flat = map1.flatten()
                    valid1 = ~np.isnan(m1_flat)
                    
                    for mineral2, map2 in list(mineral_maps.items())[i+1:]:
                        m2_flat = map2.flatten()
                        valid2 = ~np.isnan(m2_flat)
                        
                        valid_both = valid1 & valid2
                        if np.sum(valid_both) > 100:  # Need sufficient valid pixels
                            corr = np.corrcoef(m1_flat[valid_both], m2_flat[valid_both])[0, 1]
                            correlations[f"{mineral1}-{mineral2}"] = corr
                            f.write(f"{mineral1.title()} - {mineral2.title()}: {corr:.3f}\n")
                
                # Find highest correlations
                if correlations:
                    f.write("\nStrongest Positive Correlations:\n")
                    pos_corrs = {k: v for k, v in correlations.items() if v > 0}
                    sorted_pos = sorted(pos_corrs.items(), key=lambda x: x[1], reverse=True)
                    for pair, corr in sorted_pos[:3]:
                        f.write(f"{pair}: {corr:.3f}\n")
                    
                    f.write("\nStrongest Negative Correlations:\n")
                    neg_corrs = {k: v for k, v in correlations.items() if v < 0}
                    sorted_neg = sorted(neg_corrs.items(), key=lambda x: x[1])
                    for pair, corr in sorted_neg[:3]:
                        f.write(f"{pair}: {corr:.3f}\n")
            else:
                f.write("Insufficient mineral maps to calculate correlations.\n")
            
            # Add summary section
            f.write("\n\nSummary:\n")
            f.write("--------\n")
            f.write("Most prevalent alteration minerals:\n")
            
            # Sort minerals by coverage
            mineral_coverage = {}
            for mineral, mineral_map in mineral_maps.items():
                valid_pixels = mineral_map[~np.isnan(mineral_map)]
                if len(valid_pixels) > 0:
                    coverage = np.sum(valid_pixels > 0) / len(valid_pixels) * 100
                    mineral_coverage[mineral] = coverage
            
            # Sort and write top 3
            sorted_minerals = sorted(mineral_coverage.items(), key=lambda x: x[1], reverse=True)
            for mineral_name, coverage in sorted_minerals[:3]:
                f.write(f"- {mineral_name.title()}: {coverage:.2f}%\n")
            
            f.write("\nMost prevalent alteration assemblages:\n")
            # Sort assemblages by coverage
            assemblage_coverage = {}
            for assemblage, assemblage_map in assemblage_maps.items():
                valid_pixels = assemblage_map[~np.isnan(assemblage_map)]
                if len(valid_pixels) > 0:
                    coverage = np.sum(valid_pixels > 0) / len(valid_pixels) * 100
                    assemblage_coverage[assemblage] = coverage
            
            # Sort and write top 3
            sorted_assemblages = sorted(assemblage_coverage.items(), key=lambda x: x[1], reverse=True)
            for assemblage_name, coverage in sorted_assemblages[:3]:
                f.write(f"- {assemblage_name.replace('_', ' ').title()}: {coverage:.2f}%\n")
            
            # Add geological interpretation
            f.write("\n\nGeological Interpretation:\n")
            f.write("-------------------------\n")
            
            # Determine dominant alteration type
            if sorted_assemblages:
                dominant_assemblage = sorted_assemblages[0][0]
                
                if dominant_assemblage == 'propylitic' or dominant_assemblage == 'chlorite_epidote':
                    f.write("The scene is dominated by propylitic alteration characterized by chlorite and epidote. This typically indicates:\n")
                    f.write("- Distal alteration halo around intrusive centers\n")
                    f.write("- Lower temperature hydrothermal fluid interaction\n")
                    f.write("- Possible transition to higher temperature alteration zones in proximity\n")
                
                elif dominant_assemblage == 'phyllic' or dominant_assemblage == 'sericite_pyrite':
                    f.write("The scene is dominated by phyllic (sericite-pyrite) alteration. This typically indicates:\n")
                    f.write("- Moderate to high temperature hydrothermal alteration\n")
                    f.write("- Proximity to potential mineralized zones\n")
                    f.write("- Acidic hydrothermal fluids\n")
                
                elif dominant_assemblage == 'argillic':
                    f.write("The scene is dominated by argillic alteration. This typically indicates:\n")
                    f.write("- Shallow level hydrothermal or steam-heated alteration\n")
                    f.write("- Low to moderate temperature alteration\n")
                    f.write("- Possible epithermal gold system\n")
                
                elif dominant_assemblage == 'advanced_argillic':
                    f.write("The scene is dominated by advanced argillic alteration. This typically indicates:\n")
                    f.write("- High-sulfidation epithermal system\n")
                    f.write("- Very acidic fluids\n")
                    f.write("- Possible proximity to intrusive center\n")
                
                elif dominant_assemblage == 'carbonate_alteration':
                    f.write("The scene is dominated by carbonate alteration. This typically indicates:\n")
                    f.write("- Possible propylitic alteration or carbonate replacement\n")
                    f.write("- Near-neutral pH fluids\n")
                    f.write("- Possible distal alteration or lower temperature zones\n")
                    
                elif dominant_assemblage == 'sulfide_alteration':
                    f.write("The scene is dominated by sulfide-rich alteration. This typically indicates:\n")
                    f.write("- Potential for sulfide mineralization\n")
                    f.write("- Hydrothermal activity related to intrusive rocks\n")
                    f.write("- Possible economic mineralization\n")
            else:
                f.write("Insufficient data to determine the dominant alteration style.\n")
        
        # Save CSV data
        import csv
        with open(csv_file, 'w', newline='') as csvf:
            writer = csv.writer(csvf)
            writer.writerow(csv_header)
            writer.writerows(csv_data)
        
        logger.info(f"Generated alteration minerals report at {report_file}")
        logger.info(f"Generated alteration minerals data CSV at {csv_file}")
        
        return report_file

    def generate_alteration_report(self, output_dir: Path):
        """Generate comprehensive alteration mapping report"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report_file = output_dir / 'alteration_analysis_report.txt'
        with open(report_file, 'w') as f:
            f.write("ASTER Alteration Mapping Analysis Report\n")
            f.write("=====================================\n\n")
            
            f.write("Scene Information:\n")
            f.write("-----------------\n")
            if self.metadata:
                f.write(f"Acquisition Date: {self.metadata.acquisition_date}\n")
                f.write(f"Solar Azimuth: {self.metadata.solar_azimuth}°\n")
                f.write(f"Solar Elevation: {self.metadata.solar_elevation}°\n")
                f.write(f"Cloud Cover: {self.metadata.cloud_cover}%\n\n")
            
            f.write("Alteration Analysis:\n")
            f.write("-------------------\n")
            
            # Analyze each alteration type
            for alt_type in AlterationIndices:
                try:
                    index_map, confidence = self.calculate_alteration_index(alt_type)
                    
                    # Calculate statistics
                    valid_pixels = index_map[~np.isnan(index_map)]
                    if len(valid_pixels) > 0:
                        coverage = np.sum(valid_pixels > 0) / len(valid_pixels) * 100
                        mean_intensity = np.mean(valid_pixels[valid_pixels > 0])
                        mean_confidence = np.mean(confidence[~np.isnan(confidence)])
                        
                        f.write(f"\n{alt_type.value.title()}:\n")
                        f.write(f"Coverage: {coverage:.2f}%\n")
                        f.write(f"Mean Intensity: {mean_intensity:.3f}\n")
                        f.write(f"Mean Confidence: {mean_confidence:.3f}\n")
                        # Create visualization for this alteration type
                        plt.figure(figsize=(12, 4))
                        
                        plt.subplot(121)
                        plt.imshow(index_map, cmap='viridis')
                        plt.colorbar(label='Index Value')
                        plt.title(f'{alt_type.value.title()} Distribution')
                        
                        plt.subplot(122)
                        plt.imshow(confidence, cmap='RdYlGn')
                        plt.colorbar(label='Confidence')
                        plt.title('Confidence Map')
                        
                        plt.tight_layout()
                        plt.savefig(output_dir / f'{alt_type.value}_distribution.png',
                                  dpi=300, bbox_inches='tight')
                        plt.close()
                        
                        # Save as GeoTIFF
                        if self.metadata and hasattr(self.metadata, 'bounds'):
                            bounds = self.metadata.bounds
                            transform = from_bounds(
                                bounds.west, bounds.south,
                                bounds.east, bounds.north,
                                index_map.shape[1], index_map.shape[0]
                            )
                            
                            with rasterio.open(
                                output_dir / f'{alt_type.value}_map.tif',
                                'w',
                                driver='GTiff',
                                height=index_map.shape[0],
                                width=index_map.shape[1],
                                count=2,
                                dtype=index_map.dtype,
                                crs=CRS.from_epsg(4326),
                                transform=transform,
                                nodata=np.nan
                            ) as dst:
                                dst.write(index_map, 1)
                                dst.write(confidence, 2)
                                dst.set_band_description(1, f"{alt_type.value} index")
                                dst.set_band_description(2, "Confidence map")
                                
                                if self.metadata:
                                    dst.update_tags(
                                        acquisition_date=self.metadata.acquisition_date,
                                        solar_azimuth=str(self.metadata.solar_azimuth),
                                        solar_elevation=str(self.metadata.solar_elevation)
                                    )
                        
                except Exception as e:
                    f.write(f"\nError processing {alt_type.value}: {str(e)}\n")
                    logger.error(f"Error processing {alt_type.value}: {e}")
                    continue
            
            # Add summary section
            f.write("\nSummary:\n")
            f.write("--------\n")
            f.write("Most prevalent alteration types:\n")
            
            # Sort alteration types by coverage
            alteration_coverage = {}
            for alt_type in AlterationIndices:
                try:
                    index_map, _ = self.calculate_alteration_index(alt_type)
                    valid_pixels = index_map[~np.isnan(index_map)]
                    if len(valid_pixels) > 0:
                        coverage = np.sum(valid_pixels > 0) / len(valid_pixels) * 100
                        alteration_coverage[alt_type.value] = coverage
                except Exception:
                    continue
            
            # Sort and write top 3
            sorted_alterations = sorted(alteration_coverage.items(),
                                     key=lambda x: x[1], reverse=True)
            for alt_name, coverage in sorted_alterations[:3]:
                f.write(f"- {alt_name.title()}: {coverage:.2f}%\n")
        
        logger.info(f"Generated alteration report at {report_file}")
        return report_file

def main():
    """Example usage of the ASTER Geological Mapper"""
    from dataclasses import dataclass
    
    @dataclass
    class DummyProcessor:
        """Dummy processor class for demonstration"""
        reflectance_data: dict
        metadata: dict
    
    # Create dummy data for demonstration
    dummy_data = {
        1: np.random.random((100, 100)),
        2: np.random.random((100, 100)),
        3: np.random.random((100, 100)),
        4: np.random.random((100, 100)),
        5: np.random.random((100, 100)),
        6: np.random.random((100, 100)),
        7: np.random.random((100, 100)),
        8: np.random.random((100, 100)),
        9: np.random.random((100, 100))
    }
    
    dummy_metadata = {
        'acquisition_date': '2024-02-10',
        'solar_azimuth': 152.96,
        'solar_elevation': 53.52,
        'cloud_cover': 3.0,
        'bounds': {
            'west': -2.67,
            'east': -1.99,
            'south': 8.76,
            'north': 9.43
        }
    }
    
    # Create dummy processor
    processor = DummyProcessor(
        reflectance_data=dummy_data,
        metadata=dummy_metadata
    )
    
    # Initialize geological mapper
    mapper = ASTER_Geological_Mapper(processor)
    
    # Create output directory
    output_dir = Path("geological_mapping_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate alteration maps
    for alt_type in AlterationIndices:
        try:
            index_map, confidence = mapper.calculate_alteration_index(alt_type)
            logger.info(f"Generated {alt_type.value} alteration map")
        except Exception as e:
            logger.error(f"Error processing {alt_type.value}: {e}")
    
    # Create composite map
    mapper.create_composite_map(output_dir)
    
    # Generate report
    report_file = mapper.generate_alteration_report(output_dir)
    logger.info(f"Processing complete. Report saved to {report_file}")

if __name__ == "__main__":
    main()