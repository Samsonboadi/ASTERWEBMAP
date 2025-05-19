#aster_advanced_analysis.py
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Tuple, List, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import ndimage, stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import rasterio
from skimage import measure
from rasterio.features import shapes
from rasterio.transform import from_bounds
import geopandas as gpd
from shapely.geometry import shape, mapping
import json
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from skimage import feature, filters, morphology, segmentation, draw
# Assuming ASTER_L2_Processor and ASTER_Geological_Mapper are defined in a module named aster_processor
from .aster_l2_processor import ASTER_L2_Processor 
#from .aster_geological_mapper import ASTER_Geological_Mapper
from .aster_geological_mapper import ASTER_Geological_Mapper, AlterationIndices, GeologicalFeatures
from .aster_l2_processor import ASTER_L2_Processor, MineralIndices, SceneMetadata

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ASTER_Advanced_Analysis:
    def __init__(self, base_processor, geological_mapper):
        """
        Initialize advanced analysis tools
        
        Parameters:
        -----------
        base_processor : ASTER_L2_Processor
            Base ASTER processor instance
        geological_mapper : ASTER_Geological_Mapper
            Geological mapper instance
        """
        self.processor = base_processor
        self.geological_mapper = geological_mapper
        self.reflectance_data = base_processor.reflectance_data
        self.metadata = base_processor.metadata

    def create_band_ratio_matrix(self, output_dir: Path):
        """
        Create and analyze all possible band ratios
        
        Parameters:
        -----------
        output_dir : Path
            Output directory for saving results
        """
        bands = sorted(self.reflectance_data.keys())
        n_bands = len(bands)
        ratio_matrix = np.zeros((n_bands, n_bands))
        correlation_matrix = np.zeros((n_bands, n_bands))
        
        for i, band1 in enumerate(bands):
            for j, band2 in enumerate(bands):
                if i != j:
                    # Calculate band ratio
                    b1 = self.reflectance_data[band1]
                    b2 = self.reflectance_data[band2]
                    
                    valid_mask = (b1 > 0) & (b2 > 0) & ~np.isnan(b1) & ~np.isnan(b2)
                    if np.any(valid_mask):
                        ratio = b1[valid_mask] / b2[valid_mask]
                        ratio_matrix[i, j] = np.median(ratio)
                        correlation_matrix[i, j] = stats.pearsonr(
                            b1[valid_mask].flatten(),
                            b2[valid_mask].flatten()
                        )[0]
        
        # Create visualizations
        plt.figure(figsize=(15, 6))
        
        plt.subplot(121)
        sns.heatmap(ratio_matrix, annot=True, fmt='.2f', 
                   xticklabels=bands, yticklabels=bands)
        plt.title('Band Ratio Matrix')
        
        plt.subplot(122)
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f',
                   xticklabels=bands, yticklabels=bands)
        plt.title('Band Correlation Matrix')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'band_ratio_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save numerical results
        results = pd.DataFrame({
            'Band_Combinations': [f'B{b1}/B{b2}' for b1 in bands for b2 in bands if b1 != b2],
            'Median_Ratio': [ratio_matrix[i, j] for i in range(n_bands) 
                           for j in range(n_bands) if i != j],
            'Correlation': [correlation_matrix[i, j] for i in range(n_bands)
                          for j in range(n_bands) if i != j]
        })
        results.to_csv(output_dir / 'band_ratio_analysis.csv', index=False)

    def perform_spectral_clustering(self, n_clusters: int = 5) -> np.ndarray:
        """
        Perform spectral clustering on the image
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters to identify
            
        Returns:
        --------
        np.ndarray
            Cluster labels array
        """
        # Stack all bands
        bands = []
        for band in sorted(self.reflectance_data.keys()):
            data = self.reflectance_data[band]
            bands.append(data.flatten())
        
        X = np.vstack(bands).T
        valid_mask = ~np.isnan(X).any(axis=1)
        X = X[valid_mask]
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        
        # Reshape back to image dimensions
        full_labels = np.zeros(bands[0].shape)
        full_labels.flat[valid_mask] = labels
        
        return full_labels

    def extract_geological_features(self, output_dir: Path):
        """
        Extract and vectorize geological features
        
        Parameters:
        -----------
        output_dir : Path
            Output directory for saving results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        features = []
        
        try:
            # Get cluster labels
            logger.info("Performing spectral clustering...")
            try:
                cluster_labels = self.perform_spectral_clustering()
                logger.info(f"Cluster labels shape: {cluster_labels.shape if hasattr(cluster_labels, 'shape') else 'unknown'}")
                
                # Verify cluster labels are 2D
                if not hasattr(cluster_labels, 'ndim') or cluster_labels.ndim != 2:
                    logger.warning(f"Cluster labels have incorrect dimensions: {getattr(cluster_labels, 'ndim', 'unknown')}")
                    # Try to reshape if possible
                    if hasattr(self.reflectance_data, 'values') and len(self.reflectance_data) > 0:
                        first_band = next(iter(self.reflectance_data.values()))
                        if hasattr(first_band, 'shape'):
                            cluster_labels = np.reshape(cluster_labels, first_band.shape)
                            logger.info(f"Reshaped cluster labels to {cluster_labels.shape}")
                
                # Process clusters
                if hasattr(cluster_labels, 'shape') and cluster_labels.ndim == 2:
                    logger.info("Processing cluster features...")
                    for label in np.unique(cluster_labels):
                        mask = cluster_labels == label
                        try:
                            for geom, value in shapes(mask.astype(np.uint8), mask=mask):
                                features.append({
                                    'geometry': shape(geom),
                                    'properties': {
                                        'type': 'spectral_cluster',
                                        'cluster_id': int(label)
                                    }
                                })
                        except Exception as e:
                            logger.error(f"Error processing cluster {label}: {str(e)}")
                else:
                    logger.warning("Skipping cluster processing due to invalid dimensions")
            except Exception as e:
                logger.error(f"Error in spectral clustering: {str(e)}")
            
            # Get alteration zones
            logger.info("Calculating alteration index...")
            try:
                alteration_map, confidence = self.geological_mapper.calculate_alteration_index(
                    AlterationIndices.ADVANCED_ARGILLIC
                )
                logger.info(f"Alteration map shape: {alteration_map.shape if hasattr(alteration_map, 'shape') else 'unknown'}")
                
                # Process alteration zones if valid shape
                if hasattr(alteration_map, 'shape') and alteration_map.ndim == 2:
                    logger.info("Processing alteration zones...")
                    valid_data = alteration_map[~np.isnan(alteration_map)]
                    if len(valid_data) > 0:
                        threshold = np.percentile(valid_data, 75)
                        alteration_mask = alteration_map > threshold
                        try:
                            for geom, value in shapes(alteration_mask.astype(np.uint8), mask=alteration_mask):
                                features.append({
                                    'geometry': shape(geom),
                                    'properties': {
                                        'type': 'alteration_zone',
                                        'intensity': float(np.mean(alteration_map[alteration_mask]))
                                    }
                                })
                        except Exception as e:
                            logger.error(f"Error vectorizing alteration zones: {str(e)}")
                    else:
                        logger.warning("No valid data in alteration map")
                else:
                    logger.warning("Skipping alteration zones due to invalid dimensions")
            except Exception as e:
                logger.error(f"Error calculating alteration index: {str(e)}")
            
            # Get lineaments
            logger.info("Detecting geological features...")
            try:
                lineaments = self.geological_mapper.detect_geological_features(
                    GeologicalFeatures.LINEAMENTS
                )
                logger.info(f"Lineaments shape: {lineaments.shape if hasattr(lineaments, 'shape') else 'unknown'}")
                
                # Process lineaments if valid shape
                if hasattr(lineaments, 'shape') and lineaments.ndim == 2:
                    logger.info("Processing lineaments...")
                    lineament_mask = lineaments > 0
                    try:
                        for geom, value in shapes(lineament_mask.astype(np.uint8), mask=lineament_mask):
                            features.append({
                                'geometry': shape(geom),
                                'properties': {
                                    'type': 'lineament',
                                    'length': float(shape(geom).length)
                                }
                            })
                    except Exception as e:
                        logger.error(f"Error vectorizing lineaments: {str(e)}")
                else:
                    logger.warning("Skipping lineaments due to invalid dimensions")
            except Exception as e:
                logger.error(f"Error detecting lineaments: {str(e)}")
            
            # Create GeoDataFrame if we have features
            if features:
                logger.info(f"Creating GeoDataFrame with {len(features)} features")
                try:
                    gdf = gpd.GeoDataFrame.from_features(features)
                    
                    # Set CRS if available
                    if self.metadata and hasattr(self.metadata, 'crs'):
                        #gdf.set_crs(self.metadata.crs, inplace=True)
                        gdf.set_crs("EPSG:4326", inplace=True)
                    
                    # Save to file
                    output_file = output_dir / 'geological_features.gpkg'
                    gdf.to_file(output_file, driver='GPKG')
                    logger.info(f"Saved geological features to {output_file}")
                    
                    # Create visualization
                    logger.info("Creating visualization...")
                    fig, ax = plt.subplots(figsize=(12, 8))
                    
                    # Plot clusters
                    if 'type' in gdf.columns:
                        cluster_gdf = gdf[gdf['type'] == 'spectral_cluster']
                        if not cluster_gdf.empty and 'cluster_id' in cluster_gdf.columns:
                            cluster_gdf.plot(ax=ax, column='cluster_id', alpha=0.5, 
                                            legend=True, legend_kwds={'label': 'Spectral Clusters'})
                        
                        # Plot alteration zones
                        alteration_gdf = gdf[gdf['type'] == 'alteration_zone']
                        if not alteration_gdf.empty:
                            alteration_gdf.plot(ax=ax, color='red', alpha=0.3, label='Alteration Zones')
                        
                        # Plot lineaments
                        lineament_gdf = gdf[gdf['type'] == 'lineament']
                        if not lineament_gdf.empty:
                            lineament_gdf.plot(ax=ax, color='black', linewidth=0.5, label='Lineaments')
                    
                    plt.title('Extracted Geological Features')
                    plt.legend()
                    plt.axis('equal')
                    
                    plt.savefig(output_dir / 'geological_features.png', dpi=300, bbox_inches='tight')
                    plt.close()
                    logger.info("Visualization saved")
                except Exception as e:
                    logger.error(f"Error creating or saving GeoDataFrame: {str(e)}")
            else:
                logger.warning("No features extracted, skipping GeoDataFrame creation")
                
        except Exception as e:
            logger.error(f"Unexpected error in extract_geological_features: {str(e)}")
            logger.exception("Detailed stack trace:")

    def generate_mineral_potential_map(self, output_dir: Path):
        """
        Generate mineral potential map combining multiple indicators
        
        Parameters:
        -----------
        output_dir : Path
            Output directory for saving results
        """
        # Get various mineral and alteration indicators
        indicators = {}
        
        # Add mineral indices
        for mineral in MineralIndices:
            index_map, confidence = self.processor.calculate_mineral_index(mineral)
            indicators[f'{mineral.value}_index'] = index_map * confidence
        
        # Add alteration indices
        for alteration in AlterationIndices:
            index_map, confidence = self.geological_mapper.calculate_alteration_index(alteration)
            indicators[f'{alteration.value}_index'] = index_map * confidence
        
        # Add structural features
        lineaments = self.geological_mapper.detect_geological_features(
            GeologicalFeatures.LINEAMENTS
        )
        indicators['structural'] = lineaments
        
        # Normalize all indicators
        normalized_indicators = {}
        for name, indicator in indicators.items():
            valid_data = indicator[~np.isnan(indicator)]
            if len(valid_data) > 0:
                min_val, max_val = np.percentile(valid_data, [2, 98])
                normalized = np.clip((indicator - min_val) / (max_val - min_val), 0, 1)
                normalized_indicators[name] = normalized
        
        # Calculate weights using PCA
        data_stack = np.dstack(list(normalized_indicators.values()))
        valid_mask = ~np.isnan(data_stack).any(axis=2)
        
        if np.any(valid_mask):
            # Reshape for PCA
            data_2d = data_stack[valid_mask].reshape(-1, len(normalized_indicators))
            
            # Perform PCA
            pca = PCA()
            pca.fit(data_2d)
            
            # Use explained variance ratios as weights
            weights = pca.explained_variance_ratio_
            
            # Calculate weighted sum
            potential_map = np.zeros_like(list(normalized_indicators.values())[0])
            for i, (name, indicator) in enumerate(normalized_indicators.items()):
                potential_map += indicator * weights[i]
            
            # Normalize final map
            potential_map = (potential_map - potential_map[valid_mask].min()) / \
                          (potential_map[valid_mask].max() - potential_map[valid_mask].min())
            
            # Create custom colormap
            colors = ['#440154', '#3B528B', '#21908C', '#5DC863', '#FDE725']
            n_bins = 256
            cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
            
            # Create visualization
            plt.figure(figsize=(12, 8))
            
            plt.imshow(potential_map, cmap=cmap)
            plt.colorbar(label='Mineral Potential')
            plt.title('Mineral Potential Map')
            
            # Add high potential zone outlines
            high_potential = potential_map > np.percentile(potential_map[valid_mask], 90)
            contours = measure.find_contours(high_potential, 0.5)
            for contour in contours:
                plt.plot(contour[:, 1], contour[:, 0], 'k-', linewidth=0.5)
            
            plt.savefig(output_dir / 'mineral_potential_map.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save as GeoTIFF
            if self.metadata and hasattr(self.metadata, 'bounds'):
                bounds = self.metadata.bounds
                transform = from_bounds(
                    bounds.west, bounds.south,
                    bounds.east, bounds.north,
                    potential_map.shape[1], potential_map.shape[0]
                )
                
                with rasterio.open(
                    output_dir / 'mineral_potential_map.tif',
                    'w',
                    driver='GTiff',
                    height=potential_map.shape[0],
                    width=potential_map.shape[1],
                    count=1,
                    dtype=potential_map.dtype,
                    crs=self.metadata.crs,
                    transform=transform,
                    nodata=np.nan
                ) as dst:
                    dst.write(potential_map, 1)
                    dst.set_band_description(1, "Mineral Potential Index")
                    
                    # Add metadata about the indicators used
                    metadata = {
                        'indicators_used': list(normalized_indicators.keys()),
                        'indicator_weights': weights.tolist()
                    }
                    dst.update_tags(**metadata)
            
            # Generate summary statistics
            summary_stats = {
                'mean_potential': float(np.mean(potential_map[valid_mask])),
                'high_potential_area_percent': float(np.sum(high_potential) / np.sum(valid_mask) * 100),
                'indicator_weights': {name: float(weight) 
                                    for name, weight in zip(normalized_indicators.keys(), weights)}
            }
            
            # Save summary statistics
            with open(output_dir / 'potential_map_statistics.json', 'w') as f:
                json.dump(summary_stats, f, indent=4)

def main():
    # Example usage
    base_processor = ASTER_L2_Processor("vnir.hdf", "swir.hdf")
    geological_mapper = ASTER_Geological_Mapper(base_processor)
    advanced_analysis = ASTER_Advanced_Analysis(base_processor, geological_mapper)
    
    output_dir = Path("advanced_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Perform analyses
    advanced_analysis.create_band_ratio_matrix(output_dir)
    advanced_analysis.extract_geological_features(output_dir)
    advanced_analysis.generate_mineral_potential_map(output_dir)

if __name__ == "__main__":
    main()