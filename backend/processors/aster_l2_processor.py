"""
ASTER L2 Surface Reflectance Processor Module
Handles processing of ASTER L2 data including reflectance data and mineral indices
"""

import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Union
import logging
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from enum import Enum
from scipy.ndimage import zoom
from dataclasses import dataclass
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import re
from datetime import datetime
from enums import MineralIndices, GoldPathfinderIndices  # Ensure this import is present
# Set up global logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('aster_processing.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class GeographicBounds:
    """Geographic bounds of the scene"""
    west: float
    east: float
    south: float
    north: float

@dataclass
class SceneMetadata:
    """Scene metadata information"""
    bounds: GeographicBounds
    solar_azimuth: float
    solar_elevation: float
    cloud_cover: float
    acquisition_date: str
    utm_zone: Optional[int] = None

class MineralIndices(Enum):
    """Enhanced mineral mapping indices"""
    ALUNITE = "alunite"
    KAOLINITE = "kaolinite"
    CALCITE = "calcite"
    DOLOMITE = "dolomite"
    CHLORITE = "chlorite"
    EPIDOTE = "epidote"
    MUSCOVITE = "muscovite"
    ILLITE = "illite"
    IRON_OXIDE = "iron_oxide"
    FERROUS_IRON = "ferrous_iron"
    FERRIC_OXIDE = "ferric_oxide"
    HYDROXYL = "hydroxyl"
    PEGMATITE = "pegmatite"
    SPODUMENE = "spodumene"
    LABRADORITE = "labradorite"
    FELDSPAR = "feldspar"
    PYRITE = "pyrite"
    CHALCOPYRITE = "chalcopyrite"
    SPHALERITE = "sphalerite"
    GALENA = "galena"
    SERICITE = "sericite"
    CARBONATE = "carbonate"

class ASTER_L2_Processor:
    def __init__(self, vnir_file: Union[str, Path], swir_file: Union[str, Path],
                 metadata: Optional[SceneMetadata] = None):
        """Initialize ASTER L2 Surface Reflectance processor"""
        self.vnir_file = Path(vnir_file)
        self.swir_file = Path(swir_file)
        self.metadata = metadata

        logger.info(f"Initializing processor with VNIR: {self.vnir_file}, SWIR: {self.swir_file}")

        if not self.vnir_file.exists():
            raise FileNotFoundError(f"VNIR file not found: {self.vnir_file}")
        if not self.swir_file.exists():
            raise FileNotFoundError(f"SWIR file not found: {self.swir_file}")

        # Define band mappings - base names without prefixes
        self.vnir_datasets = {
            1: 'Band1',
            2: 'Band2',
            3: 'Band3N'
        }
        self.swir_datasets = {
            4: 'Band4',
            5: 'Band5',
            6: 'Band6',
            7: 'Band7',
            8: 'Band8',
            9: 'Band9'
        }

        # Initialize data storage
        self.raw_reflectance_data = {}
        self.reflectance_data = {}
        self.qa_data = {}

        # Define enhanced mineral indices with optimized band ratios
        self.mineral_indices = {
            MineralIndices.ALUNITE: {
                'ratios': [(4, 6), (4, 5)],
                'threshold': 0.50,
                'description': 'Al-OH absorption features'
            },
            MineralIndices.KAOLINITE: {
                'ratios': [(4, 5), (8, 6)],
                'threshold': 0.50,
                'description': 'Al-OH and Si-O absorption'
            },
            MineralIndices.CALCITE: {
                'ratios': [(6, 8), (9, 8)],
                'threshold': 0.50,
                'description': 'CO3 absorption features'
            },
            MineralIndices.IRON_OXIDE: {
                'ratios': [(2, 1), (4, 3)],
                'threshold': 0.50,
                'description': 'Fe3+ absorption features'
            },
            MineralIndices.FERROUS_IRON: {
                'ratios': [(5, 3), (1, 2)],
                'threshold': 0.50,
                'description': 'Fe2+ absorption features'
            },
            MineralIndices.HYDROXYL: {
                'ratios': [(4, 6), (7, 6)],
                'threshold': 0.50,
                'description': 'OH bearing minerals'
            },
            MineralIndices.SERICITE: {
                'ratios': [(4, 5), (7, 6)],
                'threshold': 0.50,
                'description': 'Sericitic alteration'
            },
            MineralIndices.PEGMATITE: {
                'ratios': [(4, 6), (7, 8), (5, 9)],
                'threshold': 0.50,
                'description': 'Muscovite and feldspar features'
            },
            MineralIndices.SPODUMENE: {
                'ratios': [(6, 8), (7, 9), (4, 5)],
                'threshold': 0.50,
                'description': 'Li-bearing mineral features'
            },
            MineralIndices.LABRADORITE: {
                'ratios': [(1, 3), (5, 7), (4, 8)],
                'threshold': 0.50,
                'description': 'Plagioclase features'
            },
            MineralIndices.FELDSPAR: {
                'ratios': [(4, 5), (7, 8), (6, 9)],
                'threshold': 0.50,
                'description': 'K-feldspar features'
            },
            MineralIndices.PYRITE: {
                'ratios': [(2, 1), (3, 4)],
                'threshold': 0.45,
                'description': 'Pyrite (iron sulfide)'
            },
            MineralIndices.CHALCOPYRITE: {
                'ratios': [(2, 1), (3, 2)],
                'threshold': 0.50,
                'description': 'Chalcopyrite (copper iron sulfide)'
            },
            MineralIndices.SPHALERITE: {
                'ratios': [(4, 5), (7, 8)],
                'threshold': 0.50,
                'description': 'Sphalerite (zinc sulfide)'
            },
            MineralIndices.GALENA: {
                'ratios': [(5, 4), (2, 1)],
                'threshold': 0.45,
                'description': 'Galena (lead sulfide)'
            },
            MineralIndices.CHLORITE: {
                'ratios': [(7, 8), (6, 8)],
                'threshold': 0.50,
                'description': 'Chlorite (magnesium-iron phyllosilicate)'
            },
            MineralIndices.SERICITE: {
                'ratios': [(4, 6), (5, 6)],
                'threshold': 0.50,
                'description': 'Sericite (fine-grained muscovite)'
            },
            MineralIndices.CARBONATE: {
                'ratios': [(8, 9), (6, 8)],
                'threshold': 0.50,
                'description': 'Carbonate minerals (calcite, dolomite)'
            },
            MineralIndices.EPIDOTE: {
                'ratios': [(7, 8), (6, 9)],
                'threshold': 0.50,
                'description': 'Epidote (calcium aluminum iron sorosilicate)'
            },
            MineralIndices.MUSCOVITE: {
                'ratios': [(4, 6), (7, 5)],
                'threshold': 0.50,
                'description': 'Muscovite (Al-rich mica)'
            },
            MineralIndices.ILLITE: {
                'ratios': [(5, 6), (4, 7)],
                'threshold': 0.50,
                'description': 'Illite (non-expanding clay)'
            },
            MineralIndices.FERRIC_OXIDE: {
                'ratios': [(2, 1), (4, 2)],
                'threshold': 0.50,
                'description': 'Ferric oxide (Fe3+)'
            },
            MineralIndices.DOLOMITE: {
                'ratios': [(6, 8), (9, 8)],
                'threshold': 0.50,
                'description': 'Dolomite (carbonate mineral)'
            }
        }

        logger.info("Starting data processing...")
        self.load_data()
        logger.info(f"Raw data loaded, bands available: {list(self.raw_reflectance_data.keys())}")

        self.resample_data()
        logger.info(f"Data resampled, bands available: {list(self.reflectance_data.keys())}")

        if not self.reflectance_data:
            raise RuntimeError("No data was successfully resampled")

        if self.metadata and self.metadata.cloud_cover > 0:
            self.apply_cloud_mask()

    def read_met_file(self, met_file: Path) -> Dict:
        """
        Read georeference information from MET file

        Parameters:
        -----------
        met_file : Path
            Path to the metadata file

        Returns:
        --------
        Dict
            Georeference information including bounds
        """
        try:
            met_file = Path(met_file)
            if not met_file.exists():
                logger.error(f"MET file not found: {met_file}")
                return None

            logger.info(f"Reading metadata from: {met_file}")

            with open(met_file, 'r') as f:
                content = f.read()

            gpolygon_pattern = r'GROUP\s+=\s+GPOLYGON(.*?)END_GROUP\s+=\s+GPOLYGON'
            gpolygon_match = re.search(gpolygon_pattern, content, re.DOTALL)

            coords = {'lats': [], 'lons': []}

            if gpolygon_match:
                logger.info("Found GPOLYGON section")
                gpolygon_content = gpolygon_match.group(1)

                lon_pattern = r'GRINGPOINTLONGITUDE.*?VALUE\s+=\s+\((.*?)\)'
                lon_match = re.search(lon_pattern, gpolygon_content, re.DOTALL)
                if lon_match:
                    lon_str = lon_match.group(1)
                    logger.info(f"Found longitude string: {lon_str}")
                    try:
                        coords['lons'] = [float(x.strip()) for x in lon_str.split(',')]
                        logger.info(f"Parsed longitudes: {coords['lons']}")
                    except Exception as e:
                        logger.error(f"Error parsing longitudes: {str(e)}")

                lat_pattern = r'GRINGPOINTLATITUDE.*?VALUE\s+=\s+\((.*?)\)'
                lat_match = re.search(lat_pattern, gpolygon_content, re.DOTALL)
                if lat_match:
                    lat_str = lat_match.group(1)
                    logger.info(f"Found latitude string: {lat_str}")
                    try:
                        coords['lats'] = [float(x.strip()) for x in lat_str.split(',')]
                        logger.info(f"Parsed latitudes: {coords['lats']}")
                    except Exception as e:
                        logger.error(f"Error parsing latitudes: {str(e)}")
            else:
                logger.error("Could not find GPOLYGON section in metadata")

            date_pattern = r'ACQUISITIONDATE.*?VALUE\s+=\s+"(.*?)"'
            date_match = re.search(date_pattern, content, re.DOTALL)
            acquisition_date = None
            if date_match:
                date_str = date_match.group(1)
                logger.info(f"Found acquisition date: {date_str}")
                try:
                    acquisition_date = date_str
                except Exception as e:
                    logger.error(f"Error parsing acquisition date: {str(e)}")

            solar_azimuth = None
            solar_elevation = None

            solar_az_pattern = r'SOLAZIMUTH.*?VALUE\s+=\s+(.*?)(?:\n|\r)'
            solar_az_match = re.search(solar_az_pattern, content, re.DOTALL)
            if solar_az_match:
                try:
                    solar_azimuth = float(solar_az_match.group(1))
                    logger.info(f"Found solar azimuth: {solar_azimuth}")
                except Exception as e:
                    logger.error(f"Error parsing solar azimuth: {str(e)}")

            solar_el_pattern = r'SOLELEVATION.*?VALUE\s+=\s+(.*?)(?:\n|\r)'
            solar_el_match = re.search(solar_el_pattern, content, re.DOTALL)
            if solar_el_match:
                try:
                    solar_elevation = float(solar_el_match.group(1))
                    logger.info(f"Found solar elevation: {solar_elevation}")
                except Exception as e:
                    logger.error(f"Error parsing solar elevation: {str(e)}")

            cloud_cover = None
            cloud_pattern = r'SCENECLOUDCOVERAGE.*?VALUE\s+=\s+(.*?)(?:\n|\r)'
            cloud_match = re.search(cloud_pattern, content, re.DOTALL)
            if cloud_match:
                try:
                    cloud_cover = float(cloud_match.group(1))
                    logger.info(f"Found cloud cover: {cloud_cover}%")
                except Exception as e:
                    logger.error(f"Error parsing cloud cover: {str(e)}")

            if coords['lats'] and coords['lons']:
                georef_info = {
                    'west': min(coords['lons']),
                    'east': max(coords['lons']),
                    'north': max(coords['lats']),
                    'south': min(coords['lats']),
                    'acquisition_date': acquisition_date or "unknown",
                    'solar_azimuth': solar_azimuth or 0.0,
                    'solar_elevation': solar_elevation or 0.0,
                    'cloud_cover': cloud_cover or 0.0
                }

                logger.info("Successfully extracted coordinates and metadata from MET file")
                logger.info(f"Bounds: N={georef_info['north']}, S={georef_info['south']}, "
                            f"E={georef_info['east']}, W={georef_info['west']}")
                return georef_info
            else:
                logger.error("Could not extract coordinates from MET file")
                return None

        except Exception as e:
            logger.error(f"Error reading MET file: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            return None

    def extract_georeference(self, hdf_file: Path) -> Dict:
        """Extract complete georeference information from HDF and MET files"""
        try:
            hdf_file = Path(hdf_file)
            # Try both .met and .hdf.met extensions
            met_file = hdf_file.with_suffix('.met')
            met_file_alt = hdf_file.with_name(hdf_file.name + '.met')  # For .hdf.met

            if met_file.exists():
                logger.info(f"Found MET file: {met_file}")
                met_info = self.read_met_file(met_file)
            elif met_file_alt.exists():
                logger.info(f"Found MET file with .hdf.met extension: {met_file_alt}")
                met_info = self.read_met_file(met_file_alt)
            else:
                logger.error(f"MET file not found at {met_file} or {met_file_alt}")
                met_info = None

            if met_info is None:
                logger.warning("Could not read MET file, falling back to HDF file")
                return self._extract_from_hdf(hdf_file)

            with Dataset(hdf_file, 'r') as hdf:
                for var_name in hdf.variables:
                    if 'Band' in var_name:
                        data = hdf.variables[var_name]
                        height, width = data.shape
                        break
                else:
                    logger.error("No band data found in HDF file")
                    raise ValueError("No band data found in HDF file")

            pixel_width = (met_info['east'] - met_info['west']) / width
            pixel_height = (met_info['north'] - met_info['south']) / height

            hemisphere = 'N' if met_info.get('north', 0) >= 0 else 'S'

            utm_zone = met_info.get('utm_zone')
            if utm_zone is None:
                center_lon = (met_info['east'] + met_info['west']) / 2
                utm_zone = int((center_lon + 180) / 6) + 1

            georef_info = {
                'transform': [
                    met_info['west'],
                    pixel_width,
                    0,
                    met_info['north'],
                    0,
                    -pixel_height
                ],
                'bounds': {
                    'west': met_info['west'],
                    'east': met_info['east'],
                    'north': met_info['north'],
                    'south': met_info['south']
                },
                'dimensions': {
                    'width': width,
                    'height': height
                },
                'utm_zone': utm_zone,
                'hemisphere': hemisphere,
                'solar_azimuth': met_info['solar_azimuth'],
                'solar_elevation': met_info['solar_elevation'],
                'cloud_cover': met_info['cloud_cover'],
                'acquisition_date': met_info['acquisition_date']
            }

            epsg = 32600 + utm_zone
            if hemisphere == 'S':
                epsg = 32700 + utm_zone
            logger.info(f"Setting CRS with EPSG: {epsg} (UTM Zone: {utm_zone}{hemisphere})")
            try:
                georef_info['crs'] = CRS.from_epsg(epsg)
            except Exception as e:
                logger.error(f"Failed to create CRS with EPSG {epsg}: {str(e)}")
                logger.info("Falling back to WGS84 (EPSG:4326)")
                georef_info['crs'] = CRS.from_epsg(4326)

            logger.info(f"Successfully extracted georeference info from MET file")
            logger.info(f"Bounds: {georef_info['bounds']}")
            logger.info(f"Transform: {georef_info['transform']}")
            logger.info(f"UTM Zone: {utm_zone}{hemisphere}")

            return georef_info

        except Exception as e:
            logger.error(f"Error extracting georeference: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            raise

    def _extract_from_hdf(self, hdf_file: Path) -> Dict:
        """Extract georeference information from HDF file as fallback"""
        try:
            with Dataset(hdf_file, 'r') as hdf:
                for var_name in hdf.variables:
                    if 'Band' in var_name:
                        data = hdf.variables[var_name]
                        height, width = data.shape
                        break
                else:
                    logger.error("No band data found in HDF file")
                    raise ValueError("No band data found in HDF file")

                if self.metadata and hasattr(self.metadata, 'bounds'):
                    bounds = self.metadata.bounds
                    ulx = bounds.west
                    uly = bounds.north
                    lrx = bounds.east
                    lry = bounds.south
                else:
                    logger.warning("No metadata bounds available, using default bounds")
                    ulx, uly = 0, height
                    lrx, lry = width, 0

                pixel_width = (lrx - ulx) / width
                pixel_height = (uly - lry) / height

                center_lon = (ulx + lrx) / 2
                center_lat = (uly + lry) / 2
                utm_zone = int((center_lon + 180) / 6) + 1
                hemisphere = 'N' if center_lat >= 0 else 'S'

                georef_info = {
                    'transform': [ulx, pixel_width, 0, uly, 0, -pixel_height],
                    'bounds': {
                        'west': ulx,
                        'east': lrx,
                        'north': uly,
                        'south': lry
                    },
                    'dimensions': {
                        'width': width,
                        'height': height
                    },
                    'utm_zone': utm_zone,
                    'hemisphere': hemisphere,
                    'solar_azimuth': self.metadata.solar_azimuth if self.metadata else 0.0,
                    'solar_elevation': self.metadata.solar_elevation if self.metadata else 0.0,
                    'cloud_cover': self.metadata.cloud_cover if self.metadata else 0.0,
                    'acquisition_date': self.metadata.acquisition_date if self.metadata else "unknown"
                }

                epsg = 32600 + utm_zone
                if hemisphere == 'S':
                    epsg = 32700 + utm_zone
                logger.info(f"Setting CRS with EPSG: {epsg} (UTM Zone: {utm_zone}{hemisphere})")
                try:
                    georef_info['crs'] = CRS.from_epsg(epsg)
                except Exception as e:
                    logger.error(f"Failed to create CRS with EPSG {epsg}: {str(e)}")
                    logger.info("Falling back to WGS84 (EPSG:4326)")
                    georef_info['crs'] = CRS.from_epsg(4326)

                logger.info(f"Extracted fallback georeference info from HDF")
                logger.info(f"Bounds: {georef_info['bounds']}")
                logger.info(f"Transform: {georef_info['transform']}")
                logger.info(f"UTM Zone: {utm_zone}{hemisphere}")

                return georef_info

        except Exception as e:
            logger.error(f"Error extracting from HDF: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            raise

    def load_data(self):
        """Load reflectance data from both VNIR and SWIR files"""
        try:
            with Dataset(self.vnir_file, 'r') as vnir:
                available_vars = list(vnir.variables.keys())
                logger.info(f"Available VNIR variables: {available_vars}")

                vnir_prefix = ""
                for var in available_vars:
                    if "Band1" in var:
                        if ":" in var:
                            vnir_prefix = var.split(":")[0] + ":"
                        break

                logger.info(f"Detected VNIR prefix: '{vnir_prefix}'")

                for band, band_name in self.vnir_datasets.items():
                    dataset_name = f"{vnir_prefix}{band_name}"
                    alt_dataset_name = band_name

                    if dataset_name in available_vars:
                        use_name = dataset_name
                    elif alt_dataset_name in available_vars:
                        use_name = alt_dataset_name
                    else:
                        logger.warning(f"Dataset {band_name} not found in VNIR file")
                        continue

                    try:
                        data = vnir.variables[use_name][:]
                        if data is None:
                            logger.warning(f"Loaded null data for VNIR band {band}")
                            continue

                        data = data.astype(np.float32)
                        data = data * 0.001
                        data[data < 0] = 0
                        data[data > 1] = 1

                        self.raw_reflectance_data[band] = data
                        logger.info(f"Loaded VNIR band {band} with shape {data.shape}")
                    except Exception as e:
                        logger.error(f"Error loading VNIR band {band}: {str(e)}")

            if hasattr(self, 'swir_file') and self.swir_file and Path(self.swir_file).exists():
                with Dataset(self.swir_file, 'r') as swir:
                    available_vars = list(swir.variables.keys())
                    logger.info(f"Available SWIR variables: {available_vars}")

                    swir_prefix = ""
                    for var in available_vars:
                        if "Band4" in var or any(f"Band{i}" in var for i in range(5, 10)):
                            if ":" in var:
                                swir_prefix = var.split(":")[0] + ":"
                            break

                    logger.info(f"Detected SWIR prefix: '{swir_prefix}'")

                    for band, band_name in self.swir_datasets.items():
                        dataset_name = f"{swir_prefix}{band_name}"
                        alt_dataset_name = band_name

                        if dataset_name in available_vars:
                            use_name = dataset_name
                        elif alt_dataset_name in available_vars:
                            use_name = alt_dataset_name
                        else:
                            logger.warning(f"Dataset {band_name} not found in SWIR file")
                            continue

                        try:
                            data = swir.variables[use_name][:]
                            if data is None:
                                logger.warning(f"Loaded null data for SWIR band {band}")
                                continue

                            data = data.astype(np.float32)
                            data = data * 0.001
                            data[data < 0] = 0
                            data[data > 1] = 1

                            self.raw_reflectance_data[band] = data
                            logger.info(f"Loaded SWIR band {band} with shape {data.shape}")
                        except Exception as e:
                            logger.error(f"Error loading SWIR band {band}: {str(e)}")
            else:
                logger.warning("No SWIR file provided or file doesn't exist. Running in VNIR-only mode.")

            if not self.raw_reflectance_data:
                raise RuntimeError("No reflectance data was successfully loaded")

            logger.info(f"Successfully loaded {len(self.raw_reflectance_data)} bands")

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            raise RuntimeError(f"Error loading data: {str(e)}")

    def validate_data(self, mineral):
        """
        Validate that the required bands for a mineral or pathfinder index are available.

        Parameters:
        -----------
        mineral : MineralIndices or GoldPathfinderIndices
            The mineral or pathfinder index to validate

        Returns:
        --------
        bool
            True if data is valid, False otherwise
        """
        try:
            required_bands = []
            
            if isinstance(mineral, MineralIndices):
                if mineral == MineralIndices.ALUNITE:
                    required_bands = [4, 5, 6]
                elif mineral == MineralIndices.KAOLINITE:
                    required_bands = [4, 5, 6]
                elif mineral == MineralIndices.CALCITE:
                    required_bands = [6, 8]
                elif mineral == MineralIndices.CHLORITE:
                    required_bands = [7, 8]
                elif mineral == MineralIndices.IRON_OXIDE:
                    required_bands = [1, 2]
                elif mineral == MineralIndices.DOLOMITE:
                    required_bands = [6, 8]  # Carbonate-related
                elif mineral == MineralIndices.EPIDOTE:
                    required_bands = [7, 8]  # MgOH-related
                elif mineral == MineralIndices.MUSCOVITE:
                    required_bands = [5, 7]  # Sericite-related
                elif mineral == MineralIndices.ILLITE:
                    required_bands = [5, 6]  # Clay-related
                elif mineral == MineralIndices.FERROUS_IRON:
                    required_bands = [1, 2]  # Iron-related
                elif mineral == MineralIndices.FERRIC_OXIDE:
                    required_bands = [1, 2]  # Iron oxide
                elif mineral == MineralIndices.HYDROXYL:
                    required_bands = [4, 6]  # AlOH-related
                elif mineral == MineralIndices.PEGMATITE:
                    required_bands = [4, 8]  # Quartz-related
                elif mineral == MineralIndices.SPODUMENE:
                    required_bands = [4, 8]  # Lithium-related
                elif mineral == MineralIndices.LABRADORITE:
                    required_bands = [4, 8]  # Feldspar-related
                elif mineral == MineralIndices.FELDSPAR:
                    required_bands = [4, 8]  # Feldspar
                elif mineral == MineralIndices.PYRITE:
                    required_bands = [2, 1]  # Sulfide-related
                elif mineral == MineralIndices.CHALCOPYRITE:
                    required_bands = [2, 1]  # Sulfide-related
                elif mineral == MineralIndices.SPHALERITE:
                    required_bands = [2, 1]  # Sulfide-related
                elif mineral == MineralIndices.GALENA:
                    required_bands = [2, 1]  # Sulfide-related
                elif mineral == MineralIndices.SERICITE:
                    required_bands = [5, 7]  # Sericite
                elif mineral == MineralIndices.CARBONATE:
                    required_bands = [6, 8]  # Carbonate
                else:
                    logger.warning(f"Unsupported mineral type: {mineral}")
                    return False

            elif isinstance(mineral, GoldPathfinderIndices):
                if mineral == GoldPathfinderIndices.GOLD_ALTERATION:
                    required_bands = [4, 6]
                elif mineral == GoldPathfinderIndices.QUARTZ_ADULARIA:
                    required_bands = [4, 8]
                elif mineral == GoldPathfinderIndices.PYRITE:
                    required_bands = [2, 1]
                elif mineral == GoldPathfinderIndices.ARSENOPYRITE:
                    required_bands = [2, 1]
                elif mineral == GoldPathfinderIndices.SILICA:
                    required_bands = [4, 8]
                elif mineral == GoldPathfinderIndices.PROPYLITIC_GOLD:
                    required_bands = [7, 8]
                elif mineral == GoldPathfinderIndices.ARGILLIC_GOLD:
                    required_bands = [4, 5]
                elif mineral == GoldPathfinderIndices.ADVANCED_ARGILLIC_GOLD:
                    required_bands = [4, 6]
                else:
                    logger.warning(f"Unsupported pathfinder type: {mineral}")
                    return False
            else:
                logger.error(f"Unsupported index type: {type(mineral)}")
                return False

            for band in required_bands:
                band_data = self.get_band_data(band)
                if band_data is None:
                    logger.warning(f"Band {band} not found for {mineral}")
                    return False
                valid_mask = ~np.isnan(band_data)
                valid_pixel_count = np.sum(valid_mask)
                if valid_pixel_count < (band_data.size * 0.05):
                    logger.warning(f"Insufficient valid data for band {band} in {mineral}")
                    return False

            return True

        except Exception as e:
            logger.error(f"Error validating data for {mineral}: {str(e)}")
            return False

    def resample_data(self):
        """Resample all bands to match SWIR resolution"""
        try:
            target_shape = None
            target_band = None

            # Determine target shape - prefer using VNIR resolution as target
            vnir_bands = [b for b in self.raw_reflectance_data.keys() if b <= 3]
            swir_bands = [b for b in self.raw_reflectance_data.keys() if b >= 4]
            
            if vnir_bands:
                target_band = max(vnir_bands)
                target_shape = self.raw_reflectance_data[target_band].shape
                logger.info(f"Using VNIR band {target_band} as resampling target with shape: {target_shape}")
            elif swir_bands:
                target_band = min(swir_bands)
                target_shape = self.raw_reflectance_data[target_band].shape
                logger.info(f"Using SWIR band {target_band} as resampling target with shape: {target_shape}")
            else:
                raise ValueError("No bands available for resampling target")

            # Check the size of SWIR bands relative to VNIR
            has_swir = len(swir_bands) > 0
            has_vnir = len(vnir_bands) > 0
            
            # Detect if SWIR is significantly smaller than VNIR (common with ASTER)
            if has_swir and has_vnir:
                vnir_pixels = self.raw_reflectance_data[vnir_bands[0]].size
                swir_pixels = self.raw_reflectance_data[swir_bands[0]].size
                ratio = vnir_pixels / swir_pixels
                
                # If VNIR has significantly more pixels, use VNIR as target
                if ratio > 1.5:  # Typically VNIR has 4x the pixels of SWIR
                    logger.info(f"VNIR resolution is {ratio:.1f}x higher than SWIR. Using VNIR as target resolution.")
                    target_band = max(vnir_bands)
                    target_shape = self.raw_reflectance_data[target_band].shape
                    
            logger.info(f"Final resampling target: Band {target_band} with shape {target_shape}")

            # Process each band
            for band, data in self.raw_reflectance_data.items():
                try:
                    # Verify data is valid before resampling
                    if np.all(np.isnan(data)) or np.count_nonzero(~np.isnan(data)) < 100:
                        logger.warning(f"Band {band} contains mostly invalid data. Skipping resampling.")
                        continue
                        
                    if data.shape != target_shape:
                        zoom_y = target_shape[0] / data.shape[0]
                        zoom_x = target_shape[1] / data.shape[1]

                        logger.info(f"Resampling band {band} from {data.shape} to {target_shape} (zoom: {zoom_y:.2f}x, {zoom_x:.2f}x)")
                        
                        # Enhanced error handling for resampling
                        try:
                            resampled = zoom(data, (zoom_y, zoom_x), order=1)
                            
                            # Verify resampled data has expected shape
                            if resampled.shape != target_shape:
                                logger.error(f"Resampling failed for band {band}: wrong output shape {resampled.shape}")
                                # Try to fix shape issues by padding or trimming
                                if resampled.shape[0] == target_shape[0] and abs(resampled.shape[1] - target_shape[1]) < 5:
                                    # Small width difference - pad or trim
                                    if resampled.shape[1] < target_shape[1]:
                                        # Pad
                                        pad_width = target_shape[1] - resampled.shape[1]
                                        resampled = np.pad(resampled, ((0, 0), (0, pad_width)), 'constant', constant_values=np.nan)
                                    else:
                                        # Trim
                                        resampled = resampled[:, :target_shape[1]]
                                elif resampled.shape[1] == target_shape[1] and abs(resampled.shape[0] - target_shape[0]) < 5:
                                    # Small height difference - pad or trim
                                    if resampled.shape[0] < target_shape[0]:
                                        # Pad
                                        pad_height = target_shape[0] - resampled.shape[0]
                                        resampled = np.pad(resampled, ((0, pad_height), (0, 0)), 'constant', constant_values=np.nan)
                                    else:
                                        # Trim
                                        resampled = resampled[:target_shape[0], :]
                                else:
                                    logger.error(f"Cannot fix resampling shape mismatch for band {band}")
                                    continue
                                    
                                logger.info(f"Fixed resampling shape for band {band} to {resampled.shape}")
                            
                            # Verify the data after resampling
                            if np.all(np.isnan(resampled)) or np.count_nonzero(~np.isnan(resampled)) < 100:
                                logger.warning(f"Resampled band {band} contains mostly invalid data. Filling with zeros.")
                                # Initialize with zeros instead of NaNs
                                resampled = np.zeros(target_shape, dtype=np.float32)
                            
                            self.reflectance_data[band] = resampled
                        except Exception as e:
                            logger.error(f"Exception during resampling for band {band}: {str(e)}")
                            # Create an empty array of the right shape instead of failing
                            logger.info(f"Creating empty placeholder for band {band}")
                            self.reflectance_data[band] = np.zeros(target_shape, dtype=np.float32)
                    else:
                        # No resampling needed, still check for invalid data
                        if np.all(np.isnan(data)) or np.count_nonzero(~np.isnan(data)) < 100:
                            logger.warning(f"Band {band} contains mostly invalid data. Filling with zeros.")
                            self.reflectance_data[band] = np.zeros(target_shape, dtype=np.float32)
                        else:
                            self.reflectance_data[band] = data.copy()

                    logger.info(f"Successfully processed band {band}")
                except Exception as e:
                    logger.error(f"Error processing band {band}: {str(e)}")
                    logger.error(f"Stack trace: {traceback.format_exc()}")
                    # Create an empty array of the right shape instead of failing
                    try:
                        self.reflectance_data[band] = np.zeros(target_shape, dtype=np.float32)
                        logger.info(f"Created empty placeholder for band {band}")
                    except:
                        logger.error(f"Could not create placeholder for band {band}")

            if not self.reflectance_data:
                raise RuntimeError("No bands were successfully resampled")

            logger.info(f"Resampling complete. Available bands: {list(self.reflectance_data.keys())}")
            
            # Additional validation step
            for band in self.reflectance_data:
                data = self.reflectance_data[band]
                # Check for invalid data
                invalid_ratio = np.sum(np.isnan(data)) / data.size
                if invalid_ratio > 0.9:  # More than 90% invalid
                    logger.warning(f"Band {band} has {invalid_ratio*100:.1f}% invalid values after resampling")
                # Check for zero data
                zero_ratio = np.sum(data == 0) / data.size
                if zero_ratio > 0.9:  # More than 90% zeros
                    logger.warning(f"Band {band} has {zero_ratio*100:.1f}% zero values after resampling")

        except Exception as e:
            logger.error(f"Error during resampling: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            raise RuntimeError(f"Resampling failed: {str(e)}")

    def apply_cloud_mask(self):
        """Apply cloud mask to the data using QA data if available"""
        try:
            if 'QA_DataPlane' in self.qa_data:
                cloud_mask = self.qa_data['QA_DataPlane'] > 0
                for band in self.vnir_datasets.keys():
                    if band in self.raw_reflectance_data:
                        self.raw_reflectance_data[band][cloud_mask] = np.nan
                logger.info("Applied cloud mask to VNIR data")

            if 'QA_DataPlane_SWIR' in self.qa_data:
                cloud_mask_swir = self.qa_data['QA_DataPlane_SWIR'] > 0
                for band in self.swir_datasets.keys():
                    if band in self.raw_reflectance_data:
                        self.raw_reflectance_data[band][cloud_mask_swir] = np.nan
                logger.info("Applied cloud mask to SWIR data")

        except Exception as e:
            logger.error(f"Error applying cloud mask: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            pass


def calculate_mineral_index(self, mineral: MineralIndices):
    """
    Calculate the mineral index and confidence map for a given mineral type.

    Parameters:
    -----------
    mineral : MineralIndices
        The mineral index to calculate

    Returns:
    --------
    tuple
        (index_map, confidence_map) - The calculated index map and confidence map
    """
    try:
        index_map = None
        confidence_map = None

        # Check if the mineral is defined in mineral_indices
        if mineral not in self.mineral_indices:
            logger.warning(f"Unsupported mineral type: {mineral}")
            return None, None

        mineral_info = self.mineral_indices[mineral]
        ratios = mineral_info['ratios']
        threshold = mineral_info['threshold']

        # Compute the index by averaging multiple band ratios
        ratio_maps = []
        valid_ratio_masks = []
        for band_num, band_den in ratios:
            num_data = self.get_band_data(band_num)
            den_data = self.get_band_data(band_den)
            if num_data is None or den_data is None:
                logger.warning(f"Required bands ({band_num}, {band_den}) not available for {mineral}")
                return None, None

            logger.info(f"Band {band_num} stats for {mineral} - Min: {np.nanmin(num_data):.3f}, Max: {np.nanmax(num_data):.3f}, Mean: {np.nanmean(num_data):.3f}, NaN count: {np.sum(np.isnan(num_data))}")
            logger.info(f"Band {band_den} stats for {mineral} - Min: {np.nanmin(den_data):.3f}, Max: {np.nanmax(den_data):.3f}, Mean: {np.nanmean(den_data):.3f}, NaN count: {np.sum(np.isnan(den_data))}")

            # Mask zeros and NaN values in the denominator
            valid_mask = (den_data != 0) & ~np.isnan(den_data) & ~np.isnan(num_data)
            if not np.any(valid_mask):
                logger.warning(f"No valid pixels for band ratio ({band_num}/{band_den}) for {mineral} after masking")
                return None, None

            ratio_map = np.full_like(num_data, np.nan, dtype=np.float32)
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio_map[valid_mask] = num_data[valid_mask] / den_data[valid_mask]

            # Exclude inf values
            valid_ratio_mask = ~np.isnan(ratio_map) & ~np.isinf(ratio_map)
            if not np.any(valid_ratio_mask):
                logger.warning(f"No valid pixels for band ratio ({band_num}/{band_den}) after excluding inf")
                return None, None

            ratio_maps.append(ratio_map)
            valid_ratio_masks.append(valid_ratio_mask)

        # Combine the ratio maps by averaging
        combined_valid_mask = np.ones_like(ratio_maps[0], dtype=bool)
        for valid_ratio_mask in valid_ratio_masks:
            combined_valid_mask &= valid_ratio_mask

        if not np.any(combined_valid_mask):
            logger.warning(f"No valid pixels for {mineral} after combining band ratios")
            return None, None

        # Average the ratio maps
        index_map = np.zeros_like(ratio_maps[0], dtype=np.float32)
        count_valid = np.zeros_like(index_map, dtype=np.float32)
        for ratio_map in ratio_maps:
            valid_pixels = ~np.isnan(ratio_map) & ~np.isinf(ratio_map)
            index_map[valid_pixels] += ratio_map[valid_pixels]
            count_valid[valid_pixels] += 1

        # Avoid division by zero
        valid_count = count_valid > 0
        index_map[valid_count] = index_map[valid_count] / count_valid[valid_count]
        index_map[~valid_count] = np.nan

        # Apply threshold to create a continuous index map
        index_map[~combined_valid_mask] = np.nan
        confidence_map = np.ones_like(index_map) * mineral_info.get('confidence', 0.9)

        return index_map, confidence_map

    except Exception as e:
        logger.error(f"Error calculating mineral index for {mineral}: {str(e)}")
        return None, None




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



    def save_mineral_map(self, mineral, output_dir: Path):
        """
        Generate and save mineral or pathfinder distribution maps as GeoTIFFs.

        Parameters:
        -----------
        mineral : MineralIndices or GoldPathfinderIndices
            The mineral or pathfinder index to calculate
        output_dir : Path
            Directory to save the GeoTIFF
        """
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"Processing mineral: {mineral.value}")

            if not self.validate_data(mineral):
                logger.warning(f"Cannot create {mineral.value} map - data validation failed")
                return

            # Calculate index map based on whether it's a mineral or pathfinder
            if isinstance(mineral, MineralIndices):
                index_map, confidence_map = self.calculate_mineral_index(mineral)
            elif isinstance(mineral, GoldPathfinderIndices):
                index_map, confidence_map = self.calculate_pathfinder_index(mineral)
            else:
                logger.error(f"Unsupported index type: {type(mineral)}")
                return

            if index_map is None or confidence_map is None:
                logger.warning(f"Skipping {mineral.value} map - index calculation failed")
                return

            # Debug: Log raw index_map and confidence_map statistics
            logger.info(f"Raw index_map stats for {mineral.value} - Min: {np.nanmin(index_map):.3f}, "
                        f"Max: {np.nanmax(index_map):.3f}, Mean: {np.nanmean(index_map):.3f}, "
                        f"NaN count: {np.sum(np.isnan(index_map))}")
            logger.info(f"Raw confidence_map stats for {mineral.value} - Min: {np.nanmin(confidence_map):.3f}, "
                        f"Max: {np.nanmax(confidence_map):.3f}, Mean: {np.nanmean(confidence_map):.3f}, "
                        f"NaN count: {np.sum(np.isnan(confidence_map))}")

            # Check for valid data (exclude NaN and inf)
            valid_mask = ~np.isnan(index_map) & ~np.isinf(index_map) & (index_map != 0)
            if not np.any(valid_mask):
                logger.error(f"Index map for {mineral.value} contains no valid data after masking NaN and inf. Skipping.")
                return

            # Normalize index_map to [0, 1] range using percentile-based stretching
            p5, p95 = np.nanpercentile(index_map[valid_mask], [5, 95])
            if p95 > p5:  # Avoid division by zero
                index_map = (index_map - p5) / (p95 - p5)
                index_map = np.clip(index_map, 0, 1)
            else:
                logger.warning(f"Index map for {mineral.value} has no range (p5 == p95), setting to 0")
                index_map = np.zeros_like(index_map)

            # Clip confidence_map to [0, 1]
            confidence_map = np.clip(confidence_map, 0, 1)

            # Debug: Log after normalization
            logger.info(f"Normalized index_map stats for {mineral.value} - Min: {np.nanmin(index_map):.3f}, "
                        f"Max: {np.nanmax(index_map):.3f}, Mean: {np.nanmean(index_map):.3f}")

            # Save a raw data visualization for debugging
            plt.figure(figsize=(8, 8))
            plt.imshow(index_map, cmap='viridis')
            plt.colorbar(label='Raw Index Value')
            plt.title(f'{mineral.value} Raw Data')
            plt.savefig(output_dir / f"{mineral.value}_raw_data.png", dpi=300)
            plt.close()

            nodata = -9999
            index_map = np.nan_to_num(index_map, nan=nodata)
            confidence_map = np.nan_to_num(confidence_map, nan=nodata)

            if self.metadata and hasattr(self.metadata, 'bounds'):
                bounds = self.metadata.bounds
                transform = from_bounds(
                    bounds.west, bounds.south,
                    bounds.east, bounds.north,
                    index_map.shape[1],
                    index_map.shape[0]
                )
                crs = CRS.from_epsg(4326)
            else:
                logger.warning(f"No metadata bounds available for {mineral.value}. Using fallback.")
                transform = from_bounds(
                    0, 0, index_map.shape[1], index_map.shape[0],
                    index_map.shape[1], index_map.shape[0]
                )
                crs = CRS.from_epsg(4326)

            logger.info(f"Applied transform for {mineral.value}: {transform}")
            if hasattr(self.metadata, 'bounds'):
                bounds = self.metadata.bounds
                logger.info(f"Bounds: W={bounds.west}, E={bounds.east}, S={bounds.south}, N={bounds.north}")
            logger.info(f"Shape: {index_map.shape}, CRS: {crs}")

            output_file = output_dir / f"{mineral.value}_map.tif"
            with rasterio.open(
                output_file,
                'w',
                driver='GTiff',
                height=index_map.shape[0],
                width=index_map.shape[1],
                count=2,
                dtype=rasterio.float32,
                crs=crs,
                transform=transform,
                nodata=nodata,
                compress='LZW'
            ) as dst:
                dst.write(index_map.astype(rasterio.float32), 1)
                dst.write(confidence_map.astype(rasterio.float32), 2)
                dst.set_band_description(1, f"{mineral.value} Index")
                dst.set_band_description(2, f"{mineral.value} Confidence")

                if self.metadata:
                    try:
                        dst.update_tags(
                            acquisition_date=self.metadata.acquisition_date,
                            solar_azimuth=str(self.metadata.solar_azimuth),
                            solar_elevation=str(self.metadata.solar_elevation),
                            cloud_cover=str(self.metadata.cloud_cover),
                            mineral_type=mineral.value,
                            creation_date=datetime.now().isoformat(),
                            data_range=f"min={np.nanmin(index_map):.3f}, max={np.nanmax(index_map):.3f}"
                        )
                    except Exception as tag_e:
                        logger.error(f"Error adding tags to {mineral.value} map: {str(tag_e)}")

            try:
                with rasterio.open(output_file, 'r') as src:
                    saved_transform = src.transform
                    saved_crs = src.crs
                    saved_bounds = src.bounds
                    logger.info(f"Verified saved GeoTIFF for {mineral.value}:")
                    logger.info(f"Transform: {saved_transform}")
                    logger.info(f"CRS: {saved_crs}")
                    logger.info(f"Bounds: {saved_bounds}")
            except Exception as verify_e:
                logger.error(f"Error verifying saved {mineral.value} map: {str(verify_e)}")

            logger.info(f"Saved and verified georeferenced GeoTIFF for {mineral.value} to {output_file}")
            
            try:
                plt.figure(figsize=(12, 6))
                
                plt.subplot(121)
                plt.imshow(np.nan_to_num(index_map, nan=0), cmap='viridis')
                plt.colorbar(label='Index Value')
                plt.title(f'{mineral.value.title()} Distribution')
                
                plt.subplot(122)
                plt.imshow(np.nan_to_num(confidence_map, nan=0), cmap='RdYlGn')
                plt.colorbar(label='Confidence')
                plt.title('Confidence Map')
                
                plt.tight_layout()
                plt.savefig(output_dir / f"{mineral.value}_visualization.png", dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"Saved visualization for {mineral.value}")
            except Exception as viz_e:
                logger.error(f"Error creating visualization for {mineral.value}: {str(viz_e)}")

        except Exception as e:
            logger.error(f"Error saving {mineral.value} map: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            raise



    def calculate_pathfinder_index(self, pathfinder: GoldPathfinderIndices):
        """
        Calculate the pathfinder index and confidence map for a given gold pathfinder type.

        Parameters:
        -----------
        pathfinder : GoldPathfinderIndices
            The pathfinder index to calculate

        Returns:
        --------
        tuple
            (index_map, confidence_map) - The calculated index map and confidence map
        """
        try:
            index_map = None
            confidence_map = None

            if pathfinder == GoldPathfinderIndices.GOLD_ALTERATION:
                band4 = self.get_band_data(4)
                band6 = self.get_band_data(6)
                if band4 is None or band6 is None:
                    logger.warning(f"Required bands (4, 6) not available for {pathfinder}")
                    return None, None
                index_map = band4 / band6
                confidence_map = np.ones_like(index_map) * 0.9

            elif pathfinder == GoldPathfinderIndices.QUARTZ_ADULARIA:
                band4 = self.get_band_data(4)
                band8 = self.get_band_data(8)
                if band4 is None or band8 is None:
                    logger.warning(f"Required bands (4, 8) not available for {pathfinder}")
                    return None, None
                index_map = band4 / band8
                confidence_map = np.ones_like(index_map) * 0.85

            elif pathfinder == GoldPathfinderIndices.PYRITE:
                band2 = self.get_band_data(2)
                band1 = self.get_band_data(1)
                if band2 is None or band1 is None:
                    logger.warning(f"Required bands (2, 1) not available for {pathfinder}")
                    return None, None
                index_map = band2 / band1
                confidence_map = np.ones_like(index_map) * 0.75

            elif pathfinder == GoldPathfinderIndices.ARSENOPYRITE:
                band2 = self.get_band_data(2)
                band1 = self.get_band_data(1)
                if band2 is None or band1 is None:
                    logger.warning(f"Required bands (2, 1) not available for {pathfinder}")
                    return None, None
                index_map = band2 / band1
                confidence_map = np.ones_like(index_map) * 0.75

            elif pathfinder == GoldPathfinderIndices.SILICA:
                band4 = self.get_band_data(4)
                band8 = self.get_band_data(8)
                if band4 is None or band8 is None:
                    logger.warning(f"Required bands (4, 8) not available for {pathfinder}")
                    return None, None
                index_map = band4 / band8
                confidence_map = np.ones_like(index_map) * 0.85

            elif pathfinder == GoldPathfinderIndices.PROPYLITIC_GOLD:
                band7 = self.get_band_data(7)
                band8 = self.get_band_data(8)
                if band7 is None or band8 is None:
                    logger.warning(f"Required bands (7, 8) not available for {pathfinder}")
                    return None, None
                index_map = band7 / band8
                confidence_map = np.ones_like(index_map) * 0.8

            elif pathfinder == GoldPathfinderIndices.ARGILLIC_GOLD:
                band4 = self.get_band_data(4)
                band5 = self.get_band_data(5)
                if band4 is None or band5 is None:
                    logger.warning(f"Required bands (4, 5) not available for {pathfinder}")
                    return None, None
                index_map = band4 / band5
                confidence_map = np.ones_like(index_map) * 0.8

            elif pathfinder == GoldPathfinderIndices.ADVANCED_ARGILLIC_GOLD:
                band4 = self.get_band_data(4)
                band6 = self.get_band_data(6)
                if band4 is None or band6 is None:
                    logger.warning(f"Required bands (4, 6) not available for {pathfinder}")
                    return None, None
                index_map = band4 / band6
                confidence_map = np.ones_like(index_map) * 0.9

            else:
                logger.warning(f"Unsupported pathfinder type: {pathfinder}")
                return None, None

            if index_map is None or confidence_map is None:
                logger.warning(f"Failed to calculate index for {pathfinder}")
                return None, None

            return index_map, confidence_map

        except Exception as e:
            logger.error(f"Error calculating pathfinder index for {pathfinder}: {str(e)}")
            return None, None



    def get_band_data(self, band_number: int) -> Optional[np.ndarray]:
        """
        Retrieve data for a specific band

        Parameters:
        -----------
        band_number : int
            The band number (1-9)

        Returns:
        --------
        Optional[np.ndarray]
            The band data as a NumPy array, or None if not available
        """
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

    def get_raster_profile(self) -> Dict:
        """
        Get the raster profile for GeoTIFF output

        Returns:
        --------
        Dict
            Rasterio profile dictionary
        """
        try:
            if not self.reflectance_data:
                logger.error("No reflectance data available for profile")
                raise ValueError("No reflectance data available")

            band = next(iter(self.reflectance_data))
            height, width = self.reflectance_data[band].shape

            if self.metadata and hasattr(self.metadata, 'bounds'):
                bounds = self.metadata.bounds
                transform = from_bounds(
                    bounds.west, bounds.south,
                    bounds.east, bounds.north,
                    width, height
                )
                crs = CRS.from_epsg(4326)
            else:
                logger.warning("No metadata bounds available, using default profile")
                transform = from_bounds(0, 0, width, height, width, height)
                crs = CRS.from_epsg(4326)

            profile = {
                'driver': 'GTiff',
                'height': height,
                'width': width,
                'count': 1,
                'dtype': rasterio.float32,
                'crs': crs,
                'transform': transform,
                'nodata': -9999,
                'compress': 'LZW'
            }

            logger.info(f"Generated raster profile: height={height}, width={width}, crs={crs}")
            return profile

        except Exception as e:
            logger.error(f"Error generating raster profile: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            raise