# ASTER Web Explorer - Backend

The backend component of the ASTER Web Explorer application, providing a Flask-based API for processing ASTER satellite data, generating mineral and alteration maps, and performing advanced geological analysis.

## Features

- Process raw ASTER data
- Generate mineral and alteration maps
- Detect geological features
- Perform gold prospectivity analysis
- Create comprehensive reports
- Serve data to the frontend application

## Technology Stack

- **Python**: Programming language
- **Flask**: Web framework
- **GDAL/Rasterio**: Geospatial data processing
- **NumPy/SciPy**: Scientific computing
- **Matplotlib**: Data visualization
- **Scikit-learn**: Machine learning
- **GeoPandas**: Geospatial data manipulation

## Getting Started

### Prerequisites

- Python 3.8+
- GDAL library (for geospatial operations)
- Virtual environment (recommended)

### Installation

1. Clone the repository
2. Navigate to the backend directory:
   ```bash
   cd aster-web-app/backend
   ```

3. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Server

Start the development server:

```bash
python app.py
```

The API will be available at http://localhost:5000

### Configuration

Configure the application by editing `config.py` or by setting environment variables. Key configurations include:

- `DATA_DIR`: Directory for storing raw and processed data
- `LOG_DIR`: Directory for logs
- `SERVER_URL`: URL for serving files
- `CORS_ORIGINS`: Allowed origins for CORS

## Project Structure

```
backend/
├── api/                # API routes and controllers
│   ├── __init__.py
│   ├── routes.py       # API route definitions
│   └── controllers.py  # API controller functions
├── processors/         # ASTER processing modules
│   ├── __init__.py
│   ├── aster_processor.py        # Main processor class
│   ├── aster_l2_processor.py     # L2 processor module
│   ├── aster_geological_mapper.py # Geological mapper
│   ├── aster_band_math.py        # Band math operations
│   ├── aster_advanced_analysis.py # Advanced analysis
│   ├── gold_prospectivity_mapper.py # Gold prospectivity analysis
│   └── aster_ai_integration.py   # AI integration
├── utils/              # Utility functions
│   ├── __init__.py
│   ├── file_utils.py   # File handling utilities
│   ├── geo_utils.py    # Geospatial utilities
│   └── enhanced_logging.py # Enhanced logging
├── models/             # Data models
│   ├── __init__.py
│   └── scene_model.py  # Scene data model
├── config.py           # Configuration settings
├── enums.py            # Enumeration definitions
├── app.py              # Flask application
└── requirements.txt    # Python dependencies
```

## API Endpoints

### Scene Management

- `GET /api/scenes`: Get all available scenes
- `GET /api/scenes/{scene_id}`: Get details for a specific scene
- `POST /api/upload`: Upload ASTER data

### Processing

- `POST /api/scenes/{scene_id}/process`: Start processing a scene
- `GET /api/scenes/{scene_id}/status`: Get processing status

### Analysis

- `GET /api/scenes/{scene_id}/layers`: Get available layers
- `GET /api/scenes/{scene_id}/statistics`: Get scene statistics
- `GET /api/scenes/{scene_id}/layers/{layer_type}/{layer_name}`: Get a specific layer
- `POST /api/scenes/{scene_id}/generate-prospectivity`: Generate prospectivity map
- `GET /api/scenes/{scene_id}/prospectivity-areas`: Get prospectivity areas as GeoJSON
- `POST /api/scenes/{scene_id}/report`: Generate a report

## ASTER Processing Pipeline

The processing pipeline consists of several stages:

1. **Data Extraction**: Unpack raw ASTER data
2. **Mineral Mapping**: Generate mineral distribution maps
3. **Alteration Mapping**: Identify alteration zones
4. **Geological Mapping**: Detect geological features
5. **Gold Pathfinder Mapping**: Create gold pathfinder maps
6. **Advanced Analysis**: Perform detailed geological analysis

Each stage is implemented as a separate module in the `processors` directory, with a coordinating `ASTERProcessor` class that manages the workflow.

## Development Notes

### Adding New Processors

To add a new processor:

1. Create a new module in the `processors` directory
2. Implement the processor class with appropriate methods
3. Integrate the processor in the main `ASTERProcessor` class
4. Update API controllers to expose the new functionality

### Testing

Run tests using:

```bash
python -m unittest discover tests
```

### Error Handling

Errors are logged using the enhanced logging utilities in `utils/enhanced_logging.py` and returned as structured JSON responses with appropriate HTTP status codes.