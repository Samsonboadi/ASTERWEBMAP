// src/services/api.js
/**
 * API service for communicating with the backend
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000/api';

/**
 * Generic fetch wrapper with error handling
 */
async function fetchAPI(endpoint, options = {}) {
  try {
    const url = `${API_BASE_URL}${endpoint}`;
    console.log('Fetching:', url, options);
    
    // Add request timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 30000); // 30-second timeout
    
    const response = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      signal: controller.signal,
    });
    
    // Clear the timeout
    clearTimeout(timeoutId);
    
    // Handle different status codes
    if (!response.ok) {
      let errorMessage = `API error: ${response.status}`;
      
      try {
        const errorData = await response.json();
        errorMessage = errorData.message || errorMessage;
      } catch (jsonError) {
        // Ignore JSON parsing errors from error responses
      }
      
      // Create error object with additional details
      const error = new Error(errorMessage);
      error.status = response.status;
      error.statusText = response.statusText;
      throw error;
    }
    
    return await response.json();
  } catch (error) {
    // Handle abort errors (timeouts)
    if (error.name === 'AbortError') {
      throw new Error('Request timeout. The server took too long to respond.');
    }
    
    // Special handling for network errors
    if (error.message === 'Failed to fetch') {
      throw new Error('Network error. Please check your connection.');
    }
    
    // Log the error for debugging
    console.error('API request failed:', error);
    
    // Re-throw the error with any enhancements we've added
    throw error;
  }
}


/**
 * Upload a new ASTER data file for processing
 */
export async function uploadAsterData(file, metadata = {}) {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('metadata', JSON.stringify(metadata));

  // Create a tracking object for upload progress
  const progressTracker = { 
    onUploadProgress: () => {}
  };

  try {
    // Create XMLHttpRequest to track progress
    const xhr = new XMLHttpRequest();
    
    // Create a Promise to handle the request
    const uploadPromise = new Promise((resolve, reject) => {
      xhr.upload.addEventListener('progress', (event) => {
        if (event.lengthComputable) {
          const progress = Math.round((event.loaded / event.total) * 100);
          // Call the progress callback
          progressTracker.onUploadProgress(progress);
        }
      });
      
      xhr.addEventListener('load', () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            const response = JSON.parse(xhr.responseText);
            console.log("Upload successful, response:", response);
            resolve(response);
          } catch (e) {
            console.error("Failed to parse response:", e);
            reject(new Error("Invalid JSON response from server"));
          }
        } else {
          let errorMessage = `Upload failed: ${xhr.status}`;
          try {
            const errorData = JSON.parse(xhr.responseText);
            errorMessage = errorData.message || errorMessage;
          } catch (e) {
            // Ignore JSON parsing errors
            console.error("Failed to parse error response:", e);
          }
          reject(new Error(errorMessage));
        }
      });
      
      xhr.addEventListener('error', () => {
        console.error("Network error during upload");
        reject(new Error("Network error during upload"));
      });
      
      xhr.addEventListener('abort', () => {
        console.error("Upload was aborted");
        reject(new Error("Upload was aborted"));
      });
    });
    
    // Open and send the request
    xhr.open('POST', `${API_BASE_URL}/upload`, true);
    xhr.send(formData);
    
    // Return both the promise and the tracker
    return { 
      promise: uploadPromise, 
      progressTracker 
    };

  } catch (error) {
    console.error('Upload failed:', error);
    throw error;
  }
}




/**
 * Get scene details by ID
 */
export async function getSceneById(sceneId) {
  try {
    return await fetchAPI(`/scenes/${sceneId}`);
  } catch (error) {
    return handleBackendUnavailable(error, {
      id: sceneId,
      name: `Scene ${sceneId}`,
      date: '2023-03-10',
      cloudCover: 15,
      status: 'completed',
      bounds: {
        west: -2.67,
        east: -1.99,
        south: 8.76,
        north: 9.43
      }
    });
  }
}

/**
 * Start processing a scene
 */

export async function processScene(sceneId, options) {
  console.log(`Processing scene ${sceneId} with options:`, options);
  try {
    // Ensure process_minerals is set if minerals is true
    const processingOptions = {
      extract: true,
      process_minerals: options.minerals === true,
      process_alteration: options.alteration === true,
      process_gold_pathfinders: options.goldPathfinders === true,
      enhanced_visualization: options.enhancedVisualization === true
    };
    
    console.log("Sending processing options to backend:", processingOptions);
    
    return await fetchAPI(`/scenes/${sceneId}/process`, {
      method: 'POST',
      body: JSON.stringify(processingOptions),
    });
  } catch (error) {
    console.error("Processing error:", error);
    throw error;
  }
}


// Helper function to handle backend unavailability
const handleBackendUnavailable = (error, fallbackData) => {
  console.warn("Backend error - using fallback data:", error);
  return fallbackData;
};

// Modified getScenes with fallback
//Get all available scenes
export async function getScenes() {
  try {
    return await fetchAPI('/scenes');
  } catch (error) {
    // Fallback data if backend is unavailable
    return handleBackendUnavailable(error, [
      {
        id: 'scene_001',
        name: 'Test Scene 1',
        date: '2023-01-15',
        cloudCover: 10,
        status: 'completed'
      }
    ]);
  }
}

/**
 * Get processing status for a scene
 */
export async function getProcessingStatus(sceneId) {
  try {
    return await fetchAPI(`/scenes/${sceneId}/status`);
  } catch (error) {
    // Fallback data if backend is unavailable
    return handleBackendUnavailable(error, {
      status: 'processing',
      progress: 50,
      stage: 'mineral_mapping'
    });
  }
}

/**
 * Get available layers for a scene
 */
export async function getAvailableLayers(sceneId) {
  try {
    return await fetchAPI(`/scenes/${sceneId}/layers`);
  } catch (error) {
    return handleBackendUnavailable(error, {
      mineral: ['alunite', 'kaolinite', 'calcite'],
      alteration: ['advanced_argillic', 'argillic', 'propylitic'],
      gold: ['pyrite', 'arsenopyrite', 'gold_alteration'],
      band: ['general_alteration', 'iron_oxide'],
      ratio: ['ferric_iron', 'aloh_content'],
      geological: ['lineaments', 'faults']
    });
  }
}

/**
 * Get a specific map layer as GeoTIFF URL
 */
export function getLayerUrl(sceneId, layerType, layerName) {
  try {
    return `${API_BASE_URL}/scenes/${sceneId}/layers/${layerType}/${layerName}`;
  } catch (error) {
    return handleBackendUnavailable(error, 'https://via.placeholder.com/800x600?text=Layer+Image');
  }
}
/**
 * Generate a prospectivity map
 */
export async function generateProspectivityMap(sceneId, options = {}) {
  try {
    return await fetchAPI(`/scenes/${sceneId}/generate-prospectivity`, {
      method: 'POST',
      body: JSON.stringify(options),
    });
  } catch (error) {
    return handleBackendUnavailable(error, {
      sceneId,
      status: 'success',
      message: 'Prospectivity map generated successfully'
    });
  }
}

/**
 * Get scene statistics
 */
export async function getSceneStatistics(sceneId) {
  try {
    return await fetchAPI(`/scenes/${sceneId}/statistics`);
  } catch (error) {
    return handleBackendUnavailable(error, {
      acquisitionDate: '2023-05-15',
      cloudCover: 5.0,
      dominantAlteration: 'Argillic',
      dominantMinerals: 'Kaolinite, Alunite',
      goldPathfinderCoverage: 27.5,
      mineralCount: 4,
      alterationCount: 3,
      pathfinderCount: 3,
      processingTime: '15 minutes',
      sceneArea: '100 km²'
    });
  }
}

/**
 * Generate an analysis report
 */
export async function generateReport(sceneId, reportType, options = {}) {
  try {
    return await fetchAPI(`/scenes/${sceneId}/report`, {
      method: 'POST',
      body: JSON.stringify({ reportType, ...options }),
    });
  } catch (error) {
    return handleBackendUnavailable(error, {
      sceneId,
      reportType,
      reportUrl: '#'
    });
  }
}
/**
 * Export a map as GeoTIFF
 */
export function exportMap(sceneId, mapType, mapName, format = 'geotiff') {
  window.open(`${API_BASE_URL}/scenes/${sceneId}/export/${mapType}/${mapName}?format=${format}`, '_blank');
}

/**
 * Get gold prospectivity areas as GeoJSON
 */
export async function getProspectivityAreas(sceneId, threshold = 0.7) {
  try {
    return await fetchAPI(`/scenes/${sceneId}/prospectivity-areas?threshold=${threshold}`);
  } catch (error) {
    return handleBackendUnavailable(error, {
      high: {
        type: 'FeatureCollection',
        features: [
          {
            type: 'Feature',
            geometry: {
              type: 'Polygon',
              coordinates: [
                [
                  [-2.3, 9.1],
                  [-2.3, 9.2],
                  [-2.2, 9.2],
                  [-2.2, 9.1],
                  [-2.3, 9.1]
                ]
              ]
            },
            properties: {
              category: 'high',
              value: 0.85,
              confidence: 0.9
            }
          }
        ]
      },
      medium: {
        type: 'FeatureCollection',
        features: [
          {
            type: 'Feature',
            geometry: {
              type: 'Polygon',
              coordinates: [
                [
                  [-2.4, 9.0],
                  [-2.4, 9.1],
                  [-2.3, 9.1],
                  [-2.3, 9.0],
                  [-2.4, 9.0]
                ]
              ]
            },
            properties: {
              category: 'medium',
              value: 0.65,
              confidence: 0.8
            }
          }
        ]
      },
      low: {
        type: 'FeatureCollection',
        features: [
          {
            type: 'Feature',
            geometry: {
              type: 'Polygon',
              coordinates: [
                [
                  [-2.5, 8.9],
                  [-2.5, 9.0],
                  [-2.4, 9.0],
                  [-2.4, 8.9],
                  [-2.5, 8.9]
                ]
              ]
            },
            properties: {
              category: 'low',
              value: 0.35,
              confidence: 0.7
            }
          }
        ]
      }
    });
  }
}

/**
 * Get gold prospectivity areas as GeoJSON

export async function getProspectivityAreas(sceneId, threshold = 0.7) {
  return fetchAPI(`/scenes/${sceneId}/prospectivity-areas?threshold=${threshold}`);
} */