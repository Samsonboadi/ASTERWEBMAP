// src/services/layerValidation.js
/**
 * Layer validation service to check if layers exist before attempting to load them
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000/api';

// Cache for validated layers to avoid repeated checks
const layerValidationCache = new Map();

/**
 * Check if a layer exists on the server
 * @param {string} sceneId - Scene ID
 * @param {string} layerType - Type of layer (mineral, alteration, etc.)
 * @param {string} layerName - Name of the layer
 * @returns {Promise<boolean>} - True if layer exists, false otherwise
 */
export async function validateLayerExists(sceneId, layerType, layerName) {
  const cacheKey = `${sceneId}/${layerType}/${layerName}`;
  
  // Check cache first
  if (layerValidationCache.has(cacheKey)) {
    return layerValidationCache.get(cacheKey);
  }
  
  try {
    const url = `${API_BASE_URL}/scenes/${sceneId}/layers/${layerType}/${layerName}`;
    
    // Use HEAD request to check if resource exists without downloading it
    const response = await fetch(url, { 
      method: 'HEAD',
      timeout: 5000 // 5 second timeout
    });
    
    const exists = response.ok;
    
    // Cache the result for 5 minutes
    layerValidationCache.set(cacheKey, exists);
    setTimeout(() => {
      layerValidationCache.delete(cacheKey);
    }, 5 * 60 * 1000);
    
    return exists;
  } catch (error) {
    console.warn(`Layer validation failed for ${cacheKey}:`, error);
    // Cache negative result for shorter time (1 minute)
    layerValidationCache.set(cacheKey, false);
    setTimeout(() => {
      layerValidationCache.delete(cacheKey);
    }, 60 * 1000);
    
    return false;
  }
}

/**
 * Validate multiple layers at once
 * @param {string} sceneId - Scene ID
 * @param {Object} selectedLayers - Object with layer types and names
 * @returns {Promise<Object>} - Object with validation results
 */
export async function validateSelectedLayers(sceneId, selectedLayers) {
  const validationPromises = [];
  const layerKeys = [];
  
  // Create validation promises for all selected layers
  Object.entries(selectedLayers).forEach(([layerType, layerName]) => {
    if (layerName && layerName.trim() !== '') {
      layerKeys.push(`${layerType}/${layerName}`);
      validationPromises.push(validateLayerExists(sceneId, layerType, layerName));
    }
  });
  
  try {
    const results = await Promise.all(validationPromises);
    
    const validationResults = {};
    layerKeys.forEach((key, index) => {
      validationResults[key] = results[index];
    });
    
    return validationResults;
  } catch (error) {
    console.error('Batch layer validation failed:', error);
    return {};
  }
}

/**
 * Clear the validation cache (useful when scene is reprocessed)
 * @param {string} sceneId - Optional scene ID to clear cache for specific scene
 */
export function clearValidationCache(sceneId = null) {
  if (sceneId) {
    // Clear cache for specific scene
    const keysToDelete = [];
    for (const key of layerValidationCache.keys()) {
      if (key.startsWith(`${sceneId}/`)) {
        keysToDelete.push(key);
      }
    }
    keysToDelete.forEach(key => layerValidationCache.delete(key));
  } else {
    // Clear entire cache
    layerValidationCache.clear();
  }
}

/**
 * Get cache statistics for debugging
 * @returns {Object} - Cache statistics
 */
export function getValidationCacheStats() {
  return {
    size: layerValidationCache.size,
    keys: Array.from(layerValidationCache.keys()),
    entries: Object.fromEntries(layerValidationCache)
  };
}