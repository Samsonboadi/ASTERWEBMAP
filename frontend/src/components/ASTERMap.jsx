// src/components/ASTERMap.jsx
import React, { useState, useEffect, useCallback } from 'react';
import { MapContainer, TileLayer, LayersControl, GeoJSON, ImageOverlay, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { getProspectivityAreas, getLayerUrl } from '../services/api';
import { toast } from '../components/ui/toast';

// Ensure Leaflet default icon images are properly imported
import icon from 'leaflet/dist/images/marker-icon.png';
import iconShadow from 'leaflet/dist/images/marker-shadow.png';

let DefaultIcon = L.icon({
  iconUrl: icon,
  shadowUrl: iconShadow,
  iconSize: [25, 41],
  iconAnchor: [12, 41],
});

L.Marker.prototype.options.icon = DefaultIcon;

// Component to control the map view when props change
const MapViewController = ({ center, zoom, bounds }) => {
  const map = useMap();
  
  useEffect(() => {
    if (bounds) {
      map.fitBounds(bounds);
    } else if (center && zoom) {
      map.setView(center, zoom);
    }
  }, [map, center, zoom, bounds]);
  
  return null;
};

// Enhanced ImageOverlay component with error handling
const SafeImageOverlay = ({ url, bounds, opacity, layerName, onError }) => {
  const [hasError, setHasError] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    setHasError(false);
    setIsLoading(true);
    
    // Test if the image URL is accessible
    const img = new Image();
    img.onload = () => {
      setIsLoading(false);
      setHasError(false);
    };
    img.onerror = () => {
      setIsLoading(false);
      setHasError(true);
      if (onError) {
        onError(layerName);
      }
    };
    img.src = url;
  }, [url, layerName, onError]);

  if (hasError || isLoading) {
    return null; // Don't render the overlay if there's an error or still loading
  }

  return (
    <ImageOverlay
      bounds={bounds}
      url={url}
      opacity={opacity}
    />
  );
};

const ASTERMap = ({
  sceneId,
  selectedLayers = {},
  showProspectivity = false,
  prospectivityThreshold = 0.7,
  imageBounds = null,
  onMapClick,
}) => {
  const [prospectivityData, setProspectivityData] = useState(null);
  const [defaultCenter] = useState([9.1, -2.3]); // Default center
  const [defaultZoom] = useState(10);
  const [isLoadingProspectivity, setIsLoadingProspectivity] = useState(false);
  const [layerErrors, setLayerErrors] = useState(new Set());
  
  // Fetch prospectivity data when needed
  const fetchProspectivityData = useCallback(async () => {
    if (!sceneId || !showProspectivity) return;
    
    setIsLoadingProspectivity(true);
    
    try {
      const data = await getProspectivityAreas(sceneId, prospectivityThreshold);
      setProspectivityData(data);
    } catch (error) {
      toast({
        title: "Error loading prospectivity data",
        description: error.message,
        variant: "destructive"
      });
      console.error("Failed to load prospectivity data:", error);
    } finally {
      setIsLoadingProspectivity(false);
    }
  }, [sceneId, showProspectivity, prospectivityThreshold]);
  
  useEffect(() => {
    fetchProspectivityData();
  }, [fetchProspectivityData]);
  
  // Define the handleMapClick function
  const handleMapClick = (e) => {
    if (onMapClick) {
      onMapClick(e.latlng);
    }
  };
  
  // Style function for GeoJSON features
  const getLayerStyle = (feature) => {
    const category = feature.properties?.category || "medium";
    
    switch (category) {
      case 'high':
        return {
          fillColor: '#ff0000',
          weight: 2,
          opacity: 1,
          color: '#ff0000',
          fillOpacity: 0.4
        };
      case 'medium':
        return {
          fillColor: '#ffa500',
          weight: 1,
          opacity: 1,
          color: '#ffa500',
          fillOpacity: 0.3
        };
      case 'low':
        return {
          fillColor: '#ffff00',
          weight: 1,
          opacity: 1,
          color: '#ffff00',
          fillOpacity: 0.2
        };
      default:
        return {
          fillColor: '#3388ff',
          weight: 2,
          opacity: 1,
          color: '#3388ff',
          fillOpacity: 0.2
        };
    }
  };
  
  // Popup content function for GeoJSON features
  const onEachFeature = (feature, layer) => {
    if (feature.properties) {
      const { confidence, value, category } = feature.properties;
      layer.bindPopup(`
        <div>
          <strong>${category.toUpperCase()} Prospectivity Area</strong><br/>
          Value: ${value?.toFixed(2) || 'N/A'}<br/>
          Confidence: ${confidence?.toFixed(2) || 'N/A'}<br/>
        </div>
      `);
    }
  };

  // Handle layer error - improved error tracking with rate limiting
  const handleLayerError = useCallback((layerType, layerName) => {
    const layerKey = `${layerType}/${layerName}`;
    
    if (!layerErrors.has(layerKey)) {
      setLayerErrors(prev => new Set([...prev, layerKey]));
      
      console.error(`Failed to load layer: ${layerKey}`);
      
      // Only show toast for the first error per layer and throttle messages
      const now = Date.now();
      const lastErrorTime = window._lastLayerErrorTime || 0;
      
      // Only show error toast if it's been more than 3 seconds since last error
      if (now - lastErrorTime > 3000) {
        toast({
          title: "Layer Unavailable",
          description: `The ${layerName.replace(/_/g, ' ')} ${layerType} layer is not available for this scene.`,
          variant: "warning"
        });
        window._lastLayerErrorTime = now;
      }
    }
  }, [layerErrors]);

  // Helper function to check if a layer should be rendered
  const shouldRenderLayer = useCallback((layerType, layerName) => {
    if (!sceneId || !imageBounds) return false;
    
    const layerKey = `${layerType}/${layerName}`;
    return !layerErrors.has(layerKey);
  }, [sceneId, imageBounds, layerErrors]);

  // Helper function to get layer URL with error handling
  const getSafeLayerUrl = useCallback((layerType, layerName) => {
    try {
      return getLayerUrl(sceneId, layerType, layerName);
    } catch (error) {
      console.error(`Error getting URL for layer ${layerType}/${layerName}:`, error);
      handleLayerError(layerType, layerName);
      return null;
    }
  }, [sceneId, handleLayerError]);
  
  return (
    <MapContainer 
      center={defaultCenter} 
      zoom={defaultZoom} 
      style={{ height: "100%", width: "100%" }}
      zoomControl={false}
      onClick={handleMapClick}
    >
      <MapViewController 
        center={defaultCenter} 
        zoom={defaultZoom} 
        bounds={imageBounds}
      />
      
      <LayersControl position="topright">
        {/* Base Layers */}
        <LayersControl.BaseLayer checked name="OpenStreetMap">
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
        </LayersControl.BaseLayer>
        
        <LayersControl.BaseLayer name="Satellite">
          <TileLayer
            attribution='&copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community'
            url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
          />
        </LayersControl.BaseLayer>
        
        <LayersControl.BaseLayer name="Terrain">
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            url="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png"
          />
        </LayersControl.BaseLayer>
        
        {/* Overlay Layers */}
        {imageBounds && (
          <LayersControl.Overlay checked name="Scene Boundary">
            <ImageOverlay
              bounds={imageBounds}
              url="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+P+/HgAFeAJ5gYKhJwAAAABJRU5ErkJggg=="
              opacity={0.01}
            />
          </LayersControl.Overlay>
        )}
        
        {/* Dynamic Layers Based on Selection */}
        {sceneId && imageBounds && (
          <>
            {/* Mineral Layer */}
            {selectedLayers.mineral && shouldRenderLayer('mineral', selectedLayers.mineral) && (
              <LayersControl.Overlay checked name={`Mineral: ${selectedLayers.mineral}`}>
                <SafeImageOverlay
                  bounds={imageBounds}
                  url={getSafeLayerUrl('mineral', selectedLayers.mineral)}
                  opacity={0.7}
                  layerName={selectedLayers.mineral}
                  onError={() => handleLayerError('mineral', selectedLayers.mineral)}
                />
              </LayersControl.Overlay>
            )}
            
            {/* Alteration Layer */}
            {selectedLayers.alteration && shouldRenderLayer('alteration', selectedLayers.alteration) && (
              <LayersControl.Overlay checked name={`Alteration: ${selectedLayers.alteration}`}>
                <SafeImageOverlay
                  bounds={imageBounds}
                  url={getSafeLayerUrl('alteration', selectedLayers.alteration)}
                  opacity={0.7}
                  layerName={selectedLayers.alteration}
                  onError={() => handleLayerError('alteration', selectedLayers.alteration)}
                />
              </LayersControl.Overlay>
            )}
            
            {/* Geological Layer */}
            {selectedLayers.geological && shouldRenderLayer('geological', selectedLayers.geological) && (
              <LayersControl.Overlay checked name={`Geological: ${selectedLayers.geological}`}>
                <SafeImageOverlay
                  bounds={imageBounds}
                  url={getSafeLayerUrl('geological', selectedLayers.geological)}
                  opacity={0.7}
                  layerName={selectedLayers.geological}
                  onError={() => handleLayerError('geological', selectedLayers.geological)}
                />
              </LayersControl.Overlay>
            )}
            
            {/* Band Combination Layer */}
            {selectedLayers.band && shouldRenderLayer('band', selectedLayers.band) && (
              <LayersControl.Overlay checked name={`Band Combination: ${selectedLayers.band}`}>
                <SafeImageOverlay
                  bounds={imageBounds}
                  url={getSafeLayerUrl('band', selectedLayers.band)}
                  opacity={0.7}
                  layerName={selectedLayers.band}
                  onError={() => handleLayerError('band', selectedLayers.band)}
                />
              </LayersControl.Overlay>
            )}
            
            {/* Band Ratio Layer */}
            {selectedLayers.ratio && shouldRenderLayer('ratio', selectedLayers.ratio) && (
              <LayersControl.Overlay checked name={`Band Ratio: ${selectedLayers.ratio}`}>
                <SafeImageOverlay
                  bounds={imageBounds}
                  url={getSafeLayerUrl('ratio', selectedLayers.ratio)}
                  opacity={0.7}
                  layerName={selectedLayers.ratio}
                  onError={() => handleLayerError('ratio', selectedLayers.ratio)}
                />
              </LayersControl.Overlay>
            )}
            
            {/* Gold Pathfinder Layer */}
            {selectedLayers.gold && shouldRenderLayer('gold', selectedLayers.gold) && (
              <LayersControl.Overlay checked name={`Gold Pathfinder: ${selectedLayers.gold}`}>
                <SafeImageOverlay
                  bounds={imageBounds}
                  url={getSafeLayerUrl('gold', selectedLayers.gold)}
                  opacity={0.7}
                  layerName={selectedLayers.gold}
                  onError={() => handleLayerError('gold', selectedLayers.gold)}
                />
              </LayersControl.Overlay>
            )}

            {/* VNIR Layer for VNIR-only mode */}
            {selectedLayers.vnir && shouldRenderLayer('vnir', selectedLayers.vnir) && (
              <LayersControl.Overlay checked name={`VNIR: ${selectedLayers.vnir}`}>
                <SafeImageOverlay
                  bounds={imageBounds}
                  url={getSafeLayerUrl('vnir', selectedLayers.vnir)}
                  opacity={0.7}
                  layerName={selectedLayers.vnir}
                  onError={() => handleLayerError('vnir', selectedLayers.vnir)}
                />
              </LayersControl.Overlay>
            )}
            
            {/* Prospectivity Layers */}
            {showProspectivity && prospectivityData && (
              <>
                {prospectivityData.high && (
                  <LayersControl.Overlay checked name="High Prospectivity">
                    <GeoJSON 
                      data={prospectivityData.high}
                      style={getLayerStyle}
                      onEachFeature={onEachFeature}
                    />
                  </LayersControl.Overlay>
                )}
                
                {prospectivityData.medium && (
                  <LayersControl.Overlay checked name="Medium Prospectivity">
                    <GeoJSON 
                      data={prospectivityData.medium}
                      style={getLayerStyle}
                      onEachFeature={onEachFeature}
                    />
                  </LayersControl.Overlay>
                )}
                
                {prospectivityData.low && (
                  <LayersControl.Overlay name="Low Prospectivity">
                    <GeoJSON 
                      data={prospectivityData.low}
                      style={getLayerStyle}
                      onEachFeature={onEachFeature}
                    />
                  </LayersControl.Overlay>
                )}
              </>
            )}
          </>
        )}
      </LayersControl>
      
      {/* Loading indicator for prospectivity data */}
      {showProspectivity && isLoadingProspectivity && (
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 bg-white p-3 rounded-md shadow-md z-50">
          <div className="flex items-center space-x-2">
            <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-red-600"></div>
            <span className="text-sm font-medium">Loading prospectivity data...</span>
          </div>
        </div>
      )}

      {/* Error indicator for failed layers */}
      {layerErrors.size > 0 && (
        <div className="absolute top-4 right-4 bg-yellow-50 border border-yellow-200 rounded-md p-3 z-50 max-w-sm">
          <div className="flex items-start">
            <div className="flex-shrink-0">
              <svg className="h-5 w-5 text-yellow-400" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
              </svg>
            </div>
            <div className="ml-3">
              <h3 className="text-sm font-medium text-yellow-800">
                Some layers failed to load
              </h3>
              <div className="mt-2 text-sm text-yellow-700">
                <p>{layerErrors.size} layer(s) are not available for this scene.</p>
              </div>
            </div>
          </div>
        </div>
      )}
    </MapContainer>
  );
};

export default ASTERMap;