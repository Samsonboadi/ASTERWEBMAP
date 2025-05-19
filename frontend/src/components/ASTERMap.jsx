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

const ASTERMap = ({
  sceneId,
  selectedLayers = {},
  showProspectivity = false,
  prospectivityThreshold = 0.7,
  imageBounds = null,
  onMapClick,
}) => {
  const [prospectivityData, setProspectivityData] = useState(null);
  const [defaultCenter] = useState([9.1, -2.3]); // Default center (can be improved with better default logic)
  const [defaultZoom] = useState(10);
  const [isLoadingProspectivity, setIsLoadingProspectivity] = useState(false);
  
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

  // Handle layer error
  const handleLayerError = (layerType, layerName) => {
    console.error(`Failed to load layer: ${layerType}/${layerName}`);
    toast({
      title: "Layer Load Error",
      description: `Failed to load the ${layerName} ${layerType} layer.`,
      variant: "destructive"
    });
  };
  
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
            {/* Alteration Layer */}
            {selectedLayers.alteration && (
              <LayersControl.Overlay checked name={`Alteration: ${selectedLayers.alteration}`}>
                <ImageOverlay
                  bounds={imageBounds}
                  url={getLayerUrl(sceneId, 'alteration', selectedLayers.alteration)}
                  opacity={0.7}
                  eventHandlers={{
                    error: () => handleLayerError('alteration', selectedLayers.alteration)
                  }}
                />
              </LayersControl.Overlay>
            )}
            
            {/* Geological Layer */}
            {selectedLayers.geological && (
              <LayersControl.Overlay checked name={`Geological: ${selectedLayers.geological}`}>
                <ImageOverlay
                  bounds={imageBounds}
                  url={getLayerUrl(sceneId, 'geological', selectedLayers.geological)}
                  opacity={0.7}
                  eventHandlers={{
                    error: () => handleLayerError('geological', selectedLayers.geological)
                  }}
                />
              </LayersControl.Overlay>
            )}
            
            {/* Band Combination Layer */}
            {selectedLayers.band && (
              <LayersControl.Overlay checked name={`Band Combination: ${selectedLayers.band}`}>
                <ImageOverlay
                  bounds={imageBounds}
                  url={getLayerUrl(sceneId, 'band', selectedLayers.band)}
                  opacity={0.7}
                  eventHandlers={{
                    error: () => handleLayerError('band', selectedLayers.band)
                  }}
                />
              </LayersControl.Overlay>
            )}
            
            {/* Band Ratio Layer */}
            {selectedLayers.ratio && (
              <LayersControl.Overlay checked name={`Band Ratio: ${selectedLayers.ratio}`}>
                <ImageOverlay
                  bounds={imageBounds}
                  url={getLayerUrl(sceneId, 'ratio', selectedLayers.ratio)}
                  opacity={0.7}
                  eventHandlers={{
                    error: () => handleLayerError('ratio', selectedLayers.ratio)
                  }}
                />
              </LayersControl.Overlay>
            )}
            
            {/* Gold Pathfinder Layer */}
            {selectedLayers.gold && (
              <LayersControl.Overlay checked name={`Gold Pathfinder: ${selectedLayers.gold}`}>
                <ImageOverlay
                  bounds={imageBounds}
                  url={getLayerUrl(sceneId, 'gold', selectedLayers.gold)}
                  opacity={0.7}
                  eventHandlers={{
                    error: () => handleLayerError('gold', selectedLayers.gold)
                  }}
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
    </MapContainer>
  );
};

export default ASTERMap;