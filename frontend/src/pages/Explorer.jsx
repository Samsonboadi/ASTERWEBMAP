// src/pages/Explorer.jsx
import React, { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import { PanelLeft, PanelRight, Map as MapIcon, Layers, Eye } from 'lucide-react';
import { Button } from '../components/ui';
import ASTERMap from '../components/ASTERMap';
import ProcessingPanel from '../components/ProcessingPanel';
import AnalysisPanel from '../components/AnalysisPanel';
import { getScenes, getSceneById, getAvailableLayers } from '../services/api';
import { toast } from '../components/ui/toast';

const Explorer = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const [scenes, setScenes] = useState([]);
  const [selectedScene, setSelectedScene] = useState(null);
  const [sceneDetails, setSceneDetails] = useState(null);
  const [availableLayers, setAvailableLayers] = useState({});
  const [selectedLayers, setSelectedLayers] = useState({
    alteration: "",
    geological: "",
    band: "",
    ratio: "",
    gold: ""
  });
  const [showProspectivity, setShowProspectivity] = useState(false);
  const [prospectivityThreshold, setProspectivityThreshold] = useState(0.7);
  const [leftSidebarExpanded, setLeftSidebarExpanded] = useState(true);
  const [rightSidebarExpanded, setRightSidebarExpanded] = useState(true);
  const [activeLeftTab, setActiveLeftTab] = useState('process'); // 'process' or 'layers'
  const [isLoading, setIsLoading] = useState(true);
  const [isVnirOnly, setIsVnirOnly] = useState(false);

  // Check for scene parameter in URL on component mount
  useEffect(() => {
    const sceneFromUrl = searchParams.get('scene');
    if (sceneFromUrl) {
      setSelectedScene(sceneFromUrl);
      // Switch to layers tab if coming from Scene Manager
      setActiveLeftTab('layers');
    }
    fetchScenes();
  }, [searchParams]);

  // Fetch scenes on component mount
  useEffect(() => {
    if (sceneDetails) {
      setIsVnirOnly(sceneDetails.processingMode === 'VNIR-only');
    }
  }, [sceneDetails]);

  // Fetch scene details when a scene is selected
  useEffect(() => {
    if (selectedScene) {
      fetchSceneDetails();
      fetchAvailableLayers();
      // Update URL parameter only if it's different
      const currentSceneParam = searchParams.get('scene');
      if (currentSceneParam !== selectedScene) {
        setSearchParams({ scene: selectedScene });
      }
    }
  }, [selectedScene, searchParams, setSearchParams]);

  const fetchScenes = async () => {
    setIsLoading(true);
    try {
      const data = await getScenes();
      setScenes(data);
      
      // If no scene is selected but we have scenes, and no URL parameter, select the first one
      if (!selectedScene && !searchParams.get('scene') && data.length > 0) {
        setSelectedScene(data[0].id);
      }
    } catch (error) {
      toast({
        title: "Error Loading Scenes",
        description: error.message || "Failed to load available scenes.",
        variant: "destructive"
      });
      console.error("Failed to load scenes:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const fetchSceneDetails = async () => {
    if (!selectedScene) return;
    
    try {
      const data = await getSceneById(selectedScene);
      setSceneDetails(data);
    } catch (error) {
      toast({
        title: "Error Loading Scene Details",
        description: error.message || "Failed to load scene details.",
        variant: "destructive"
      });
      console.error("Failed to load scene details:", error);
    }
  };

  const fetchAvailableLayers = async () => {
    if (!selectedScene) return;
    
    try {
      const data = await getAvailableLayers(selectedScene);
      setAvailableLayers(data);
    } catch (error) {
      toast({
        title: "Error Loading Available Layers",
        description: error.message || "Failed to load available layers.",
        variant: "destructive"
      });
      console.error("Failed to load available layers:", error);
    }
  };

  const toggleLeftSidebar = () => {
    setLeftSidebarExpanded(!leftSidebarExpanded);
  };

  const toggleRightSidebar = () => {
    setRightSidebarExpanded(!rightSidebarExpanded);
  };

  const handleLayerChange = (layerType, value) => {
    setSelectedLayers(prev => ({
      ...prev,
      [layerType]: value
    }));
  };

  const handleSceneChange = (sceneId) => {
    setSelectedScene(sceneId);
  };

  const handleProcessingComplete = (sceneId) => {
    setSelectedScene(sceneId);
    fetchSceneDetails();
    fetchAvailableLayers();
    
    // Show success message and suggest viewing layers
    toast({
      title: "Processing Complete",
      description: "Scene processing completed successfully. Check out the available layers!",
      variant: "success"
    });
    
    // Automatically switch to layers tab
    setActiveLeftTab('layers');
  };

  const handleProspectivityToggle = (value) => {
    setShowProspectivity(value);
  };

  const handleProspectivityThresholdChange = (value) => {
    setProspectivityThreshold(value);
  };

  // Calculate image bounds for the map
  const getImageBounds = () => {
    if (!sceneDetails || !sceneDetails.bounds) return null;
    
    const { west, east, south, north } = sceneDetails.bounds;
    return [
      [south, west], // Southwest corner [lat, lng]
      [north, east]  // Northeast corner [lat, lng]
    ];
  };

  return (
    <div className="flex h-screen bg-gray-100">
      {/* Left Sidebar */}
      <div className={`bg-white shadow-lg transition-all duration-300 z-10 ${leftSidebarExpanded ? 'w-80' : 'w-16'}`}>
        <div className="p-4 h-full flex flex-col">
          <div className="flex justify-between items-center mb-4">
            {leftSidebarExpanded && <h2 className="text-xl font-bold text-gray-800">ASTER Explorer</h2>}
            <button 
              onClick={toggleLeftSidebar}
              className="p-2 rounded-md bg-gray-200 hover:bg-gray-300"
            >
              {leftSidebarExpanded ? <PanelLeft className="w-5 h-5" /> : <PanelRight className="w-5 h-5" />}
            </button>
          </div>

          {leftSidebarExpanded && !isLoading && (
            <>
              <div className="mb-4">
                <div className="flex space-x-1 mb-4">
                  <Button
                    variant={activeLeftTab === 'process' ? 'default' : 'outline'}
                    className="flex-1"
                    onClick={() => setActiveLeftTab('process')}
                  >
                    <MapIcon className="w-4 h-4 mr-2" />
                    Process
                  </Button>
                  <Button
                    variant={activeLeftTab === 'layers' ? 'default' : 'outline'}
                    className="flex-1"
                    onClick={() => setActiveLeftTab('layers')}
                  >
                    <Layers className="w-4 h-4 mr-2" />
                    Layers
                  </Button>
                </div>
                
                {activeLeftTab === 'process' && (
                  <ProcessingPanel 
                    onProcessingComplete={handleProcessingComplete}
                    sceneId={selectedScene}
                  />
                )}
                
                {activeLeftTab === 'layers' && (
                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">Scene Selection</label>
                      <select 
                        value={selectedScene || ''}
                        onChange={(e) => handleSceneChange(e.target.value)}
                        className="w-full p-2 border border-gray-300 rounded-md"
                        disabled={isLoading}
                      >
                        <option value="">Select a scene</option>
                        {scenes.map(scene => (
                          <option key={scene.id} value={scene.id}>
                            {scene.name || scene.id}
                          </option>
                        ))}
                      </select>
                    </div>

                    {selectedScene && sceneDetails && (
                      <div className="bg-blue-50 p-3 rounded-md">
                        <h4 className="text-sm font-medium text-blue-900 mb-1">Scene Info</h4>
                        <div className="text-xs text-blue-700 space-y-1">
                          <div>Date: {sceneDetails.date !== 'Unknown' ? new Date(sceneDetails.date).toLocaleDateString() : 'Unknown'}</div>
                          <div>Cloud Cover: {sceneDetails.cloudCover || 0}%</div>
                          <div>Status: <span className="font-medium capitalize">{sceneDetails.status}</span></div>
                          {sceneDetails.processingMode === 'VNIR-only' && (
                            <div className="text-blue-600 font-medium">Mode: VNIR Only</div>
                          )}
                        </div>
                      </div>
                    )}
                    
                    {/* Layer Selection */}
                    <div className="space-y-3">
                      <h3 className="text-sm font-medium text-gray-700">Available Layers</h3>
                      
                      {/* Alteration Layer Selection */}
                      <div>
                        <label className="block text-xs text-gray-500 mb-1">
                          Alteration Indices
                          {isVnirOnly && (
                            <span className="ml-2 text-xs text-blue-500">
                              (Not available in VNIR-only mode)
                            </span>
                          )}
                        </label>
                        <select 
                          value={selectedLayers.alteration}
                          onChange={(e) => handleLayerChange('alteration', e.target.value)}
                          className="w-full p-2 border border-gray-300 rounded-md text-sm"
                          disabled={isVnirOnly}
                        >
                          <option value="">None</option>
                          {availableLayers.alteration?.map(layer => (
                            <option key={layer} value={layer}>
                              {layer.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
                            </option>
                          ))}
                        </select>
                      </div>

                      {/* Add a special NDVI layer option for VNIR-only mode */}
                      {isVnirOnly && (
                        <div>
                          <label className="block text-xs text-gray-500 mb-1">VNIR Indices</label>
                          <select 
                            value={selectedLayers.vnir || ''}
                            onChange={(e) => handleLayerChange('vnir', e.target.value)}
                            className="w-full p-2 border border-gray-300 rounded-md text-sm"
                          >
                            <option value="">None</option>
                            <option value="ndvi">NDVI (Vegetation Index)</option>
                            <option value="band1">Band 1 (Blue)</option>
                            <option value="band2">Band 2 (Red)</option>
                            <option value="band3">Band 3 (NIR)</option>
                          </select>
                        </div>
                      )}
                      
                      {/* Geological Layer Selection */}
                      <div>
                        <label className="block text-xs text-gray-500 mb-1">Geological Features</label>
                        <select 
                          value={selectedLayers.geological}
                          onChange={(e) => handleLayerChange('geological', e.target.value)}
                          className="w-full p-2 border border-gray-300 rounded-md text-sm"
                        >
                          <option value="">None</option>
                          {availableLayers.geological?.map(layer => (
                            <option key={layer} value={layer}>
                              {layer.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
                            </option>
                          ))}
                        </select>
                      </div>
                      
                      {/* Band Combination Layer Selection */}
                      <div>
                        <label className="block text-xs text-gray-500 mb-1">Band Combinations</label>
                        <select 
                          value={selectedLayers.band}
                          onChange={(e) => handleLayerChange('band', e.target.value)}
                          className="w-full p-2 border border-gray-300 rounded-md text-sm"
                        >
                          <option value="">None</option>
                          {availableLayers.band?.map(layer => (
                            <option key={layer} value={layer}>
                              {layer.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
                            </option>
                          ))}
                        </select>
                      </div>
                      
                      {/* Band Ratio Layer Selection */}
                      <div>
                        <label className="block text-xs text-gray-500 mb-1">Band Ratios</label>
                        <select 
                          value={selectedLayers.ratio}
                          onChange={(e) => handleLayerChange('ratio', e.target.value)}
                          className="w-full p-2 border border-gray-300 rounded-md text-sm"
                        >
                          <option value="">None</option>
                          {availableLayers.ratio?.map(layer => (
                            <option key={layer} value={layer}>
                              {layer.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
                            </option>
                          ))}
                        </select>
                      </div>
                      
                      {/* Gold Pathfinder Layer Selection */}
                      <div>
                        <label className="block text-xs text-gray-500 mb-1">
                          Gold Pathfinders
                          {isVnirOnly && (
                            <span className="ml-2 text-xs text-blue-500">
                              (Limited in VNIR-only mode)
                            </span>
                          )}
                        </label>
                        <select 
                          value={selectedLayers.gold}
                          onChange={(e) => handleLayerChange('gold', e.target.value)}
                          className="w-full p-2 border border-gray-300 rounded-md text-sm"
                        >
                          <option value="">None</option>
                          {availableLayers.gold?.map(layer => (
                            <option key={layer} value={layer}>
                              {layer.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
                            </option>
                          ))}
                        </select>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </>
          )}

          {isLoading && leftSidebarExpanded && (
            <div className="flex justify-center items-center p-8">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600"></div>
            </div>
          )}
        </div>
      </div>

      {/* Main content area with map */}
      <div className="flex-1 relative">
        {selectedScene ? (
          <ASTERMap 
            sceneId={selectedScene}
            selectedLayers={selectedLayers}
            showProspectivity={showProspectivity}
            prospectivityThreshold={prospectivityThreshold}
            imageBounds={getImageBounds()}
          />
        ) : (
          <div className="flex items-center justify-center h-full bg-gray-50">
            <div className="text-center">
              <MapIcon className="w-16 h-16 text-gray-400 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">No Scene Selected</h3>
              <p className="text-gray-500 mb-4">
                {isLoading ? "Loading scenes..." : "Select a scene from the sidebar to view its layers"}
              </p>
              {!isLoading && scenes.length === 0 && (
                <p className="text-sm text-gray-400">
                  Upload and process ASTER data to get started
                </p>
              )}
            </div>
          </div>
        )}
        
        {/* Map overlay controls and info panel */}
        {selectedScene && (
          <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 bg-white shadow-md rounded-md p-2 z-10">
            <div className="flex items-center space-x-2">
              <div className="text-sm font-medium flex items-center">
                <Eye className="w-4 h-4 mr-1" />
                Active Layers:
              </div>
              {Object.entries(selectedLayers).map(([type, value]) => 
                value ? (
                  <div key={type} className="flex items-center">
                    <div className="w-3 h-3 rounded-full mr-1" 
                      style={{ 
                        backgroundColor: 
                          type === 'alteration' ? '#9333ea' : 
                          type === 'geological' ? '#2563eb' : 
                          type === 'band' ? '#16a34a' : 
                          type === 'ratio' ? '#d97706' : 
                          type === 'vnir' ? '#06b6d4' :
                          '#eab308'
                      }}
                    ></div>
                    <span className="text-xs">
                      {value.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
                    </span>
                  </div>
                ) : null
              )}
              {showProspectivity && (
                <div className="flex items-center">
                  <div className="w-3 h-3 rounded-full mr-1 bg-red-500"></div>
                  <span className="text-xs">Gold Prospectivity</span>
                </div>
              )}
              {Object.values(selectedLayers).every(v => !v) && !showProspectivity && (
                <span className="text-xs text-gray-500 italic">No layers active</span>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Right Sidebar */}
      <div className={`bg-white shadow-lg transition-all duration-300 z-10 ${rightSidebarExpanded ? 'w-80' : 'w-16'}`}>
        <div className="p-4 h-full flex flex-col">
          <div className="flex justify-between items-center mb-4">
            {rightSidebarExpanded && <h2 className="text-xl font-bold text-gray-800">Analysis</h2>}
            <button 
              onClick={toggleRightSidebar}
              className="p-2 rounded-md bg-gray-200 hover:bg-gray-300"
            >
              {rightSidebarExpanded ? <PanelRight className="w-5 h-5" /> : <PanelLeft className="w-5 h-5" />}
            </button>
          </div>

          {rightSidebarExpanded && (
            <AnalysisPanel
              sceneId={selectedScene}
              showProspectivity={showProspectivity}
              onShowProspectivityChange={handleProspectivityToggle}
              prospectivityThreshold={prospectivityThreshold}
              onProspectivityThresholdChange={handleProspectivityThresholdChange}
            />
          )}
        </div>
      </div>
    </div>
  );
};

export default Explorer;