// src/pages/Explorer.jsx
import React, { useState, useEffect } from 'react';
import { PanelLeft, PanelRight, Map as MapIcon, Layers, Eye } from 'lucide-react';
import { Button } from '../components/ui';
import ASTERMap from '../components/ASTERMap';
import ProcessingPanel from '../components/ProcessingPanel';
import AnalysisPanel from '../components/AnalysisPanel';
import { getScenes, getSceneById, getAvailableLayers } from '../services/api';
import { toast } from '../components/ui/toast';

const Explorer = () => {
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

  // Fetch scenes on component mount
  useEffect(() => {
    fetchScenes();
  }, []);

  // Fetch scene details when a scene is selected
  useEffect(() => {
    if (selectedScene) {
      fetchSceneDetails();
      fetchAvailableLayers();
    }
  }, [selectedScene]);

  const fetchScenes = async () => {
    setIsLoading(true);
    try {
      const data = await getScenes();
      setScenes(data);
      
      // Select the first scene if available
      if (data.length > 0) {
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

  const handleProcessingComplete = (sceneId) => {
    setSelectedScene(sceneId);
    fetchSceneDetails();
    fetchAvailableLayers();
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

          {leftSidebarExpanded && (
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
                        onChange={(e) => setSelectedScene(e.target.value)}
                        className="w-full p-2 border border-gray-300 rounded-md"
                        disabled={isLoading}
                      >
                        <option value="">Select a scene</option>
                        {scenes.map(scene => (
                          <option key={scene.id} value={scene.id}>{scene.name}</option>
                        ))}
                      </select>
                    </div>
                    
                    {/* Layer Selection */}
                    <div className="space-y-3">
                      <h3 className="text-sm font-medium text-gray-700">Available Layers</h3>
                      
                      {/* Alteration Layer Selection */}
                      <div>
                        <label className="block text-xs text-gray-500 mb-1">Alteration Indices</label>
                        <select 
                          value={selectedLayers.alteration}
                          onChange={(e) => handleLayerChange('alteration', e.target.value)}
                          className="w-full p-2 border border-gray-300 rounded-md text-sm"
                        >
                          <option value="">None</option>
                          {availableLayers.alteration?.map(layer => (
                            <option key={layer} value={layer}>
                              {layer.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
                            </option>
                          ))}
                        </select>
                      </div>
                      
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
                        <label className="block text-xs text-gray-500 mb-1">Gold Pathfinders</label>
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
        </div>
      </div>

      {/* Main content area with map */}
      <div className="flex-1 relative">
        <ASTERMap 
          sceneId={selectedScene}
          selectedLayers={selectedLayers}
          showProspectivity={showProspectivity}
          prospectivityThreshold={prospectivityThreshold}
          imageBounds={getImageBounds()}
        />
        
        {/* Map overlay controls and info panel */}
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