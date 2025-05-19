// src/components/AnalysisPanel.jsx
import React, { useState, useEffect } from 'react';
import { 
  Card, 
  Button,
  Progress,
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
  Separator,
  Accordion,
  AccordionItem,
  AccordionTrigger,
  AccordionContent,
  Tabs,
  TabsList,
  TabsTrigger,
  TabsContent,
  Switch,
  Label
} from '../components/ui';
import { 
  BarChart2, 
  Download, 
  FileText, 
  Zap, 
  Layers,
  ChevronRight,
  Loader2
} from 'lucide-react';
import { getSceneStatistics, generateProspectivityMap, generateReport } from '../services/api';
import { ReportTypes, AlterationIndices, GoldPathfinderIndices } from '../enums';
import { toast } from '../components/ui';

const AnalysisPanel = ({ 
  sceneId, 
  showProspectivity = false,
  onShowProspectivityChange,
  prospectivityThreshold = 0.7,
  onProspectivityThresholdChange
}) => {
  const [statistics, setStatistics] = useState(null);
  const [isLoadingStats, setIsLoadingStats] = useState(false);
  const [isGeneratingProspectivity, setIsGeneratingProspectivity] = useState(false);
  const [isGeneratingReport, setIsGeneratingReport] = useState(false);
  const [selectedReport, setSelectedReport] = useState(ReportTypes.COMPREHENSIVE);
  const [selectedPathfinders, setSelectedPathfinders] = useState([
    GoldPathfinderIndices.ADVANCED_ARGILLIC_GOLD,
    GoldPathfinderIndices.PYRITE,
    GoldPathfinderIndices.ARSENOPYRITE
  ]);
  const [selectedAlterations, setSelectedAlterations] = useState([
    AlterationIndices.ADVANCED_ARGILLIC,
    AlterationIndices.PHYLLIC
  ]);
  
  useEffect(() => {
    if (sceneId) {
      fetchStatistics();
    }
  }, [sceneId]);
  
  const fetchStatistics = async () => {
    if (!sceneId) return;
    
    setIsLoadingStats(true);
    try {
      const data = await getSceneStatistics(sceneId);
      setStatistics(data);
    } catch (error) {
      toast({
        title: "Error Loading Statistics",
        description: error.message || "Failed to load scene statistics.",
        variant: "destructive"
      });
      console.error("Failed to load scene statistics:", error);
    } finally {
      setIsLoadingStats(false);
    }
  };
  
  const handleGenerateProspectivityMap = async () => {
    if (!sceneId) return;
    
    setIsGeneratingProspectivity(true);
    try {
      const options = {
        threshold: prospectivityThreshold,
        pathfinders: selectedPathfinders,
        alterations: selectedAlterations
      };
      
      await generateProspectivityMap(sceneId, options);
      
      if (onShowProspectivityChange) {
        onShowProspectivityChange(true);
      }
      
      toast({
        title: "Prospectivity Map Generated",
        description: "Gold prospectivity map has been generated successfully.",
      });
    } catch (error) {
      toast({
        title: "Error Generating Prospectivity Map",
        description: error.message || "Failed to generate prospectivity map.",
        variant: "destructive"
      });
      console.error("Failed to generate prospectivity map:", error);
    } finally {
      setIsGeneratingProspectivity(false);
    }
  };
  
  const handleGenerateReport = async () => {
    if (!sceneId) return;
    
    setIsGeneratingReport(true);
    try {
      const options = {
        includeProspectivity: showProspectivity,
        prospectivityThreshold: prospectivityThreshold,
        pathfinders: selectedPathfinders,
        alterations: selectedAlterations
      };
      
      const result = await generateReport(sceneId, selectedReport, options);
      
      if (result && result.reportUrl) {
        window.open(result.reportUrl, '_blank');
      }
      
      toast({
        title: "Report Generated",
        description: "Analysis report has been generated successfully.",
      });
    } catch (error) {
      toast({
        title: "Error Generating Report",
        description: error.message || "Failed to generate analysis report.",
        variant: "destructive"
      });
      console.error("Failed to generate report:", error);
    } finally {
      setIsGeneratingReport(false);
    }
  };
  
  const handleToggleProspectivity = (checked) => {
    if (onShowProspectivityChange) {
      onShowProspectivityChange(checked);
    }
  };
  
  const handleThresholdChange = (e) => {
    const value = parseFloat(e.target.value);
    if (onProspectivityThresholdChange) {
      onProspectivityThresholdChange(value);
    }
  };
  
  const handlePathfinderToggle = (pathfinder) => {
    setSelectedPathfinders(prev => {
      if (prev.includes(pathfinder)) {
        return prev.filter(p => p !== pathfinder);
      } else {
        return [...prev, pathfinder];
      }
    });
  };
  
  const handleAlterationToggle = (alteration) => {
    setSelectedAlterations(prev => {
      if (prev.includes(alteration)) {
        return prev.filter(a => a !== alteration);
      } else {
        return [...prev, alteration];
      }
    });
  };
  
  return (
    <Card className="p-4 h-full flex flex-col">
      <h2 className="text-xl font-bold mb-4">Analysis Tools</h2>
      
      <Accordion type="single" collapsible defaultValue="statistics">
        {/* Statistics Section */}
        <AccordionItem value="statistics">
          <AccordionTrigger className="py-2">
            <div className="flex items-center">
              <BarChart2 className="w-5 h-5 mr-2" />
              <span>Scene Statistics</span>
            </div>
          </AccordionTrigger>
          <AccordionContent>
            <div className="py-2">
              {isLoadingStats ? (
                <div className="flex justify-center p-4">
                  <Loader2 className="w-6 h-6 text-blue-500 animate-spin" />
                </div>
              ) : !statistics ? (
                <div className="text-center text-gray-500 p-4">
                  {sceneId ? 
                    "No statistics available for this scene." : 
                    "Select a scene to view statistics."}
                </div>
              ) : (
                <div className="space-y-3">
                  <div className="bg-gray-50 p-3 rounded-md space-y-2">
                    <div className="flex justify-between">
                      <span className="text-sm font-medium">Scene ID:</span>
                      <span className="text-sm">{sceneId}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm font-medium">Acquisition Date:</span>
                      <span className="text-sm">{statistics.acquisitionDate || "Unknown"}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm font-medium">Cloud Cover:</span>
                      <span className="text-sm">{statistics.cloudCover || 0}%</span>
                    </div>
                  </div>
                  
                  <Separator />
                  
                  <div className="bg-gray-50 p-3 rounded-md space-y-2">
                    <div className="flex justify-between">
                      <span className="text-sm font-medium">Dominant Alteration:</span>
                      <span className="text-sm">{statistics.dominantAlteration || "None"}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm font-medium">Dominant Minerals:</span>
                      <span className="text-sm">{statistics.dominantMinerals || "None"}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm font-medium">Gold Pathfinder Coverage:</span>
                      <span className="text-sm">{statistics.goldPathfinderCoverage || 0}%</span>
                    </div>
                  </div>
                  
                  <Button 
                    variant="outline" 
                    onClick={fetchStatistics}
                    size="sm"
                    className="w-full"
                  >
                    Refresh Statistics
                  </Button>
                </div>
              )}
            </div>
          </AccordionContent>
        </AccordionItem>
        
        {/* Gold Prospectivity Section */}
        <AccordionItem value="prospectivity">
          <AccordionTrigger className="py-2">
            <div className="flex items-center">
              <Zap className="w-5 h-5 mr-2 text-yellow-500" />
              <span>Gold Prospectivity</span>
            </div>
          </AccordionTrigger>
          <AccordionContent>
            <div className="py-2 space-y-4">
              <div className="flex items-center justify-between">
                <Label htmlFor="show-prospectivity" className="text-sm font-medium">
                  Show Prospectivity Map
                </Label>
                <Switch
                  id="show-prospectivity"
                  checked={showProspectivity}
                  onCheckedChange={handleToggleProspectivity}
                />
              </div>
              
              <div>
                <Label htmlFor="prospectivity-threshold" className="text-sm font-medium mb-1">
                  Prospectivity Threshold: {prospectivityThreshold}
                </Label>
                <input
                  type="range"
                  id="prospectivity-threshold"
                  min="0.1"
                  max="0.9"
                  step="0.05"
                  value={prospectivityThreshold}
                  onChange={handleThresholdChange}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>0.1 (Low)</span>
                  <span>0.5</span>
                  <span>0.9 (High)</span>
                </div>
              </div>
              
              <Separator />
              
              <div>
                <Label className="text-sm font-medium mb-2 block">Pathfinder Minerals</Label>
                <div className="grid grid-cols-2 gap-2">
                  {Object.values(GoldPathfinderIndices).map(pathfinder => (
                    <div 
                      key={pathfinder}
                      className={`
                        px-2 py-1 text-xs rounded-md cursor-pointer
                        ${selectedPathfinders.includes(pathfinder) 
                          ? 'bg-yellow-100 border border-yellow-300' 
                          : 'bg-gray-100 border border-gray-200'}
                      `}
                      onClick={() => handlePathfinderToggle(pathfinder)}
                    >
                      {pathfinder.replace(/_/g, ' ')}
                    </div>
                  ))}
                </div>
              </div>
              
              <div>
                <Label className="text-sm font-medium mb-2 block">Alteration Types</Label>
                <div className="grid grid-cols-2 gap-2">
                  {Object.values(AlterationIndices).map(alteration => (
                    <div 
                      key={alteration}
                      className={`
                        px-2 py-1 text-xs rounded-md cursor-pointer
                        ${selectedAlterations.includes(alteration) 
                          ? 'bg-purple-100 border border-purple-300' 
                          : 'bg-gray-100 border border-gray-200'}
                      `}
                      onClick={() => handleAlterationToggle(alteration)}
                    >
                      {alteration.replace(/_/g, ' ')}
                    </div>
                  ))}
                </div>
              </div>
              
              <Button
                onClick={handleGenerateProspectivityMap}
                disabled={!sceneId || isGeneratingProspectivity}
                className="w-full"
              >
                {isGeneratingProspectivity ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Generating...
                  </>
                ) : (
                  <>
                    <Zap className="w-4 h-4 mr-2" />
                    Generate Prospectivity Map
                  </>
                )}
              </Button>
              
              <div className="p-3 bg-yellow-50 border border-yellow-200 rounded-md">
                <div className="flex items-start">
                  <div className="flex items-center">
                    <div className="w-4 h-4 bg-red-500 mr-2"></div>
                    <span className="text-xs">High Prospectivity (&gt;{prospectivityThreshold})</span>
                  </div>
                </div>
                <div className="flex items-center mt-1">
                  <div className="w-4 h-4 bg-orange-400 mr-2"></div>
                  <span className="text-xs">Medium Prospectivity (0.5-{prospectivityThreshold})</span>
                </div>
                <div className="flex items-center mt-1">
                  <div className="w-4 h-4 bg-yellow-300 mr-2"></div>
                  <span className="text-xs">Low Prospectivity (&lt;0.5)</span>
                </div>
              </div>
            </div>
          </AccordionContent>
        </AccordionItem>
        
        {/* Reports Section */}
        <AccordionItem value="reports">
          <AccordionTrigger className="py-2">
            <div className="flex items-center">
              <FileText className="w-5 h-5 mr-2" />
              <span>Analysis Reports</span>
            </div>
          </AccordionTrigger>
          <AccordionContent>
            <div className="py-2 space-y-4">
              <div>
                <Label htmlFor="report-type" className="text-sm font-medium mb-2 block">
                  Report Type
                </Label>
                <Select
                  value={selectedReport}
                  onValueChange={setSelectedReport}
                >
                  <SelectTrigger id="report-type">
                    <SelectValue placeholder="Select report type" />
                  </SelectTrigger>
                  <SelectContent>
                    {Object.values(ReportTypes).map(type => (
                      <SelectItem key={type} value={type}>
                        {type.replace(/_/g, ' ')}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm">Include Prospectivity Analysis</span>
                  <ChevronRight className="w-4 h-4 text-gray-400" />
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm">Include Recommendations</span>
                  <ChevronRight className="w-4 h-4 text-gray-400" />
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm">Include Statistical Graphs</span>
                  <ChevronRight className="w-4 h-4 text-gray-400" />
                </div>
              </div>
              
              <Button
                onClick={handleGenerateReport}
                disabled={!sceneId || isGeneratingReport}
                className="w-full"
              >
                {isGeneratingReport ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Generating...
                  </>
                ) : (
                  <>
                    <FileText className="w-4 h-4 mr-2" />
                    Generate Report
                  </>
                )}
              </Button>
              
              <Button
                variant="outline"
                disabled={!sceneId}
                className="w-full"
              >
                <Download className="w-4 h-4 mr-2" />
                Export All Maps
              </Button>
            </div>
          </AccordionContent>
        </AccordionItem>
        
        {/* Layer Combinations Section */}
        <AccordionItem value="layers">
          <AccordionTrigger className="py-2">
            <div className="flex items-center">
              <Layers className="w-5 h-5 mr-2" />
              <span>Layer Combinations</span>
            </div>
          </AccordionTrigger>
          <AccordionContent>
            <div className="py-2 space-y-4">
              <div className="text-sm text-gray-500">
                Create custom layer combinations to enhance visualization and analysis.
              </div>
              
              <Button
                variant="outline"
                disabled={!sceneId}
                className="w-full"
              >
                <Layers className="w-4 h-4 mr-2" />
                Create Custom Combination
              </Button>
            </div>
          </AccordionContent>
        </AccordionItem>
      </Accordion>
    </Card>
  );
};

export default AnalysisPanel;