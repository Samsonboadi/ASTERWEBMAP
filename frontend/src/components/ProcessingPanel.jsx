// src/components/ProcessingPanel.jsx
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
  Alert,
  AlertTitle,
  AlertDescription
} from '../components/ui';
import { 
  Upload, 
  Server, 
  UploadCloud, 
  Loader2, 
  CheckCircle, 
  AlertCircle,
  Clock,
  Filter
} from 'lucide-react';
import { processScene, getProcessingStatus, uploadAsterData } from '../services/api';
import { ProcessingStatus } from '../enums';
import { toast } from '../components/ui';

const ProcessingPanel = ({ sceneId, onProcessingComplete }) => {
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [selectedFile, setSelectedFile] = useState(null);
  const [processingOptions, setProcessingOptions] = useState({
    minerals: true,
    alteration: true,
    goldPathfinders: true,
    enhancedVisualization: true
  });
  const [processingStatus, setProcessingStatus] = useState(null);
  const [processingProgress, setProcessingProgress] = useState(0);
  const [processingStage, setProcessingStage] = useState(null);
  const [isPolling, setIsPolling] = useState(false);

  // Handle file selection
  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
    }
  };

  // Handle option changes
  const handleOptionChange = (option, value) => {
    setProcessingOptions(prev => ({
      ...prev,
      [option]: value
    }));
  };

  // Upload and process file
const handleUploadAndProcess = async () => {
  if (!selectedFile) {
    toast({
      title: "No File Selected",
      description: "Please select an ASTER data file to upload.",
      variant: "destructive"
    });
    return;
  }
  
  setIsUploading(true);
  setUploadProgress(0);
  
  try {
    // Get the returned object with promise and progressTracker
    const upload = await uploadAsterData(selectedFile);
    
    // Set up the progress tracking function
    upload.progressTracker.onUploadProgress = (progress) => {
      setUploadProgress(progress);
    };
    
    // Wait for the actual upload to complete
    const result = await upload.promise;
    
    // Ensure the progress bar shows 100%
    setUploadProgress(100);
    
    toast({
      title: "Upload Complete",
      description: "ASTER data has been uploaded successfully.",
    });
    
    // Start processing the uploaded scene
    if (result && result.sceneId) {
      setIsUploading(false);
      startProcessing(result.sceneId);
    }
    
    // Reset file selection
    setSelectedFile(null);
  } catch (error) {
    setIsUploading(false);
    toast({
      title: "Upload Failed",
      description: error.message || "An unknown error occurred during upload.",
      variant: "destructive"
    });
    console.error("Upload error:", error);
  }
};

  // Start processing an existing scene
  const startProcessing = async (id) => {
    const targetSceneId = id || sceneId;
    if (!targetSceneId) {
      toast({
        title: "No Scene Selected",
        description: "Please select a scene to process or upload a new file.",
        variant: "destructive"
      });
      return;
    }
    
    try {
      const result = await processScene(targetSceneId, processingOptions);
      
      toast({
        title: "Processing Started",
        description: "Scene processing has been initiated.",
      });
      
      // Begin polling for status
      setProcessingStatus(ProcessingStatus.PROCESSING);
      setIsPolling(true);
    } catch (error) {
      toast({
        title: "Processing Failed",
        description: error.message || "Failed to start processing.",
        variant: "destructive"
      });
      console.error("Processing error:", error);
    }
  };
  
  // Polling for processing status
  useEffect(() => {
    let interval;
    
    if (isPolling && sceneId) {
      interval = setInterval(async () => {
        try {
          const status = await getProcessingStatus(sceneId);
          setProcessingStatus(status.status);
          setProcessingProgress(status.progress || 0);
          setProcessingStage(status.stage);
          
          // Stop polling when processing is completed or failed
          if (status.status === ProcessingStatus.COMPLETED || 
              status.status === ProcessingStatus.FAILED) {
            setIsPolling(false);
            
            if (status.status === ProcessingStatus.COMPLETED && onProcessingComplete) {
              onProcessingComplete(sceneId);
              toast({
                title: "Processing Complete",
                description: "Scene has been processed successfully.",
                variant: "success"
              });
            } else if (status.status === ProcessingStatus.FAILED) {
              toast({
                title: "Processing Failed",
                description: status.error || "An error occurred during processing.",
                variant: "destructive"
              });
            }
          }
        } catch (error) {
          console.error("Error checking processing status:", error);
        }
      }, 3000);
    }
    
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isPolling, sceneId, onProcessingComplete]);
  
  // Check status on initial load if sceneId is provided
  useEffect(() => {
    if (sceneId) {
      const checkStatus = async () => {
        try {
          const status = await getProcessingStatus(sceneId);
          setProcessingStatus(status.status);
          setProcessingProgress(status.progress || 0);
          setProcessingStage(status.stage);
          
          if (status.status === ProcessingStatus.PROCESSING) {
            setIsPolling(true);
          }
        } catch (error) {
          console.error("Error checking initial status:", error);
        }
      };
      
      checkStatus();
    }
  }, [sceneId]);
  
  // Render status indicator
  const renderStatusIndicator = () => {
    switch (processingStatus) {
      case ProcessingStatus.IDLE:
        return (
          <div className="flex items-center text-gray-500">
            <Clock className="w-5 h-5 mr-2" />
            <span>Idle - Ready to Process</span>
          </div>
        );
      case ProcessingStatus.QUEUED:
        return (
          <div className="flex items-center text-yellow-500">
            <Clock className="w-5 h-5 mr-2" />
            <span>Queued - Waiting to Start</span>
          </div>
        );
      case ProcessingStatus.PROCESSING:
        return (
          <div className="flex flex-col space-y-2">
            <div className="flex items-center text-blue-500">
              <Loader2 className="w-5 h-5 mr-2 animate-spin" />
              <span>Processing - {processingStage || "Initializing"}</span>
            </div>
            <Progress value={processingProgress} />
            <div className="text-xs text-gray-500 text-right">{Math.round(processingProgress)}% Complete</div>
          </div>
        );
      case ProcessingStatus.COMPLETED:
        return (
          <div className="flex items-center text-green-500">
            <CheckCircle className="w-5 h-5 mr-2" />
            <span>Completed - Ready to Explore</span>
          </div>
        );
      case ProcessingStatus.FAILED:
        return (
          <div className="flex items-center text-red-500">
            <AlertCircle className="w-5 h-5 mr-2" />
            <span>Failed - Processing Error</span>
          </div>
        );
      default:
        return null;
    }
  };

  return (
    <Card className="p-4 h-full flex flex-col">
      <h2 className="text-xl font-bold mb-4">Process ASTER Data</h2>
      
      {/* File Upload Section */}
      <div className="mb-4">
        <h3 className="text-sm font-medium mb-2">Upload New Data</h3>
        
        <div className="border-2 border-dashed border-gray-300 rounded-md p-4 mb-2">
          <input
            type="file"
            id="aster-file"
            accept=".zip,.hdf,.tif,.tiff"
            onChange={handleFileChange}
            disabled={isUploading || processingStatus === ProcessingStatus.PROCESSING}
            className="hidden"
          />
          <label 
            htmlFor="aster-file" 
            className="flex flex-col items-center justify-center cursor-pointer"
          >
            <UploadCloud className="w-8 h-8 text-gray-400 mb-2" />
            <span className="text-sm text-gray-500">
              {selectedFile ? selectedFile.name : "Click to select ASTER data file"}
            </span>
            <span className="text-xs text-gray-400 mt-1">
              (.zip, .hdf, .tif formats supported)
            </span>
          </label>
        </div>
        
        {selectedFile && (
          <div>
            <Button 
              onClick={handleUploadAndProcess} 
              disabled={isUploading || !selectedFile || processingStatus === ProcessingStatus.PROCESSING}
              className="w-full"
            >
              {isUploading ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Uploading...
                </>
              ) : (
                <>
                  <Upload className="w-4 h-4 mr-2" />
                  Upload & Process
                </>
              )}
            </Button>
            
            {isUploading && (
              <Progress value={uploadProgress} className="mt-2" />
            )}
          </div>
        )}
      </div>
      
      <Separator className="my-4" />
      
      {/* Processing Options */}
      <div className="mb-4">
        <h3 className="text-sm font-medium mb-2">Processing Options</h3>
        
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-sm">Mineral Mapping</span>
            <input
              type="checkbox"
              checked={processingOptions.minerals}
              onChange={(e) => handleOptionChange('minerals', e.target.checked)}
              disabled={processingStatus === ProcessingStatus.PROCESSING}
              className="h-4 w-4 rounded border-gray-300 text-indigo-600 focus:ring-indigo-500"
            />
          </div>
          
          <div className="flex items-center justify-between">
            <span className="text-sm">Alteration Mapping</span>
            <input
              type="checkbox"
              checked={processingOptions.alteration}
              onChange={(e) => handleOptionChange('alteration', e.target.checked)}
              disabled={processingStatus === ProcessingStatus.PROCESSING}
              className="h-4 w-4 rounded border-gray-300 text-indigo-600 focus:ring-indigo-500"
            />
          </div>
          
          <div className="flex items-center justify-between">
            <span className="text-sm">Gold Pathfinders</span>
            <input
              type="checkbox"
              checked={processingOptions.goldPathfinders}
              onChange={(e) => handleOptionChange('goldPathfinders', e.target.checked)}
              disabled={processingStatus === ProcessingStatus.PROCESSING}
              className="h-4 w-4 rounded border-gray-300 text-indigo-600 focus:ring-indigo-500"
            />
          </div>
          
          <div className="flex items-center justify-between">
            <span className="text-sm">Enhanced Visualization</span>
            <input
              type="checkbox"
              checked={processingOptions.enhancedVisualization}
              onChange={(e) => handleOptionChange('enhancedVisualization', e.target.checked)}
              disabled={processingStatus === ProcessingStatus.PROCESSING}
              className="h-4 w-4 rounded border-gray-300 text-indigo-600 focus:ring-indigo-500"
            />
          </div>
        </div>
      </div>
      
      {/* Status and Actions */}
      <div className="mt-auto">
        <Separator className="mb-4" />
        
        <div className="space-y-4">
          <div className="bg-gray-50 p-3 rounded-md">
            <h3 className="text-sm font-medium mb-2">Processing Status</h3>
            {renderStatusIndicator()}
          </div>
          
          {sceneId && processingStatus !== ProcessingStatus.PROCESSING && (
            <Button 
              onClick={() => startProcessing()} 
              disabled={processingStatus === ProcessingStatus.PROCESSING}
              className="w-full"
            >
              <Server className="w-4 h-4 mr-2" />
              {processingStatus === ProcessingStatus.COMPLETED ? "Reprocess Scene" : "Process Scene"}
            </Button>
          )}
        </div>
        
        {processingStatus === ProcessingStatus.FAILED && (
          <Alert variant="destructive" className="mt-4">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Processing Error</AlertTitle>
            <AlertDescription>
              There was an error processing this scene. Please try again or contact support.
            </AlertDescription>
          </Alert>
        )}
      </div>
    </Card>
  );
};

export default ProcessingPanel;