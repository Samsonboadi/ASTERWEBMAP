// src/pages/SceneManager.jsx
import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
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
  Tabs,
  TabsList,
  TabsTrigger,
  TabsContent
} from '../components/ui';
import {
  Upload,
  Loader2,
  MapPin,
  ChevronRight,
  Search,
  Calendar,
  Cloud,
  Check,
  AlertCircle,
  Clock,
  Trash2,
  Download
} from 'lucide-react';
import { getScenes, getProcessingStatus, uploadAsterData } from '../services/api';
import { ProcessingStatus } from '../enums';
import { toast } from '../components/ui';

const SceneManager = () => {
  const [scenes, setScenes] = useState([]);
  const [filteredScenes, setFilteredScenes] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [selectedFile, setSelectedFile] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');
  const [dateFilter, setDateFilter] = useState('all');
  const [activeTab, setActiveTab] = useState('all');

  useEffect(() => {
    fetchScenes();
  }, []);

  useEffect(() => {
    filterScenes();
  }, [scenes, searchQuery, statusFilter, dateFilter, activeTab]);

  const fetchScenes = async () => {
  setIsLoading(true);
  try {
    console.log("Fetching scenes...");
    const data = await getScenes();
    console.log("Received scenes:", data);
    
    if (!data || data.length === 0) {
      console.log("No scenes found or empty response");
      setScenes([]);
      setFilteredScenes([]);
      setIsLoading(false);
      return;
    }
    
    // Fetch processing status for each scene
    const scenesWithStatus = await Promise.all(
      data.map(async (scene) => {
        try {
          const status = await getProcessingStatus(scene.id);
          return {
            ...scene,
            status: status.status,
            progress: status.progress,
            stage: status.stage
          };
        } catch (error) {
          console.error(`Error fetching status for scene ${scene.id}:`, error);
          return {
            ...scene,
            status: ProcessingStatus.IDLE,
            progress: 0,
            stage: null
          };
        }
      })
    );
    
    console.log("Scenes with status:", scenesWithStatus);
    setScenes(scenesWithStatus);
    
    // Trigger filtering immediately
    filterScenes(scenesWithStatus);
  } catch (error) {
    console.error("Failed to load scenes:", error);
    toast({
      title: "Error Loading Scenes",
      description: error.message || "Failed to load scenes.",
      variant: "destructive"
    });
    
    // Clear scenes on error
    setScenes([]);
    setFilteredScenes([]);
  } finally {
    setIsLoading(false);
  }
};


// Updated filterScenes function that accepts scenes as parameter
const filterScenes = (scenesToFilter = scenes) => {
  let filtered = [...scenesToFilter];
  
  // Filter by search query
  if (searchQuery) {
    filtered = filtered.filter(scene => 
      (scene.name && scene.name.toLowerCase().includes(searchQuery.toLowerCase())) ||
      (scene.id && scene.id.toLowerCase().includes(searchQuery.toLowerCase()))
    );
  }
  
  // Filter by status
  if (statusFilter !== 'all') {
    filtered = filtered.filter(scene => scene.status === statusFilter);
  }
  
  // Filter by active tab
  if (activeTab !== 'all') {
    if (activeTab === 'completed') {
      filtered = filtered.filter(scene => scene.status === ProcessingStatus.COMPLETED);
    } else if (activeTab === 'processing') {
      filtered = filtered.filter(scene => scene.status === ProcessingStatus.PROCESSING);
    } else if (activeTab === 'pending') {
      filtered = filtered.filter(scene => 
        scene.status === ProcessingStatus.IDLE || scene.status === ProcessingStatus.QUEUED
      );
    }
  }
  
  // Filter by date (only if date is valid)
  if (dateFilter !== 'all') {
    const now = new Date();
    let dateLimit = new Date();
    
    if (dateFilter === 'today') {
      dateLimit.setDate(now.getDate() - 1);
    } else if (dateFilter === 'week') {
      dateLimit.setDate(now.getDate() - 7);
    } else if (dateFilter === 'month') {
      dateLimit.setMonth(now.getMonth() - 1);
    } else if (dateFilter === 'year') {
      dateLimit.setFullYear(now.getFullYear() - 1);
    }
    
    filtered = filtered.filter(scene => {
      // Only filter if scene has a valid date
      if (!scene.date || scene.date === 'Unknown') return true;
      
      try {
        const sceneDate = new Date(scene.date);
        return !isNaN(sceneDate) && sceneDate >= dateLimit;
      } catch (e) {
        console.error("Invalid date format:", scene.date);
        return true; // Include scenes with invalid dates
      }
    });
  }
  
  console.log("Filtered scenes:", filtered);
  setFilteredScenes(filtered);
};
  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
    }
  };

const handleUpload = async () => {
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
      description: result.processingStarted 
        ? "ASTER data has been uploaded and processing has started automatically."
        : "ASTER data has been uploaded successfully. You can now start processing.",
      variant: "success"
    });
    
    // Add a delay to allow backend to set up the scene directories
    setTimeout(() => {
      // Refresh scenes list to show the new scene
      fetchScenes();
      
      // Reset file selection
      setSelectedFile(null);
      
      // If processing was started automatically, switch to Processing tab
      if (result.processingStarted) {
        setActiveTab('processing');
      }
    }, 1500);
    
  } catch (error) {
    toast({
      title: "Upload Failed",
      description: error.message || "An unknown error occurred during upload.",
      variant: "destructive"
    });
    console.error("Upload error:", error);
  } finally {
    setIsUploading(false);
  }
};

  const renderStatusIcon = (status) => {
    switch (status) {
      case ProcessingStatus.IDLE:
        return <Clock className="w-5 h-5 text-gray-500" />;
      case ProcessingStatus.QUEUED:
        return <Clock className="w-5 h-5 text-yellow-500" />;
      case ProcessingStatus.PROCESSING:
        return <Loader2 className="w-5 h-5 text-blue-500 animate-spin" />;
      case ProcessingStatus.COMPLETED:
        return <Check className="w-5 h-5 text-green-500" />;
      case ProcessingStatus.FAILED:
        return <AlertCircle className="w-5 h-5 text-red-500" />;
      default:
        return null;
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case ProcessingStatus.IDLE:
        return 'bg-gray-100 text-gray-700';
      case ProcessingStatus.QUEUED:
        return 'bg-yellow-100 text-yellow-700';
      case ProcessingStatus.PROCESSING:
        return 'bg-blue-100 text-blue-700';
      case ProcessingStatus.COMPLETED:
        return 'bg-green-100 text-green-700';
      case ProcessingStatus.FAILED:
        return 'bg-red-100 text-red-700';
      default:
        return 'bg-gray-100 text-gray-700';
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="flex justify-between items-center mb-8">
        <div>
          <h1 className="text-2xl font-bold mb-2">Scene Manager</h1>
          <p className="text-gray-600">Manage your ASTER scenes and processing jobs</p>
        </div>
        <Button onClick={fetchScenes}>
          Refresh
        </Button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
        {/* Upload Card */}
        <Card className="p-4">
          <h2 className="text-lg font-semibold mb-4 flex items-center">
            <Upload className="w-5 h-5 mr-2 text-indigo-600" />
            Upload ASTER Data
          </h2>
          
          <div className="space-y-4">
            <div className="border-2 border-dashed border-gray-300 rounded-md p-4 hover:border-indigo-300 transition-colors">
              <input
                type="file"
                id="aster-file"
                accept=".zip,.hdf,.tif,.tiff"
                onChange={handleFileChange}
                disabled={isUploading}
                className="hidden"
              />
              <label 
                htmlFor="aster-file" 
                className="flex flex-col items-center justify-center cursor-pointer"
              >
                <Upload className={`w-8 h-8 ${selectedFile ? 'text-indigo-500' : 'text-gray-400'} mb-2`} />
                <span className="text-sm text-gray-700">
                  {selectedFile ? selectedFile.name : "Click or drag to upload ASTER data file"}
                </span>
                <span className="text-xs text-gray-400 mt-1">
                  (.zip, .hdf, .tif formats supported)
                </span>
              </label>
            </div>

            {selectedFile && (
              <div className="mt-4">
                <Button 
                  onClick={handleUpload} 
                  disabled={isUploading || !selectedFile}
                  className="w-full"
                >
                  {isUploading ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      Uploading... {Math.round(uploadProgress)}%
                    </>
                  ) : (
                    <>
                      <Upload className="w-4 h-4 mr-2" />
                      Upload File
                    </>
                  )}
                </Button>
                
                {isUploading && (
                  <div className="mt-2">
                    <Progress value={uploadProgress} />
                    <div className="text-xs text-gray-500 mt-1 text-right">
                      {Math.round(uploadProgress)}% complete
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
          
          <Separator className="my-4" />
          
          <div className="space-y-2">
            <h3 className="text-sm font-medium mb-2">Upload Guidelines</h3>
            <ul className="text-xs text-gray-500 space-y-1 list-disc pl-4">
              <li>Supported formats: ZIP archive with HDF files</li>
              <li>Maximum file size: 1GB</li>
              <li>ASTER L1B/L2 data recommended</li>
              <li>Include both VNIR and SWIR bands</li>
            </ul>
          </div>
        </Card>
        
        {/* Statistics Card */}
        <Card className="p-4 col-span-2">
          <h2 className="text-lg font-semibold mb-4">Scene Statistics</h2>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
            <div className="bg-gray-50 p-3 rounded-md text-center">
              <p className="text-2xl font-bold text-gray-800">{scenes.length}</p>
              <p className="text-xs text-gray-500">Total Scenes</p>
            </div>
            
            <div className="bg-green-50 p-3 rounded-md text-center">
              <p className="text-2xl font-bold text-green-600">
                {scenes.filter(s => s.status === ProcessingStatus.COMPLETED).length}
              </p>
              <p className="text-xs text-gray-500">Processed</p>
            </div>
            
            <div className="bg-blue-50 p-3 rounded-md text-center">
              <p className="text-2xl font-bold text-blue-600">
                {scenes.filter(s => s.status === ProcessingStatus.PROCESSING).length}
              </p>
              <p className="text-xs text-gray-500">Processing</p>
            </div>
            
            <div className="bg-red-50 p-3 rounded-md text-center">
              <p className="text-2xl font-bold text-red-600">
                {scenes.filter(s => s.status === ProcessingStatus.FAILED).length}
              </p>
              <p className="text-xs text-gray-500">Failed</p>
            </div>
          </div>
          
          <Separator className="mb-4" />
          
          {/* Search and Filters */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <Search className="h-4 w-4 text-gray-400" />
              </div>
              <input
                type="text"
                placeholder="Search scenes..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10 pr-4 py-2 border border-gray-300 rounded-md w-full"
              />
            </div>
            
            <Select value={statusFilter} onValueChange={setStatusFilter}>
              <SelectTrigger>
                <SelectValue placeholder="Filter by status" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Statuses</SelectItem>
                <SelectItem value={ProcessingStatus.COMPLETED}>Completed</SelectItem>
                <SelectItem value={ProcessingStatus.PROCESSING}>Processing</SelectItem>
                <SelectItem value={ProcessingStatus.QUEUED}>Queued</SelectItem>
                <SelectItem value={ProcessingStatus.IDLE}>Idle</SelectItem>
                <SelectItem value={ProcessingStatus.FAILED}>Failed</SelectItem>
              </SelectContent>
            </Select>
            
            <Select value={dateFilter} onValueChange={setDateFilter}>
              <SelectTrigger>
                <SelectValue placeholder="Filter by date" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Time</SelectItem>
                <SelectItem value="today">Today</SelectItem>
                <SelectItem value="week">This Week</SelectItem>
                <SelectItem value="month">This Month</SelectItem>
                <SelectItem value="year">This Year</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </Card>
      </div>
      
      {/* Scene List Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab} className="mb-6">
        <TabsList className="grid grid-cols-4 w-full md:w-1/2">
          <TabsTrigger value="all">All</TabsTrigger>
          <TabsTrigger value="processing">Processing</TabsTrigger>
          <TabsTrigger value="completed">Completed</TabsTrigger>
          <TabsTrigger value="pending">Pending</TabsTrigger>
        </TabsList>
      </Tabs>
      
      {/* Scene List */}
      <div className="bg-white rounded-md shadow overflow-hidden">
        {isLoading ? (
          <div className="flex justify-center items-center p-8">
            <Loader2 className="w-8 h-8 text-indigo-600 animate-spin" />
            <span className="ml-2 text-gray-600">Loading scenes...</span>
          </div>
        ) : filteredScenes.length === 0 ? (
          <div className="text-center p-8">
            <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-gray-100 mb-4">
              <MapPin className="w-8 h-8 text-gray-400" />
            </div>
            <h3 className="text-lg font-medium text-gray-900 mb-1">No scenes found</h3>
            <p className="text-gray-500 max-w-md mx-auto">
              {searchQuery || statusFilter !== 'all' || dateFilter !== 'all' ? 
                "Try adjusting your search filters to find what you're looking for." : 
                "Upload ASTER data to get started with processing and analysis."}
            </p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Scene
                  </th>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Date
                  </th>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Status
                  </th>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Cloud Cover
                  </th>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Progress
                  </th>
                  <th scope="col" className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {filteredScenes.map((scene) => (
                  <tr key={scene.id} className="hover:bg-gray-50">
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-sm font-medium text-gray-900">{scene.name}</div>
                      <div className="text-xs text-gray-500">{scene.id}</div>
                      {scene.processingMode === 'VNIR-only' && (
                        <span className="inline-flex items-center px-2 py-0.5 mt-1 rounded text-xs font-medium bg-blue-100 text-blue-800">
                          VNIR Only
                        </span>
                      )}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-sm font-medium text-gray-900">{scene.name}</div>
                      <div className="text-xs text-gray-500">{scene.id}</div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        <Calendar className="h-4 w-4 text-gray-400 mr-1" />
                        <span className="text-sm text-gray-500">
                          {new Date(scene.date).toLocaleDateString()}
                        </span>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor(scene.status)}`}>
                        {renderStatusIcon(scene.status)}
                        <span className="ml-1 capitalize">{scene.status}</span>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        <Cloud className="h-4 w-4 text-gray-400 mr-1" />
                        <span className="text-sm text-gray-500">
                          {scene.cloudCover || 0}%
                        </span>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      {scene.status === ProcessingStatus.PROCESSING && (
                        <Progress value={scene.progress || 0} />
                      )}
                      {scene.status === ProcessingStatus.COMPLETED && (
                        <div className="flex items-center text-green-600">
                          <Check className="h-4 w-4 mr-1" />
                          <span className="text-xs">Complete</span>
                        </div>
                      )}
                      {scene.status === ProcessingStatus.FAILED && (
                        <div className="flex items-center text-red-600">
                          <AlertCircle className="h-4 w-4 mr-1" />
                          <span className="text-xs">Failed</span>
                        </div>
                      )}
                      {(scene.status === ProcessingStatus.IDLE || scene.status === ProcessingStatus.QUEUED) && (
                        <div className="flex items-center text-gray-500">
                          <Clock className="h-4 w-4 mr-1" />
                          <span className="text-xs">Waiting</span>
                        </div>
                      )}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                      <div className="flex justify-end space-x-2">
                        <Link to={`/explorer?scene=${scene.id}`}>
                          <Button size="sm" variant="outline">
                            View
                          </Button>
                        </Link>
                        <Button size="sm" variant="outline">
                          <Download className="h-4 w-4" />
                        </Button>
                        <Button size="sm" variant="outline" className="text-red-600 hover:text-red-800">
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};

export default SceneManager;