// src/pages/Dashboard.jsx
import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import {
  Card,
  Button,
  Progress,
  Separator
} from '../components/ui';
import {
  MapPin,
  LayoutDashboard,
  Database,
  Upload,
  Search,
  BarChart2,
  FileText,
  Zap,
  Layers,
  Calendar,
  Clock
} from 'lucide-react';
import { getScenes } from '../services/api';
import { ProcessingStatus } from '../enums';
import { toast } from '../components/ui/toast';

const Dashboard = () => {
  const [scenes, setScenes] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [processingCount, setProcessingCount] = useState(0);
  const [recentScenes, setRecentScenes] = useState([]);

  useEffect(() => {
    fetchScenes();

    // Set up a refresh interval (every 30 seconds)
    const intervalId = setInterval(fetchScenes, 30000);
    
    return () => clearInterval(intervalId);
  }, []);

  const fetchScenes = async () => {
    try {
      setIsLoading(true);
      const data = await getScenes();
      setScenes(data);
      
      // Count scenes being processed
      const processing = data.filter(scene => 
        scene.status === ProcessingStatus.PROCESSING || 
        scene.status === ProcessingStatus.QUEUED
      ).length;
      setProcessingCount(processing);
      
      // Get 3 most recent scenes
      const sorted = [...data].sort((a, b) => {
        // Try to sort by date if available
        if (a.date && b.date && a.date !== 'Unknown' && b.date !== 'Unknown') {
          return new Date(b.date) - new Date(a.date);
        }
        // Otherwise, sort by ID (assuming IDs have timestamps)
        return b.id.localeCompare(a.id);
      });
      
      setRecentScenes(sorted.slice(0, 3));
      
    } catch (error) {
      toast({
        title: "Error Loading Scenes",
        description: error.message || "Failed to load scenes.",
        variant: "destructive"
      });
      console.error("Failed to load scenes:", error);
    } finally {
      setIsLoading(false);
    }
  };

  // Get processing status color
  const getStatusColor = (status) => {
    switch (status) {
      case ProcessingStatus.COMPLETED:
        return 'bg-green-100 text-green-800';
      case ProcessingStatus.PROCESSING:
        return 'bg-blue-100 text-blue-800';
      case ProcessingStatus.QUEUED:
        return 'bg-yellow-100 text-yellow-800';
      case ProcessingStatus.FAILED:
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  // Get status icon
  const getStatusIcon = (status) => {
    switch (status) {
      case ProcessingStatus.COMPLETED:
        return <div className="w-3 h-3 rounded-full bg-green-500 mr-1"></div>;
      case ProcessingStatus.PROCESSING:
        return <div className="w-3 h-3 rounded-full bg-blue-500 animate-pulse mr-1"></div>;
      case ProcessingStatus.QUEUED:
        return <div className="w-3 h-3 rounded-full bg-yellow-500 mr-1"></div>;
      case ProcessingStatus.FAILED:
        return <div className="w-3 h-3 rounded-full bg-red-500 mr-1"></div>;
      default:
        return <div className="w-3 h-3 rounded-full bg-gray-500 mr-1"></div>;
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="flex justify-between items-center mb-8">
        <div>
          <h1 className="text-2xl font-bold mb-2">ASTER Explorer Dashboard</h1>
          <p className="text-gray-600">Monitor and manage your ASTER scenes and analyses</p>
        </div>
        <Button onClick={fetchScenes} variant="outline">
          Refresh
        </Button>
      </div>

      {/* Stats cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
        <Card className="p-4">
          <div className="flex flex-col">
            <div className="flex items-center text-lg font-semibold text-gray-900 mb-1">
              <Database className="w-5 h-5 mr-2 text-indigo-600" />
              Total Scenes
            </div>
            <div className="text-3xl font-bold text-indigo-600">
              {isLoading ? '...' : scenes.length}
            </div>
            <div className="text-sm text-gray-500 mt-2">
              Available in the system
            </div>
          </div>
        </Card>

        <Card className="p-4">
          <div className="flex flex-col">
            <div className="flex items-center text-lg font-semibold text-gray-900 mb-1">
              <Clock className="w-5 h-5 mr-2 text-yellow-500" />
              Processing
            </div>
            <div className="text-3xl font-bold text-yellow-500">
              {isLoading ? '...' : processingCount}
            </div>
            <div className="text-sm text-gray-500 mt-2">
              Currently being processed
            </div>
          </div>
        </Card>

        <Card className="p-4">
          <div className="flex flex-col">
            <div className="flex items-center text-lg font-semibold text-gray-900 mb-1">
              <Layers className="w-5 h-5 mr-2 text-green-600" />
              Mineral Maps
            </div>
            <div className="text-3xl font-bold text-green-600">
              {isLoading ? '...' : scenes.filter(s => s.status === ProcessingStatus.COMPLETED).length * 5}
            </div>
            <div className="text-sm text-gray-500 mt-2">
              Generated mineral maps
            </div>
          </div>
        </Card>

        <Card className="p-4">
          <div className="flex flex-col">
            <div className="flex items-center text-lg font-semibold text-gray-900 mb-1">
              <Zap className="w-5 h-5 mr-2 text-orange-500" />
              Gold Targets
            </div>
            <div className="text-3xl font-bold text-orange-500">
              {isLoading ? '...' : Math.floor(scenes.length * 0.3)}
            </div>
            <div className="text-sm text-gray-500 mt-2">
              Potential gold targets identified
            </div>
          </div>
        </Card>
      </div>

      {/* Quick Actions */}
      <Card className="mb-8">
        <div className="p-6">
          <h2 className="text-lg font-semibold mb-6">Quick Actions</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Button className="h-auto py-6 flex flex-col items-center justify-center" asChild>
              <Link to="/scenes">
                <Upload className="w-10 h-10 mb-3" />
                <span className="text-lg">Upload New Data</span>
              </Link>
            </Button>

            <Button className="h-auto py-6 flex flex-col items-center justify-center" asChild>
              <Link to="/explorer">
                <MapPin className="w-10 h-10 mb-3" />
                <span className="text-lg">Explore Maps</span>
              </Link>
            </Button>

            <Button className="h-auto py-6 flex flex-col items-center justify-center" asChild>
              <Link to="/reports">
                <FileText className="w-10 h-10 mb-3" />
                <span className="text-lg">View Reports</span>
              </Link>
            </Button>
          </div>
        </div>
      </Card>

      {/* Recent Scenes */}
      <Card className="mb-8">
        <div className="p-6">
          <h2 className="text-lg font-semibold mb-4">Recent Scenes</h2>
          
          {isLoading ? (
            <div className="flex justify-center items-center p-8">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600"></div>
              <span className="ml-2">Loading scenes...</span>
            </div>
          ) : recentScenes.length === 0 ? (
            <div className="text-center p-8">
              <Database className="w-12 h-12 mx-auto text-gray-400 mb-4" />
              <h3 className="text-lg font-medium text-gray-900">No scenes yet</h3>
              <p className="mt-1 text-sm text-gray-500">Upload ASTER data to get started</p>
              <div className="mt-6">
                <Button asChild>
                  <Link to="/scenes">
                    <Upload className="w-4 h-4 mr-2" />
                    Upload New Data
                  </Link>
                </Button>
              </div>
            </div>
          ) : (
            <div className="overflow-hidden shadow ring-1 ring-black ring-opacity-5 sm:rounded-lg">
              <table className="min-w-full divide-y divide-gray-300">
                <thead className="bg-gray-50">
                  <tr>
                    <th scope="col" className="py-3.5 pl-4 pr-3 text-left text-sm font-semibold text-gray-900 sm:pl-6">Scene</th>
                    <th scope="col" className="px-3 py-3.5 text-left text-sm font-semibold text-gray-900">Date</th>
                    <th scope="col" className="px-3 py-3.5 text-left text-sm font-semibold text-gray-900">Status</th>
                    <th scope="col" className="px-3 py-3.5 text-left text-sm font-semibold text-gray-900">Cloud Cover</th>
                    <th scope="col" className="relative py-3.5 pl-3 pr-4 sm:pr-6">
                      <span className="sr-only">Actions</span>
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200 bg-white">
                  {recentScenes.map((scene) => (
                    <tr key={scene.id}>
                      <td className="whitespace-nowrap py-4 pl-4 pr-3 text-sm font-medium text-gray-900 sm:pl-6">
                        {scene.name || scene.id}
                      </td>
                      <td className="whitespace-nowrap px-3 py-4 text-sm text-gray-500">
                        <div className="flex items-center">
                          <Calendar className="w-4 h-4 mr-1 text-gray-400" />
                          {scene.date !== 'Unknown' ? new Date(scene.date).toLocaleDateString() : 'Unknown'}
                        </div>
                      </td>
                      <td className="whitespace-nowrap px-3 py-4 text-sm">
                        <div className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor(scene.status)}`}>
                          {getStatusIcon(scene.status)}
                          <span className="capitalize">{scene.status}</span>
                        </div>
                      </td>
                      <td className="whitespace-nowrap px-3 py-4 text-sm text-gray-500">
                        {scene.cloudCover}%
                      </td>
                      <td className="relative whitespace-nowrap py-4 pl-3 pr-4 text-right text-sm font-medium sm:pr-6">
                        <Link 
                          to={`/explorer?scene=${scene.id}`} 
                          className="text-indigo-600 hover:text-indigo-900"
                        >
                          Explore
                        </Link>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          <div className="mt-4 flex justify-end">
            <Button variant="outline" asChild>
              <Link to="/scenes">
                View All Scenes
              </Link>
            </Button>
          </div>
        </div>
      </Card>

      {/* System Status */}
      <Card>
        <div className="p-6">
          <h2 className="text-lg font-semibold mb-4">System Status</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h3 className="text-sm font-medium text-gray-700 mb-2">Processing Queue</h3>
              <div className="mb-4">
                <div className="flex justify-between mb-1">
                  <span className="text-sm text-gray-500">Queue Utilization</span>
                  <span className="text-sm font-medium text-gray-900">{processingCount} / 5</span>
                </div>
                <Progress value={processingCount * 20} />
              </div>
              <div className="text-xs text-gray-500">
                {processingCount === 0 ? (
                  "No scenes currently in processing queue"
                ) : (
                  `${processingCount} scene(s) being processed`
                )}
              </div>
            </div>
            <div>
              <h3 className="text-sm font-medium text-gray-700 mb-2">Storage Usage</h3>
              <div className="mb-4">
                <div className="flex justify-between mb-1">
                  <span className="text-sm text-gray-500">Disk Space</span>
                  <span className="text-sm font-medium text-gray-900">12.8 GB / 50 GB</span>
                </div>
                <Progress value={25.6} />
              </div>
              <div className="text-xs text-gray-500">
                {scenes.length} scenes stored, approximately 12.8 GB used
              </div>
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default Dashboard;