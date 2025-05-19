// src/pages/Settings.jsx
import React from 'react';
import { Card, Button, Separator } from '../components/ui';
import { Settings as SettingsIcon, Database, Map, UploadCloud, BarChart } from 'lucide-react';

const Settings = () => {
  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="flex justify-between items-center mb-8">
        <div>
          <h1 className="text-2xl font-bold mb-2">Settings</h1>
          <p className="text-gray-600">Configure application preferences and connection settings</p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="md:col-span-1">
          <Card className="p-4">
            <div className="space-y-1">
              <button className="w-full flex items-center px-3 py-2 text-sm font-medium rounded-md bg-indigo-50 text-indigo-800">
                <Database className="mr-2 h-5 w-5" />
                Data Management
              </button>
              <button className="w-full flex items-center px-3 py-2 text-sm font-medium rounded-md text-gray-600 hover:bg-gray-50">
                <Map className="mr-2 h-5 w-5" />
                Map Settings
              </button>
              <button className="w-full flex items-center px-3 py-2 text-sm font-medium rounded-md text-gray-600 hover:bg-gray-50">
                <UploadCloud className="mr-2 h-5 w-5" />
                Upload Configuration
              </button>
              <button className="w-full flex items-center px-3 py-2 text-sm font-medium rounded-md text-gray-600 hover:bg-gray-50">
                <BarChart className="mr-2 h-5 w-5" />
                Analysis Preferences
              </button>
              <button className="w-full flex items-center px-3 py-2 text-sm font-medium rounded-md text-gray-600 hover:bg-gray-50">
                <SettingsIcon className="mr-2 h-5 w-5" />
                General Settings
              </button>
            </div>
          </Card>
        </div>

        <div className="md:col-span-2">
          <Card className="p-6">
            <h2 className="text-lg font-semibold mb-4">Data Management</h2>
            <Separator className="my-4" />
            
            <div className="space-y-4">
              <div>
                <h3 className="text-sm font-medium mb-1">Data Storage Location</h3>
                <div className="flex items-center mt-1">
                  <input
                    type="text"
                    value="/app/data"
                    readOnly
                    className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                  />
                  <Button variant="outline" className="ml-2">
                    Browse
                  </Button>
                </div>
              </div>

              <div>
                <h3 className="text-sm font-medium mb-1">Data Retention Policy</h3>
                <select
                  className="block w-full mt-1 px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                >
                  <option>Keep data for 30 days</option>
                  <option>Keep data for 60 days</option>
                  <option>Keep data for 90 days</option>
                  <option>Keep data indefinitely</option>
                </select>
              </div>

              <div>
                <h3 className="text-sm font-medium mb-1">Data Backup</h3>
                <div className="flex items-center mt-1">
                  <input
                    type="checkbox"
                    className="h-4 w-4 text-indigo-600 border-gray-300 rounded focus:ring-indigo-500"
                  />
                  <label className="ml-2 block text-sm text-gray-900">
                    Enable automatic daily backups
                  </label>
                </div>
              </div>

              <div>
                <h3 className="text-sm font-medium mb-1">Storage Usage</h3>
                <div className="bg-gray-200 rounded-full h-2.5 mb-1">
                  <div className="bg-indigo-600 h-2.5 rounded-full w-[45%]"></div>
                </div>
                <div className="text-xs text-gray-500">
                  12.8 GB used of 50 GB (25.6%)
                </div>
              </div>
              
              <div className="flex justify-end pt-4">
                <Button variant="outline" className="mr-2">
                  Reset to Default
                </Button>
                <Button>
                  Save Changes
                </Button>
              </div>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
};

// Add the default export
export default Settings;