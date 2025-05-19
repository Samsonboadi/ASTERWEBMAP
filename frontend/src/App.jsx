// src/App.jsx
import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import { Toaster } from './components/ui/toaster';
import ErrorBoundary from './components/ErrorBoundary';
// Check these imports:
import Explorer from './pages/Explorer';
import Dashboard from './pages/Dashboard';
import ReportViewer from './pages/ReportViewer';
import SceneManager from './pages/SceneManager';
import Settings from './pages/Settings';
import { MapPin, Home, Settings as SettingsIcon, FileText, Database } from 'lucide-react';

const App = () => {
  return (
    <ErrorBoundary>
      <Router>
        <div className="min-h-screen bg-gray-50">
          {/* Top Navigation */}
          <header className="bg-white border-b border-gray-200">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
              <div className="flex justify-between h-16">
                <div className="flex">
                  <div className="flex-shrink-0 flex items-center">
                    <MapPin className="h-8 w-8 text-indigo-600" />
                    <span className="ml-2 text-xl font-bold text-indigo-600">ASTER Web Explorer</span>
                  </div>
                  <nav className="ml-10 flex space-x-8">
                    <Link to="/" className="inline-flex items-center px-1 pt-1 border-b-2 border-indigo-500 text-sm font-medium text-gray-900">
                      <Home className="h-4 w-4 mr-1" />
                      Dashboard
                    </Link>
                    <Link to="/explorer" className="inline-flex items-center px-1 pt-1 border-b-2 border-transparent text-sm font-medium text-gray-500 hover:border-gray-300 hover:text-gray-700">
                      <MapPin className="h-4 w-4 mr-1" />
                      Explorer
                    </Link>
                    <Link to="/scenes" className="inline-flex items-center px-1 pt-1 border-b-2 border-transparent text-sm font-medium text-gray-500 hover:border-gray-300 hover:text-gray-700">
                      <Database className="h-4 w-4 mr-1" />
                      Scenes
                    </Link>
                    <Link to="/reports" className="inline-flex items-center px-1 pt-1 border-b-2 border-transparent text-sm font-medium text-gray-500 hover:border-gray-300 hover:text-gray-700">
                      <FileText className="h-4 w-4 mr-1" />
                      Reports
                    </Link>
                  </nav>
                </div>
                <div className="flex items-center">
                  <Link to="/settings" className="p-2 rounded-full text-gray-400 hover:text-gray-500">
                    <SettingsIcon className="h-6 w-6" />
                  </Link>
                </div>
              </div>
            </div>
          </header>

          {/* Main Content */}
          <main className="flex-1">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/explorer" element={<Explorer />} />
              <Route path="/scenes" element={<SceneManager />} />
              <Route path="/reports" element={<ReportViewer />} />
              <Route path="/settings" element={<Settings />} />
            </Routes>
          </main>

          {/* Toast notifications */}
          <Toaster />
        </div>
      </Router>
    </ErrorBoundary>
  );
};

export default App;