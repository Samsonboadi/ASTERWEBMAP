// frontend/src/components/ErrorBoundary.jsx

import React from 'react';
import { AlertCircle, RefreshCw } from 'lucide-react';
import { Button } from '../components/ui';

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error) {
    // Update state so the next render will show the fallback UI
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    // Log the error to an error reporting service
    console.error('Error caught by ErrorBoundary:', error, errorInfo);
    this.setState({ errorInfo });
  }

  handleReload = () => {
    window.location.reload();
  };

  handleGoHome = () => {
    window.location.href = '/';
  };

  render() {
    if (this.state.hasError) {
      // You can render any custom fallback UI
      return (
        <div className="min-h-screen flex flex-col items-center justify-center bg-gray-50 p-4">
          <div className="bg-white p-6 rounded-lg shadow-md max-w-lg w-full">
            <div className="flex items-center justify-center text-red-600 mb-4">
              <AlertCircle size={48} />
            </div>
            <h1 className="text-2xl font-bold text-gray-900 text-center mb-4">
              Something went wrong
            </h1>
            <div className="bg-gray-100 p-4 rounded-md mb-4 overflow-auto max-h-40">
              <p className="text-sm text-gray-800 font-mono">
                {this.state.error && this.state.error.toString()}
              </p>
            </div>
            <div className="flex flex-col sm:flex-row gap-2 justify-center">
              <Button onClick={this.handleReload} className="flex-1">
                <RefreshCw className="w-4 h-4 mr-2" />
                Reload Page
              </Button>
              <Button onClick={this.handleGoHome} variant="outline" className="flex-1">
                Go to Home
              </Button>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children; 
  }
}

export default ErrorBoundary;