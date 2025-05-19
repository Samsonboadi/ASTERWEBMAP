// src/pages/ReportViewer.jsx
import React, { useState, useEffect } from 'react';
import { Card, Button } from '../components/ui';
import { FileText, Download, Search, Filter, Calendar, ChevronRight } from 'lucide-react';

const ReportViewer = () => {
  const [reports, setReports] = useState([]);
  const [selectedReport, setSelectedReport] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  // This would normally fetch reports from the API
  useEffect(() => {
    setIsLoading(true);
    
    // Mock data for development
    setTimeout(() => {
      setReports([
        {
          id: 'report1',
          title: 'Comprehensive Analysis - Scene 001',
          date: new Date('2023-07-15'),
          type: 'comprehensive',
          url: '#'
        },
        {
          id: 'report2',
          title: 'Mineral Distribution Report - Scene 002',
          date: new Date('2023-07-10'),
          type: 'mineral_distribution',
          url: '#'
        },
        {
          id: 'report3',
          title: 'Alteration Analysis - Scene 003',
          date: new Date('2023-07-05'),
          type: 'alteration_analysis',
          url: '#'
        }
      ]);
      setIsLoading(false);
    }, 1000);
  }, []);

  const handleViewReport = (report) => {
    setSelectedReport(report);
    // In a real app, this would open the report in an iframe or new window
    window.open(report.url, '_blank');
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="flex justify-between items-center mb-8">
        <div>
          <h1 className="text-2xl font-bold mb-2">Analysis Reports</h1>
          <p className="text-gray-600">View and download analysis reports for your ASTER scenes</p>
        </div>
        <div className="flex space-x-2">
          <Button variant="outline">
            <Filter className="w-4 h-4 mr-2" />
            Filter
          </Button>
          <Button variant="outline">
            <Search className="w-4 h-4 mr-2" />
            Search
          </Button>
        </div>
      </div>

      {isLoading ? (
        <div className="flex justify-center items-center p-12">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-500"></div>
        </div>
      ) : reports.length === 0 ? (
        <Card className="p-12 text-center">
          <FileText className="w-16 h-16 mx-auto text-gray-400 mb-4" />
          <h2 className="text-xl font-semibold text-gray-700 mb-2">No Reports Available</h2>
          <p className="text-gray-500 mb-6">Process some ASTER data to generate analysis reports.</p>
          <Button>
            Go to Explorer
          </Button>
        </Card>
      ) : (
        <div className="space-y-4">
          {reports.map(report => (
            <Card key={report.id} className="p-6">
              <div className="flex justify-between items-center">
                <div>
                  <h3 className="text-lg font-semibold mb-1">{report.title}</h3>
                  <div className="flex items-center text-sm text-gray-500 space-x-4">
                    <div className="flex items-center">
                      <Calendar className="w-4 h-4 mr-1" />
                      {report.date.toLocaleDateString()}
                    </div>
                    <div>
                      Type: {report.type.replace('_', ' ')}
                    </div>
                  </div>
                </div>
                <div className="flex space-x-2">
                  <Button variant="outline" onClick={() => handleViewReport(report)}>
                    <FileText className="w-4 h-4 mr-2" />
                    View
                  </Button>
                  <Button variant="outline">
                    <Download className="w-4 h-4 mr-2" />
                    Download
                  </Button>
                </div>
              </div>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
};

export default ReportViewer;