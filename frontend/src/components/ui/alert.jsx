// src/components/ui/alert.jsx
import React from 'react';

export const Alert = ({ children, variant = 'default', className = '', ...props }) => {
  const getVariantClasses = () => {
    switch (variant) {
      case 'destructive':
        return 'bg-red-50 border-red-200 text-red-800';
      case 'warning':
        return 'bg-yellow-50 border-yellow-200 text-yellow-800';
      case 'success':
        return 'bg-green-50 border-green-200 text-green-800';
      default:
        return 'bg-blue-50 border-blue-200 text-blue-800';
    }
  };

  return (
    <div
      className={`p-4 rounded-md border ${getVariantClasses()} ${className}`}
      role="alert"
      {...props}
    >
      {children}
    </div>
  );
};

export const AlertTitle = ({ children, className = '', ...props }) => (
  <h5 className={`font-medium mb-1 text-sm flex items-center ${className}`} {...props}>
    {children}
  </h5>
);

export const AlertDescription = ({ children, className = '', ...props }) => (
  <div className={`text-sm ${className}`} {...props}>
    {children}
  </div>
);