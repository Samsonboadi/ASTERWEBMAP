// src/components/ui/toast.jsx
import React from 'react';

export const Toast = ({ children, variant = 'default', className = '', ...props }) => {
  const getVariantClasses = () => {
    switch (variant) {
      case 'destructive':
        return 'bg-red-600 text-white';
      case 'success':
        return 'bg-green-600 text-white';
      default:
        return 'bg-white text-gray-800 border border-gray-200';
    }
  };

  return (
    <div
      className={`rounded-md shadow-lg p-4 ${getVariantClasses()} ${className}`}
      {...props}
    >
      {children}
    </div>
  );
};

export const ToastProvider = ({ children }) => {
  return <>{children}</>;
};

export const ToastViewport = ({ className = '', ...props }) => {
  return (
    <div
      className={`fixed top-0 right-0 flex flex-col p-4 gap-2 w-auto max-w-[420px] z-50 ${className}`}
      {...props}
    />
  );
};