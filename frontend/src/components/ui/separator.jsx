// src/components/ui/separator.jsx
import React from 'react';

export const Separator = ({ orientation = 'horizontal', className = '', ...props }) => {
  return (
    <div 
      className={`${orientation === 'horizontal' ? 'h-px w-full' : 'h-full w-px'} bg-gray-200 ${className}`}
      {...props}
    />
  );
};