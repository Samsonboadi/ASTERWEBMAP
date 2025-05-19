// src/components/ui/progress.jsx
import React from 'react';

export const Progress = ({ value = 0, max = 100, className = '', ...props }) => {
  const percentage = Math.min(100, Math.max(0, (value / max) * 100));
  
  return (
    <div className={`w-full h-2 bg-gray-200 rounded-full overflow-hidden ${className}`} {...props}>
      <div 
        className="h-full bg-indigo-600 transition-all duration-300 ease-in-out"
        style={{ width: `${percentage}%` }}
        role="progressbar"
        aria-valuenow={value}
        aria-valuemin="0"
        aria-valuemax={max}
      />
    </div>
  );
};