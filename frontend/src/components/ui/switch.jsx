// src/components/ui/switch.jsx
import React from 'react';

export const Switch = ({ 
  checked = false, 
  onCheckedChange, 
  disabled = false, 
  className = '', 
  ...props 
}) => {
  const handleChange = (e) => {
    if (onCheckedChange) {
      onCheckedChange(e.target.checked);
    }
  };
  
  return (
    <label className={`relative inline-flex items-center cursor-pointer ${disabled ? 'opacity-50 cursor-not-allowed' : ''} ${className}`}>
      <input
        type="checkbox"
        className="sr-only"
        checked={checked}
        onChange={handleChange}
        disabled={disabled}
        {...props}
      />
      <div className={`relative w-11 h-6 bg-gray-200 rounded-full transition-colors ${checked ? 'bg-indigo-600' : ''}`}>
        <div className={`absolute left-1 top-1 w-4 h-4 bg-white rounded-full transition-transform ${checked ? 'transform translate-x-5' : ''}`}></div>
      </div>
    </label>
  );
};