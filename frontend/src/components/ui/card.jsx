// src/components/ui/card.jsx
import React from 'react';

export const Card = ({ children, className = '', ...props }) => {
  return (
    <div 
      className={`bg-white rounded-lg shadow border border-gray-100 ${className}`}
      {...props}
    >
      {children}
    </div>
  );
};