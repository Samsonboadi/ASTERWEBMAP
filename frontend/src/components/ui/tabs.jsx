// src/components/ui/tabs.jsx
import React, { useState, createContext, useContext } from 'react';

const TabsContext = createContext({
  value: null,
  onValueChange: () => {},
});

export const Tabs = ({ 
  children, 
  defaultValue = null, 
  value, 
  onValueChange,
  className = '',
  ...props 
}) => {
  const [internalValue, setInternalValue] = useState(defaultValue);
  
  const contextValue = {
    value: value !== undefined ? value : internalValue,
    onValueChange: onValueChange || setInternalValue,
  };

  return (
    <TabsContext.Provider value={contextValue}>
      <div className={className} {...props}>
        {children}
      </div>
    </TabsContext.Provider>
  );
};

export const TabsList = ({ children, className = '', ...props }) => {
  return (
    <div 
      className={`flex space-x-1 rounded-lg bg-gray-100 p-1 ${className}`}
      role="tablist"
      {...props}
    >
      {children}
    </div>
  );
};

export const TabsTrigger = ({ children, value, className = '', ...props }) => {
  const { value: selectedValue, onValueChange } = useContext(TabsContext);
  const isSelected = selectedValue === value;

  return (
    <button
      role="tab"
      aria-selected={isSelected}
      className={`px-3 py-2 text-sm font-medium rounded-md focus:outline-none transition-colors ${
        isSelected 
          ? 'bg-white text-gray-900 shadow-sm' 
          : 'text-gray-600 hover:text-gray-900'
      } ${className}`}
      onClick={() => onValueChange(value)}
      {...props}
    >
      {children}
    </button>
  );
};

export const TabsContent = ({ children, value, className = '', ...props }) => {
  const { value: selectedValue } = useContext(TabsContext);
  
  if (selectedValue !== value) {
    return null;
  }

  return (
    <div 
      role="tabpanel"
      className={`mt-2 ${className}`}
      {...props}
    >
      {children}
    </div>
  );
};