// src/components/ui/accordion.jsx
import React, { useState, useContext, createContext } from 'react';

// Create context for accordion state
const AccordionContext = createContext({
  expandedValue: null,
  setExpandedValue: () => {},
  type: 'single'
});

export const Accordion = ({ 
  children, 
  type = 'single', 
  collapsible = false, 
  defaultValue = null, 
  className = '', 
  ...props 
}) => {
  const [expandedValue, setExpandedValue] = useState(defaultValue);

  const handleValueChange = (value) => {
    if (type === 'single') {
      if (collapsible && expandedValue === value) {
        setExpandedValue(null);
      } else {
        setExpandedValue(value);
      }
    }
  };

  return (
    <AccordionContext.Provider value={{ expandedValue, setExpandedValue: handleValueChange, type }}>
      <div className={`space-y-1 ${className}`} {...props}>
        {children}
      </div>
    </AccordionContext.Provider>
  );
};

export const AccordionItem = ({ children, value, className = '', ...props }) => {
  return (
    <div className={`border rounded-md overflow-hidden ${className}`} {...props}>
      {React.Children.map(children, child => 
        React.cloneElement(child, { value })
      )}
    </div>
  );
};

export const AccordionTrigger = ({ children, className = '', value, ...props }) => {
  const { expandedValue, setExpandedValue } = useContext(AccordionContext);
  const isExpanded = expandedValue === value;

  return (
    <button
      className={`flex w-full justify-between px-4 py-3 text-left font-medium text-gray-900 hover:bg-gray-50 ${className}`}
      onClick={() => setExpandedValue(value)}
      aria-expanded={isExpanded}
      {...props}
    >
      {children}
      <svg
        xmlns="http://www.w3.org/2000/svg"
        className={`h-5 w-5 text-gray-500 transition-transform duration-200 ${isExpanded ? 'rotate-180' : ''}`}
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
      >
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
      </svg>
    </button>
  );
};

export const AccordionContent = ({ children, className = '', value, ...props }) => {
  const { expandedValue } = useContext(AccordionContext);
  const isExpanded = expandedValue === value;

  if (!isExpanded) {
    return null;
  }

  return (
    <div 
      className={`px-4 pb-4 pt-0 ${className}`}
      {...props}
    >
      {children}
    </div>
  );
};