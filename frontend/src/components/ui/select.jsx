// src/components/ui/select.jsx
import React, { useState, useRef, useEffect } from 'react';

export const Select = ({ children, value, onValueChange, disabled = false }) => {
  const [isOpen, setIsOpen] = useState(false);
  const ref = useRef(null);
  
  // Close the select when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (ref.current && !ref.current.contains(event.target)) {
        setIsOpen(false);
      }
    };
    
    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [ref]);
  
  return (
    <div ref={ref} className="relative">
      {React.Children.map(children, (child) => {
        return React.cloneElement(child, {
          isOpen,
          setIsOpen,
          value,
          onValueChange,
          disabled
        });
      })}
    </div>
  );
};

export const SelectTrigger = ({ children, isOpen, setIsOpen, disabled }) => {
  return (
    <button
      type="button"
      className={`w-full flex items-center justify-between rounded-md border border-gray-300 bg-white px-3 py-2 text-sm shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer hover:bg-gray-50'}`}
      onClick={() => !disabled && setIsOpen(!isOpen)}
      disabled={disabled}
    >
      {children}
      <svg 
        className={`h-5 w-5 text-gray-400 ${isOpen ? 'transform rotate-180' : ''}`} 
        xmlns="http://www.w3.org/2000/svg" 
        viewBox="0 0 20 20" 
        fill="currentColor"
      >
        <path fillRule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd" />
      </svg>
    </button>
  );
};

export const SelectValue = ({ placeholder, children }) => {
  return <span>{children || <span className="text-gray-400">{placeholder}</span>}</span>;
};

export const SelectContent = ({ children, isOpen }) => {
  if (!isOpen) return null;
  
  return (
    <div className="absolute z-10 mt-1 w-full rounded-md bg-white shadow-lg">
      <ul className="max-h-60 overflow-auto rounded-md py-1 text-base ring-1 ring-black ring-opacity-5 focus:outline-none sm:text-sm">
        {children}
      </ul>
    </div>
  );
};

export const SelectItem = ({ children, value, onValueChange, activeValue }) => {
  const isSelected = value === activeValue;
  
  return (
    <li
      className={`relative cursor-pointer select-none py-2 pl-3 pr-9 text-gray-900 hover:bg-indigo-50 ${isSelected ? 'bg-indigo-100' : ''}`}
      onClick={() => onValueChange && onValueChange(value)}
    >
      <span className="block truncate">{children}</span>
      
      {isSelected && (
        <span className="absolute inset-y-0 right-0 flex items-center pr-4 text-indigo-600">
          <svg className="h-5 w-5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
            <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
          </svg>
        </span>
      )}
    </li>
  );
};