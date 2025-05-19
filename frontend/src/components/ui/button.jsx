// src/components/ui/button.jsx
import React from 'react';

export const Button = ({ children, variant = 'default', size = 'default', className = '', asChild, ...props }) => {
  const Comp = asChild ? props.as || 'div' : 'button';
  
  const getVariantClasses = () => {
    switch (variant) {
      case 'outline':
        return 'bg-white text-gray-700 border border-gray-300 hover:bg-gray-50';
      case 'destructive':
        return 'bg-red-600 text-white hover:bg-red-700';
      default:
        return 'bg-indigo-600 text-white hover:bg-indigo-700';
    }
  };
  
  const getSizeClasses = () => {
    switch (size) {
      case 'sm':
        return 'px-2 py-1 text-sm';
      case 'lg':
        return 'px-4 py-3 text-lg';
      default:
        return 'px-3 py-2 text-sm';
    }
  };
  
  const classes = `inline-flex items-center justify-center rounded-md font-medium focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 transition-colors ${getVariantClasses()} ${getSizeClasses()} ${className}`;
  
  return (
    <Comp className={classes} {...props}>
      {children}
    </Comp>
  );
};